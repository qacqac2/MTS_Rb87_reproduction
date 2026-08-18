import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from sympy import S
from sympy.physics.wigner import clebsch_gordan


# ============================================================
# Stage D-2A
# 87Rb D2 F=2 -> F'=3
# Zeeman-resolved optical Bloch equations
# + optical pumping
# + intensity / power dependence
# ============================================================
#
# 这一阶段的目标不是直接生成最终 MTS。
#
# 它建立后续完整 MTS Floquet/sideband 模型所需要的
# “原子动力学核心”：
#
#   1. mF Zeeman 子能级
#   2. Clebsch-Gordan 系数
#   3. 偏振选择规则
#   4. spontaneous-emission redistribution
#   5. optical pumping
#   6. laser intensity -> Rabi frequency
#   7. saturation
#   8. power broadening
#
# 下一阶段再把：
#
#   phase-modulated pump
#   + weak probe
#   + modulation harmonics
#
# 加到同一个 OBE 框架中。
# ============================================================


# ============================================================
# 0. 原子参数
# ============================================================

Fg = 2
Fe = 3

mg_values = np.arange(
    -Fg,
    Fg + 1
)

me_values = np.arange(
    -Fe,
    Fe + 1
)


# ------------------------------------------------------------
# 87Rb D2 natural linewidth
#
# Gamma / (2*pi) ~ 6.065 MHz
#
# 内部 Hamiltonian / Liouvillian 全部使用
# angular-frequency units。
# ------------------------------------------------------------

GAMMA_MHZ = 6.065

TWOPI = 2.0 * np.pi

GAMMA = (
    TWOPI
    * GAMMA_MHZ
)


# ------------------------------------------------------------
# F=2,m=+/-2 -> F'=3,m'=+/-3
# cycling transition steady-state saturation intensity
#
# 这里用它把“实际光强”映射成电场 / Rabi frequency。
# ------------------------------------------------------------

ISAT_CYCLING = 1.66933      # mW/cm^2


# ------------------------------------------------------------
# Steck 给出的稳态有效饱和强度：
#
# pi polarization:
# ~3.05381 mW/cm^2
#
# sigma+ cycling after optical pumping:
# ~1.66933 mW/cm^2
#
# 这个量只用于结果交叉检查，
# Hamiltonian 本身仍然从 CG + E-field 计算。
# ------------------------------------------------------------

ISAT_PI_REFERENCE = 3.05381


# ============================================================
# 1. 光偏振
# ============================================================
#
# 可选：
#
#   "pi"
#   "sigma+"
#   "sigma-"
#
# pi：
#   Delta m = 0
#
# sigma+：
#   Delta m = +1
#
# sigma-：
#   Delta m = -1
# ============================================================

POLARIZATION = "pi"


def polarization_components(
    polarization
):
    """
    spherical polarization components epsilon_q

    q = -1, 0, +1
    """

    if polarization == "pi":

        return {
            -1: 0.0,
             0: 1.0,
            +1: 0.0
        }

    elif polarization == "sigma+":

        return {
            -1: 0.0,
             0: 0.0,
            +1: 1.0
        }

    elif polarization == "sigma-":

        return {
            -1: 1.0,
             0: 0.0,
            +1: 0.0
        }

    else:

        raise ValueError(
            "POLARIZATION must be "
            "'pi', 'sigma+' or 'sigma-'"
        )


EPS = polarization_components(
    POLARIZATION
)


# ============================================================
# 2. 磁场
# ============================================================
#
# 先设 B=0：
#
#   Stage D-2A 主要研究 power / optical pumping。
#
# 后面可直接改成例如：
#
#   B_GAUSS = 0.1
#
# 研究 Zeeman splitting。
# ============================================================

B_GAUSS = 0.0


# ------------------------------------------------------------
# low-field approximate gF
#
# 对本阶段 B=0 不影响结果。
# ------------------------------------------------------------

GF_G = 0.5
GF_E = 2.0 / 3.0


# Bohr magneton / h
# approximately MHz/G

MU_B_OVER_H_MHZ_PER_G = 1.3996246


# ============================================================
# 3. Hilbert space
# ============================================================

NG = len(
    mg_values
)

NE = len(
    me_values
)

N = (
    NG + NE
)


def g_index(
    mg
):
    return int(
        mg + Fg
    )


def e_index(
    me
):
    return (
        NG
        + int(
            me + Fe
        )
    )


basis = [
    qt.basis(
        N,
        i
    )
    for i in range(N)
]


# ============================================================
# 4. Clebsch-Gordan coefficient
# ============================================================

def cg_coefficient(
    mg,
    q
):
    """
    <Fg,mg; 1,q | Fe,me>

    me = mg + q
    """

    me = (
        mg + q
    )

    if (
        me < -Fe
        or me > Fe
    ):
        return 0.0


    value = clebsch_gordan(

        S(Fg),
        S(1),
        S(Fe),

        S(int(mg)),
        S(int(q)),
        S(int(me))
    )


    return float(
        value
    )


# ============================================================
# 5. 输出完整 CG 表
# ============================================================

CG_TABLE = {}


for mg in mg_values:

    for q in [
        -1,
        0,
        +1
    ]:

        me = (
            mg + q
        )

        if (
            me < -Fe
            or me > Fe
        ):
            continue

        cg = cg_coefficient(
            mg,
            q
        )

        if abs(cg) > 1e-14:

            CG_TABLE[
                (
                    int(mg),
                    q,
                    int(me)
                )
            ] = cg


# ============================================================
# 6. 原子 raising operator
#
# D_plus =
#
# sum epsilon_q * C_mq |e><g|
# ============================================================

D_PLUS = (
    0
    * qt.qeye(N)
)


for (
    mg,
    q,
    me
), cg in CG_TABLE.items():

    eps = EPS[q]

    if abs(eps) == 0:
        continue

    ket_e = basis[
        e_index(me)
    ]

    bra_g = basis[
        g_index(mg)
    ].dag()


    D_PLUS += (

        eps
        * cg

        * ket_e
        * bra_g
    )


D_MINUS = (
    D_PLUS.dag()
)


# ============================================================
# 7. 投影算符
# ============================================================

GROUND_PROJECTORS = {}

EXCITED_PROJECTORS = {}


for mg in mg_values:

    ket = basis[
        g_index(mg)
    ]

    GROUND_PROJECTORS[
        int(mg)
    ] = (
        ket
        * ket.dag()
    )


for me in me_values:

    ket = basis[
        e_index(me)
    ]

    EXCITED_PROJECTORS[
        int(me)
    ] = (
        ket
        * ket.dag()
    )


P_GROUND = sum(
    GROUND_PROJECTORS.values()
)

P_EXCITED = sum(
    EXCITED_PROJECTORS.values()
)


# ============================================================
# 8. spontaneous-emission collapse operators
# ============================================================
#
# 对 F'=3 -> F=2：
#
# 每个 excited mF'
# 可以通过 q = -1,0,+1
# 衰变到相应 ground mF。
#
# decay rate:
#
# Gamma * |CG|^2
# ============================================================

C_OPS = []

BRANCHING_CHECK = {}


for me in me_values:

    total_branch = 0.0

    for mg in mg_values:

        q = int(
            me - mg
        )

        if q not in [
            -1,
            0,
            +1
        ]:
            continue


        cg = cg_coefficient(
            mg,
            q
        )

        probability = (
            cg**2
        )


        if probability < 1e-14:
            continue


        total_branch += (
            probability
        )


        ket_g = basis[
            g_index(mg)
        ]

        bra_e = basis[
            e_index(me)
        ].dag()


        collapse = (

            np.sqrt(
                GAMMA
                * probability
            )

            * ket_g
            * bra_e
        )


        C_OPS.append(
            collapse
        )


    BRANCHING_CHECK[
        int(me)
    ] = total_branch


# ============================================================
# 9. Rabi frequency from intensity
# ============================================================

def cycling_rabi_from_intensity(
    intensity_mW_cm2
):
    """
    Steck convention:

        I / Isat = 2 * (Omega / Gamma)^2

    hence:

        Omega =
            Gamma * sqrt(
                I / (2 Isat)
            )

    Omega is angular frequency.
    """

    s = (

        intensity_mW_cm2

        /

        ISAT_CYCLING
    )


    return (

        GAMMA

        * np.sqrt(
            s / 2.0
        )
    )


# ============================================================
# 10. Zeeman Hamiltonian
# ============================================================

def zeeman_hamiltonian():
    """
    angular-frequency units
    """

    H = (
        0
        * qt.qeye(N)
    )


    for mg in mg_values:

        shift_MHz = (

            GF_G
            * MU_B_OVER_H_MHZ_PER_G
            * B_GAUSS
            * mg
        )


        H += (

            TWOPI
            * shift_MHz

            * GROUND_PROJECTORS[
                int(mg)
            ]
        )


    for me in me_values:

        shift_MHz = (

            GF_E
            * MU_B_OVER_H_MHZ_PER_G
            * B_GAUSS
            * me
        )


        H += (

            TWOPI
            * shift_MHz

            * EXCITED_PROJECTORS[
                int(me)
            ]
        )


    return H


H_ZEEMAN = (
    zeeman_hamiltonian()
)


# ============================================================
# 11. rotating-frame Hamiltonian
# ============================================================

def hamiltonian(
    detuning_MHz,
    intensity_mW_cm2
):
    """
    Delta =
        laser frequency
        - atomic resonance

    rotating-frame excited energy:

        -Delta |e><e|
    """

    Delta = (

        TWOPI
        * detuning_MHz
    )


    Omega_cyc = (
        cycling_rabi_from_intensity(
            intensity_mW_cm2
        )
    )


    H_detuning = (

        -Delta
        * P_EXCITED
    )


    H_interaction = (

        0.5
        * Omega_cyc

        * (
            D_PLUS
            + D_MINUS
        )
    )


    return (

        H_ZEEMAN
        + H_detuning
        + H_interaction
    )


# ============================================================
# 12. steady-state OBE
# ============================================================

def steady_state(
    detuning_MHz,
    intensity_mW_cm2
):

    H = hamiltonian(
        detuning_MHz,
        intensity_mW_cm2
    )


    rho_ss = qt.steadystate(

        H,
        C_OPS,

        method="direct"
    )


    return rho_ss


# ============================================================
# 13. observable extraction
# ============================================================

def observables(
    detuning_MHz,
    intensity_mW_cm2
):

    rho = steady_state(
        detuning_MHz,
        intensity_mW_cm2
    )


    excited_fraction = float(
        np.real(
            qt.expect(
                P_EXCITED,
                rho
            )
        )
    )


    # --------------------------------------------------------
    # complex optical polarization
    #
    # overall density / dipole constants omitted.
    #
    # Re(P) -> dispersion proxy
    # Im(P) -> absorption proxy
    # --------------------------------------------------------

    polarization = complex(
        qt.expect(
            D_PLUS,
            rho
        )
    )


    ground_pop = {}


    for mg in mg_values:

        ground_pop[
            int(mg)
        ] = float(
            np.real(
                qt.expect(
                    GROUND_PROJECTORS[
                        int(mg)
                    ],
                    rho
                )
            )
        )


    return (
        excited_fraction,
        polarization,
        ground_pop
    )


# ============================================================
# 14. intensity choices for full spectra
# ============================================================

INTENSITIES = np.array([
    0.03,
    0.10,
    0.30,
    1.00,
    3.00,
    10.00
])


DETUNING = np.linspace(
    -50.0,
    +50.0,
    601
)


# ============================================================
# 15. spectrum calculation
# ============================================================

spectra = {}


print()
print(
    "Calculating Zeeman-resolved OBE spectra..."
)


for intensity in INTENSITIES:

    print(
        f"  I = {intensity:.3f} mW/cm^2"
    )


    excited = np.zeros_like(
        DETUNING
    )

    polarization = np.zeros(
        len(DETUNING),
        dtype=np.complex128
    )


    for j, det in enumerate(
        DETUNING
    ):

        (
            excited[j],
            polarization[j],
            _
        ) = observables(
            det,
            intensity
        )


    spectra[
        intensity
    ] = {

        "excited":
            excited,

        "polarization":
            polarization,

        "absorption":
            np.imag(
                polarization
            ),

        "dispersion":
            np.real(
                polarization
            )
    }


# ============================================================
# 16. ensure absorption sign is positive
# ============================================================

reference_abs = spectra[
    INTENSITIES[-1]
]["absorption"]


center_index = np.argmin(
    np.abs(
        DETUNING
    )
)


if reference_abs[
    center_index
] < 0:

    for intensity in INTENSITIES:

        spectra[
            intensity
        ]["absorption"] *= -1.0


# ============================================================
# 17. FWHM extraction
# ============================================================

def fwhm(x, y):

    ymax = np.max(y)
    half = 0.5 * ymax

    imax = np.argmax(y)

    left = np.where(
        y[:imax] <= half
    )[0]

    right = np.where(
        y[imax:] <= half
    )[0]

    if len(left) == 0 or len(right) == 0:
        return np.nan

    il = left[-1]
    ir = imax + right[0]

    x_left = np.interp(
        half,
        [y[il], y[il + 1]],
        [x[il], x[il + 1]]
    )

    x_right = np.interp(
        half,
        [y[ir], y[ir - 1]],
        [x[ir], x[ir - 1]]
    )

    return x_right - x_left


FWHM_VALUES = []


for intensity in INTENSITIES:

    width = fwhm(

        DETUNING,

        spectra[
            intensity
        ]["excited"]
    )


    FWHM_VALUES.append(
        width
    )


FWHM_VALUES = np.array(
    FWHM_VALUES
)


# ============================================================
# 18. continuous power scan
# ============================================================

POWER_SCAN = np.logspace(

    -2.0,
    np.log10(20.0),

    30
)


excited_resonance = []

dispersion_slope = []

ground_population_scan = {
    int(mg): []
    for mg in mg_values
}


# finite-difference detuning step
SLOPE_STEP_MHZ = 0.10


print()
print(
    "Calculating resonant power scan..."
)


for intensity in POWER_SCAN:

    (
        excited_0,
        P0,
        ground_0
    ) = observables(
        0.0,
        intensity
    )


    (
        _,
        P_plus,
        _
    ) = observables(
        +SLOPE_STEP_MHZ,
        intensity
    )


    (
        _,
        P_minus,
        _
    ) = observables(
        -SLOPE_STEP_MHZ,
        intensity
    )


    slope = (

        np.real(
            P_plus
        )

        -

        np.real(
            P_minus
        )

    ) / (

        2.0
        * SLOPE_STEP_MHZ
    )


    excited_resonance.append(
        excited_0
    )

    dispersion_slope.append(
        abs(
            slope
        )
    )


    for mg in mg_values:

        ground_population_scan[
            int(mg)
        ].append(

            ground_0[
                int(mg)
            ]
        )


excited_resonance = np.array(
    excited_resonance
)

dispersion_slope = np.array(
    dispersion_slope
)


for mg in mg_values:

    ground_population_scan[
        int(mg)
    ] = np.array(

        ground_population_scan[
            int(mg)
        ]
    )


# ============================================================
# 19. effective saturation intensity
#
# For the usual saturated two-level steady state:
#
# rho_ee = 1/4
#
# at s = 1.
#
# Therefore use rho_ee=0.25 as a practical
# numerical definition of effective Isat.
# ============================================================

def interpolate_intensity_at_population(
    intensity,
    population,
    target=0.25
):

    diff = (
        population
        - target
    )


    idx = np.where(

        diff[:-1]
        * diff[1:]
        <= 0

    )[0]


    if len(idx) == 0:

        return np.nan


    i = idx[0]


    # interpolate in log intensity
    lx1 = np.log(
        intensity[i]
    )

    lx2 = np.log(
        intensity[i + 1]
    )


    y1 = population[i]
    y2 = population[i + 1]


    if y2 == y1:

        return np.exp(
            0.5
            * (
                lx1 + lx2
            )
        )


    lx = (

        lx1

        +

        (
            target - y1
        )

        * (
            lx2 - lx1
        )

        / (
            y2 - y1
        )
    )


    return np.exp(
        lx
    )


ISAT_NUMERICAL = (
    interpolate_intensity_at_population(

        POWER_SCAN,
        excited_resonance,

        target=0.25
    )
)


# ============================================================
# 20. two-level power-broadening reference
# ============================================================

if POLARIZATION == "pi":

    ISAT_REFERENCE = (
        ISAT_PI_REFERENCE
    )

else:

    ISAT_REFERENCE = (
        ISAT_CYCLING
    )


I_REFERENCE_LINE = np.logspace(
    -2,
    np.log10(20.0),
    300
)


FWHM_TWO_LEVEL = (

    GAMMA_MHZ

    * np.sqrt(

        1.0

        +

        I_REFERENCE_LINE
        / ISAT_REFERENCE
    )
)


# ============================================================
# Figure 1
# CG coupling strengths
# ============================================================

coupling_mg = []
coupling_strength = []


for mg in mg_values:

    strength = 0.0

    for q in [
        -1,
        0,
        +1
    ]:

        eps = EPS[q]

        if abs(eps) == 0:
            continue

        cg = cg_coefficient(
            mg,
            q
        )

        strength += (

            abs(eps)**2
            * cg**2
        )


    coupling_mg.append(
        mg
    )

    coupling_strength.append(
        strength
    )


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    coupling_mg,
    coupling_strength
)

plt.xlabel(
    r"Ground-state $m_F$"
)

plt.ylabel(
    r"Relative coupling $|C_{m_F,q}|^2$"
)

plt.title(
    rf"$^{{87}}$Rb "
    rf"$F=2\rightarrow F'=3$: "
    rf"{POLARIZATION} polarization"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 2
# excited-state population spectra
# ============================================================

plt.figure(
    figsize=(8, 5)
)


for intensity in INTENSITIES:

    plt.plot(

        DETUNING,

        spectra[
            intensity
        ]["excited"],

        label=(
            f"{intensity:g} "
            r"mW/cm$^2$"
        )
    )


plt.xlabel(
    "Laser detuning (MHz)"
)

plt.ylabel(
    "Total excited-state population"
)

plt.title(
    r"$^{87}$Rb Zeeman-resolved OBE: "
    "excited population"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 3
# absorption proxy
#
# Common normalization across ALL powers.
# ============================================================

ABS_SCALE = max(

    np.max(
        np.abs(
            spectra[I]["absorption"]
        )
    )

    for I in INTENSITIES
)


plt.figure(
    figsize=(8, 5)
)


for intensity in INTENSITIES:

    plt.plot(

        DETUNING,

        spectra[
            intensity
        ]["absorption"]
        / ABS_SCALE,

        label=(
            f"{intensity:g} "
            r"mW/cm$^2$"
        )
    )


plt.xlabel(
    "Laser detuning (MHz)"
)

plt.ylabel(
    "Absorption proxy "
    "(common normalization)"
)

plt.title(
    "OBE absorption versus intensity"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 4
# dispersion proxy
#
# Again common normalization.
# ============================================================

DISP_SCALE = max(

    np.max(
        np.abs(
            spectra[I]["dispersion"]
        )
    )

    for I in INTENSITIES
)


plt.figure(
    figsize=(8, 5)
)


for intensity in INTENSITIES:

    plt.plot(

        DETUNING,

        spectra[
            intensity
        ]["dispersion"]
        / DISP_SCALE,

        label=(
            f"{intensity:g} "
            r"mW/cm$^2$"
        )
    )


plt.axhline(
    0,
    linewidth=0.8
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=0.8
)

plt.xlabel(
    "Laser detuning (MHz)"
)

plt.ylabel(
    "Dispersion proxy "
    "(common normalization)"
)

plt.title(
    "OBE dispersion versus intensity"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 5
# power broadening
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    INTENSITIES,

    FWHM_VALUES,

    marker="o",

    label="Zeeman-resolved OBE"
)


plt.plot(

    I_REFERENCE_LINE,

    FWHM_TWO_LEVEL,

    linestyle="--",

    label=(
        r"$\Gamma\sqrt{1+I/I_{\rm sat}}$"
        " reference"
    )
)


plt.xscale(
    "log"
)

plt.xlabel(
    r"Intensity (mW/cm$^2$)"
)

plt.ylabel(
    "FWHM (MHz)"
)

plt.title(
    "Power broadening"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 6
# resonant excitation + dispersive slope
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(8, 5)
)


line1 = ax1.plot(

    POWER_SCAN,

    excited_resonance,

    marker="o",

    label="Resonant excited population"
)


ax1.set_xscale(
    "log"
)

ax1.set_xlabel(
    r"Intensity (mW/cm$^2$)"
)

ax1.set_ylabel(
    "Excited-state population"
)


ax2 = ax1.twinx()


line2 = ax2.plot(

    POWER_SCAN,

    dispersion_slope
    / np.max(
        dispersion_slope
    ),

    marker="s",

    label="Dispersion slope"
)


ax2.set_ylabel(
    "Normalized dispersion slope"
)


lines = (
    line1
    + line2
)

labels = [
    x.get_label()
    for x in lines
]


ax1.legend(
    lines,
    labels,
    loc="best"
)

plt.title(
    "Power dependence of atomic response"
)

ax1.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 7
# optical pumping among ground mF states
# ============================================================

plt.figure(
    figsize=(8, 5)
)


for mg in mg_values:

    plt.plot(

        POWER_SCAN,

        ground_population_scan[
            int(mg)
        ],

        marker="o",

        label=(
            rf"$m_F={mg}$"
        )
    )


plt.xscale(
    "log"
)

plt.xlabel(
    r"Intensity (mW/cm$^2$)"
)

plt.ylabel(
    "Ground-state population"
)

plt.title(
    rf"Optical pumping: "
    rf"{POLARIZATION} polarization"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 21. Console diagnostics
# ============================================================

print()
print(
    "===================================================="
)

print(
    "Stage D-2A: Zeeman-resolved OBE + power"
)

print(
    "===================================================="
)


print()
print(
    "Atomic manifold:"
)

print(
    "Fg =",
    Fg,
    "   Fe =",
    Fe
)

print(
    "Ground states =",
    NG
)

print(
    "Excited states =",
    NE
)

print(
    "Total Hilbert dimension =",
    N
)


print()
print(
    "Natural linewidth:"
)

print(
    "Gamma/(2pi) =",
    GAMMA_MHZ,
    "MHz"
)


print()
print(
    "Polarization =",
    POLARIZATION
)


print()
print(
    "Clebsch-Gordan transitions:"
)

print(
    " mg    q    me       CG        CG^2"
)


for (
    mg,
    q,
    me
), cg in CG_TABLE.items():

    print(
        f"{mg:3d} "
        f"{q:4d} "
        f"{me:5d} "
        f"{cg:10.6f} "
        f"{cg**2:10.6f}"
    )


print()
print(
    "Spontaneous branching check:"
)


for me in me_values:

    print(
        f"me={me:+d}: "
        f"sum branching = "
        f"{BRANCHING_CHECK[int(me)]:.12f}"
    )


print()
print(
    "Intensity calibration:"
)

print(
    "Cycling-transition Isat =",
    ISAT_CYCLING,
    "mW/cm^2"
)

print(
    "Reference steady-state Isat "
    f"for {POLARIZATION} polarization =",
    ISAT_REFERENCE,
    "mW/cm^2"
)

print(
    "Numerical Isat from rho_ee=0.25 =",
    ISAT_NUMERICAL,
    "mW/cm^2"
)


print()
print(
    "Selected-power linewidths:"
)

print(
    f"{'I (mW/cm2)':>14} "
    f"{'FWHM (MHz)':>14}"
)


for I, width in zip(
    INTENSITIES,
    FWHM_VALUES
):

    print(
        f"{I:14.5f} "
        f"{width:14.6f}"
    )


print()
print(
    "High-power excited population =",
    excited_resonance[-1]
)

print(
    "Maximum normalized "
    "dispersion slope occurs at I =",
    POWER_SCAN[
        np.argmax(
            dispersion_slope
        )
    ],
    "mW/cm^2"
)


print()
print(
    "All finite =",
    (
        np.all(
            np.isfinite(
                excited_resonance
            )
        )
        and
        np.all(
            np.isfinite(
                dispersion_slope
            )
        )
        and
        np.all(
            np.isfinite(
                FWHM_VALUES
            )
        )
    )
)

print(
    "===================================================="
)


plt.show()