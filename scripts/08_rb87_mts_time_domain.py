import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from scipy.special import jv
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import find_peaks

from sympy import S
from sympy.physics.wigner import clebsch_gordan


# ============================================================
# Stage D-2B++ : quantitative low-power MTS validation
#
# 87Rb F=2 -> F'=3
#
# Zeeman-resolved OBE
# + phase-modulated pump
# + counter-propagating weak probe
# + spatial Fourier projection
# + temporal Fourier extraction
# + MTS heterodyne detection
#
# Main upgrades:
#
# 1. fine center-frequency grid
# 2. refined zero crossing
# 3. nearest-extrema Vpp / peak separation
# 4. phase unwrapping modulo 180 deg
# 5. slope / pump-power efficiency
# 6. Vpp / pump-power efficiency
# 7. optimized-phase and fixed-phase comparison
# ============================================================


# ============================================================
# 0. Calculation mode
# ============================================================

FAST_MODE = True

RUN_CONVERGENCE_CHECK = True


# ============================================================
# 1. Atomic parameters
# ============================================================

Fg = 2
Fe = 3

mg_values = np.arange(-Fg, Fg + 1)
me_values = np.arange(-Fe, Fe + 1)

NG = len(mg_values)
NE = len(me_values)

N = NG + NE

TWOPI = 2.0 * np.pi


# natural linewidth
# Gamma/(2pi) = 6.065 MHz

GAMMA_MHZ = 6.065
GAMMA = TWOPI * GAMMA_MHZ


# cycling-transition saturation intensity

ISAT_CYCLING = 1.66933    # mW/cm^2


# ============================================================
# 2. MTS parameters
# ============================================================

FM_MHZ = 12.5

OMEGA_M = (
    TWOPI * FM_MHZ
)

T_MOD = (
    1.0 / FM_MHZ
)   # microseconds


BETA = 0.28

J0 = jv(0, BETA)
J1 = jv(1, BETA)


# reference operating point

PUMP_INTENSITY = 3.0
PROBE_INTENSITY = 0.10


# ============================================================
# 3. Numerical parameters
# ============================================================

N_THETA = 8

N_SETTLE_PERIODS = 25
N_RECORD_PERIODS = 3
POINTS_PER_PERIOD = 64


# reference spectrum

DETUNING_SCAN = np.linspace(
    -20.0,
    +20.0,
    61
)


# center slope fitting region

CENTER_FIT_WINDOW_MHZ = 2.0


# true lock point is only searched here

ZERO_SEARCH_WINDOW_MHZ = 2.0


# local extrema belonging to central MTS feature
# sideband resonances near +-12.5 MHz are deliberately excluded

EXTREMA_SEARCH_WINDOW_MHZ = 10.0


# relative prominence threshold used when identifying
# true local extrema

EXTREMA_PROMINENCE_FRACTION = 0.01


# ============================================================
# 4. Pump-power scan
# ============================================================

PUMP_SCAN = np.array([
    0.10,
    0.30,
    1.00,
    3.00,
    10.0,
    30.0
])


if FAST_MODE:

    # The previous 1 MHz spacing was too coarse for zero finding.
    # 0.5 MHz is used over the broad region, with additional
    # 0.25 MHz sampling near resonance.

    coarse_grid = np.arange(
        -10.0,
        +10.0001,
        0.5
    )

    fine_center_grid = np.arange(
        -2.0,
        +2.0001,
        0.25
    )

else:

    coarse_grid = np.arange(
        -10.0,
        +10.0001,
        0.25
    )

    fine_center_grid = np.arange(
        -2.0,
        +2.0001,
        0.125
    )


PUMP_DETUNING_SCAN = np.unique(
    np.concatenate([
        coarse_grid,
        fine_center_grid
    ])
)


# ============================================================
# 5. Polarization
# ============================================================

POLARIZATION = "pi"


def polarization_components(name):

    if name == "pi":

        return {
            -1: 0.0,
             0: 1.0,
            +1: 0.0
        }

    if name == "sigma+":

        return {
            -1: 0.0,
             0: 0.0,
            +1: 1.0
        }

    if name == "sigma-":

        return {
            -1: 1.0,
             0: 0.0,
            +1: 0.0
        }

    raise ValueError("Unknown polarization.")


EPS = polarization_components(
    POLARIZATION
)


# ============================================================
# 6. Hilbert-space indexing
# ============================================================

def g_index(mg):

    return int(
        mg + Fg
    )


def e_index(me):

    return (
        NG
        + int(me + Fe)
    )


basis = [
    qt.basis(N, i)
    for i in range(N)
]


# ============================================================
# 7. Clebsch-Gordan coefficient
# ============================================================

def cg_coefficient(mg, q):

    me = mg + q

    if me < -Fe or me > Fe:
        return 0.0

    value = clebsch_gordan(
        S(Fg),
        S(1),
        S(Fe),

        S(int(mg)),
        S(int(q)),
        S(int(me))
    )

    return float(value)


# ============================================================
# 8. Dipole operators
# ============================================================

D_PLUS = 0 * qt.qeye(N)


for mg in mg_values:

    for q in [-1, 0, +1]:

        me = mg + q

        if me < -Fe or me > Fe:
            continue

        eps = EPS[q]

        if abs(eps) < 1e-15:
            continue

        cg = cg_coefficient(
            mg,
            q
        )

        D_PLUS += (
            eps
            * cg
            * basis[e_index(me)]
            * basis[g_index(mg)].dag()
        )


D_MINUS = D_PLUS.dag()


X_DIPOLE = (
    D_PLUS + D_MINUS
)


Y_DIPOLE = (
    1j
    * (
        D_PLUS - D_MINUS
    )
)


# ============================================================
# 9. Projectors
# ============================================================

GROUND_PROJECTORS = {}
EXCITED_PROJECTORS = {}


for mg in mg_values:

    ket = basis[
        g_index(mg)
    ]

    GROUND_PROJECTORS[int(mg)] = (
        ket * ket.dag()
    )


for me in me_values:

    ket = basis[
        e_index(me)
    ]

    EXCITED_PROJECTORS[int(me)] = (
        ket * ket.dag()
    )


P_GROUND = sum(
    GROUND_PROJECTORS.values()
)

P_EXCITED = sum(
    EXCITED_PROJECTORS.values()
)


# ============================================================
# 10. Coherence-preserving spontaneous emission
# ============================================================

C_OPS = []


for q in [-1, 0, +1]:

    Jq = 0 * qt.qeye(N)

    for mg in mg_values:

        me = mg + q

        if me < -Fe or me > Fe:
            continue

        cg = cg_coefficient(
            mg,
            q
        )

        if abs(cg) < 1e-15:
            continue

        Jq += (
            cg
            * basis[g_index(mg)]
            * basis[e_index(me)].dag()
        )

    C_OPS.append(
        np.sqrt(GAMMA) * Jq
    )


DECAY_CHECK = (
    sum(
        c.dag() * c
        for c in C_OPS
    )

    - GAMMA * P_EXCITED
)


DECAY_CHECK_NORM = (
    DECAY_CHECK.norm()
)


# ============================================================
# 11. Intensity -> Rabi frequency
# ============================================================

def cycling_rabi_from_intensity(intensity):

    if intensity <= 0:
        return 0.0

    return (
        GAMMA
        * np.sqrt(
            intensity
            / (
                2.0 * ISAT_CYCLING
            )
        )
    )


OMEGA_PUMP = cycling_rabi_from_intensity(
    PUMP_INTENSITY
)

OMEGA_PROBE = cycling_rabi_from_intensity(
    PROBE_INTENSITY
)


# ============================================================
# 12. Initial state
# ============================================================

RHO_INITIAL = (
    P_GROUND / NG
)


# ============================================================
# 13. Hamiltonian
# ============================================================

def build_hamiltonian(
    detuning_MHz,
    theta,
    pump_intensity=PUMP_INTENSITY,
    probe_intensity=PROBE_INTENSITY,
    beta=BETA
):

    Delta = (
        TWOPI * detuning_MHz
    )


    Omega_p = cycling_rabi_from_intensity(
        pump_intensity
    )

    Omega_s = cycling_rabi_from_intensity(
        probe_intensity
    )


    j0 = jv(
        0,
        beta
    )

    j1 = jv(
        1,
        beta
    )


    H_detuning = (
        -Delta * P_EXCITED
    )


    H_pump_carrier = (
        0.5
        * Omega_p
        * j0
        * X_DIPOLE
    )


    H_probe = (
        0.5
        * Omega_s
        * (
            np.cos(theta)
            * X_DIPOLE

            -

            np.sin(theta)
            * Y_DIPOLE
        )
    )


    H_static = (
        H_detuning
        + H_pump_carrier
        + H_probe
    )


    H_pm_operator = (
        Omega_p
        * j1
        * Y_DIPOLE
    )


    def pm_coefficient(t, **kwargs):

        return np.sin(
            OMEGA_M * t
        )


    return [
        H_static,

        [
            H_pm_operator,
            pm_coefficient
        ]
    ]


# ============================================================
# 14. Periodic steady-state solver
# ============================================================

SETTLE_OPTIONS = {
    "atol": 1e-9,
    "rtol": 1e-8,
    "nsteps": 30000,
    "store_states": True
}


RECORD_OPTIONS = {
    "atol": 1e-9,
    "rtol": 1e-8,
    "nsteps": 30000
}


def solve_periodic_polarization(
    detuning_MHz,
    theta,
    pump_intensity=PUMP_INTENSITY,
    probe_intensity=PROBE_INTENSITY,
    beta=BETA,
    n_settle=N_SETTLE_PERIODS,
    n_record=N_RECORD_PERIODS,
    points_per_period=POINTS_PER_PERIOD
):

    H = build_hamiltonian(
        detuning_MHz,
        theta,
        pump_intensity,
        probe_intensity,
        beta
    )


    settle_end = (
        n_settle * T_MOD
    )


    settle_result = qt.mesolve(
        H,
        RHO_INITIAL,
        [
            0.0,
            settle_end
        ],
        C_OPS,
        e_ops=[],
        options=SETTLE_OPTIONS
    )


    rho_start = (
        settle_result.states[-1]
    )


    record_end = (
        settle_end
        + n_record * T_MOD
    )


    t_record = np.linspace(
        settle_end,
        record_end,

        n_record
        * points_per_period
        + 1
    )


    result = qt.mesolve(
        H,
        rho_start,
        t_record,
        C_OPS,

        e_ops=[
            D_MINUS,
            P_EXCITED
        ],

        options=RECORD_OPTIONS
    )


    polarization = np.asarray(
        result.expect[0],
        dtype=np.complex128
    )


    excited_population = np.asarray(
        result.expect[1],
        dtype=float
    )


    return (
        t_record,
        polarization,
        excited_population
    )


# ============================================================
# 15. Probe-direction spatial Fourier projection
# ============================================================

def probe_direction_polarization(
    detuning_MHz,
    pump_intensity=PUMP_INTENSITY,
    probe_intensity=PROBE_INTENSITY,
    beta=BETA,
    n_theta=N_THETA,
    n_settle=N_SETTLE_PERIODS
):

    theta_values = (
        2.0
        * np.pi
        * np.arange(n_theta)
        / n_theta
    )


    all_polarization = []

    t_reference = None


    for theta in theta_values:

        t, P_t, _ = (
            solve_periodic_polarization(
                detuning_MHz,
                theta,
                pump_intensity,
                probe_intensity,
                beta,
                n_settle=n_settle
            )
        )


        if t_reference is None:
            t_reference = t


        all_polarization.append(
            P_t
        )


    all_polarization = np.asarray(
        all_polarization
    )


    phase_factor = np.exp(
        1j * theta_values
    )


    P_probe_t = np.mean(
        all_polarization
        * phase_factor[:, None],
        axis=0
    )


    return (
        t_reference,
        P_probe_t
    )


# ============================================================
# 16. Temporal Fourier components
# ============================================================

def time_fourier_components(t, P_t):

    # Remove the duplicated endpoint.
    tt = t[:-1]
    pp = P_t[:-1]


    P_carrier = np.mean(
        pp
    )


    P_upper = np.mean(
        pp
        * np.exp(
            +1j
            * OMEGA_M
            * tt
        )
    )


    P_lower = np.mean(
        pp
        * np.exp(
            -1j
            * OMEGA_M
            * tt
        )
    )


    return (
        P_carrier,
        P_upper,
        P_lower
    )


# ============================================================
# 17. Complex MTS heterodyne response
# ============================================================

def mts_complex_signal(
    detuning_MHz,
    pump_intensity=PUMP_INTENSITY,
    probe_intensity=PROBE_INTENSITY,
    beta=BETA,
    n_theta=N_THETA,
    n_settle=N_SETTLE_PERIODS
):

    t, P_probe_t = (
        probe_direction_polarization(
            detuning_MHz,
            pump_intensity,
            probe_intensity,
            beta,
            n_theta,
            n_settle
        )
    )


    (
        P0,
        P_upper,
        P_lower
    ) = time_fourier_components(
        t,
        P_probe_t
    )


    Omega_probe = (
        cycling_rabi_from_intensity(
            probe_intensity
        )
    )


    # Maxwell propagation:
    #
    # E_generated ~ i P
    #
    # Heterodyne RF response:
    #
    # Z ~ i (P_upper - P_lower*)

    Z = (
        1j
        * Omega_probe
        * (
            P_upper
            - np.conj(P_lower)
        )
    )


    return (
        Z,
        P0,
        P_upper,
        P_lower
    )


# ============================================================
# 18. Analysis utilities
# ============================================================

def demodulate(Z, phi):

    return np.real(
        Z
        * np.exp(
            1j * phi
        )
    )


def complex_center_slope(
    x,
    Z,
    fit_window=CENTER_FIT_WINDOW_MHZ
):

    mask = (
        np.abs(x)
        <= fit_window
    )


    xx = x[mask]
    ZZ = Z[mask]


    if len(xx) < 5:

        raise ValueError(
            "Not enough center points "
            "for center-slope fitting."
        )


    cr = np.polyfit(
        xx,
        np.real(ZZ),
        3
    )


    ci = np.polyfit(
        xx,
        np.imag(ZZ),
        3
    )


    return (
        cr[-2]
        + 1j * ci[-2]
    )


# ============================================================
# 19. Refined zero crossing
# ============================================================

def refined_zero_crossing(
    x,
    y,
    search_window=ZERO_SEARCH_WINDOW_MHZ
):

    mask = (
        np.abs(x)
        <= search_window
    )


    xx = np.asarray(
        x[mask]
    )

    yy = np.asarray(
        y[mask]
    )


    if len(xx) < 3:
        return np.nan


    interpolation = PchipInterpolator(
        xx,
        yy
    )


    xd = np.linspace(
        xx.min(),
        xx.max(),
        4001
    )


    yd = interpolation(
        xd
    )


    # Exact numerical zero if present

    i0 = np.argmin(
        np.abs(yd)
    )


    if abs(yd[i0]) < 1e-14:

        return float(
            xd[i0]
        )


    crossing_indices = np.where(
        yd[:-1]
        * yd[1:]
        < 0
    )[0]


    if len(crossing_indices) == 0:

        return np.nan


    centers = (
        xd[crossing_indices]
        + xd[crossing_indices + 1]
    ) / 2.0


    k = crossing_indices[
        np.argmin(
            np.abs(centers)
        )
    ]


    a = xd[k]
    b = xd[k + 1]


    try:

        root = brentq(
            interpolation,
            a,
            b
        )

    except ValueError:

        return np.nan


    return float(root)


# ============================================================
# 20. Nearest local extrema around lock point
# ============================================================

def nearest_extrema_metrics(
    x,
    y,
    zero,
    search_window=EXTREMA_SEARCH_WINDOW_MHZ,
    prominence_fraction=EXTREMA_PROMINENCE_FRACTION
):

    if not np.isfinite(zero):

        return {
            "Vpp": np.nan,
            "peak_separation": np.nan,
            "x_left": np.nan,
            "x_right": np.nan,
            "y_left": np.nan,
            "y_right": np.nan
        }


    mask = (
        (x >= zero - search_window)
        &
        (x <= zero + search_window)
    )


    xx = np.asarray(
        x[mask]
    )

    yy = np.asarray(
        y[mask]
    )


    if len(xx) < 5:

        return {
            "Vpp": np.nan,
            "peak_separation": np.nan,
            "x_left": np.nan,
            "x_right": np.nan,
            "y_left": np.nan,
            "y_right": np.nan
        }


    interpolation = PchipInterpolator(
        xx,
        yy
    )


    xd = np.linspace(
        xx.min(),
        xx.max(),
        8001
    )


    yd = interpolation(
        xd
    )


    dynamic_range = (
        np.max(yd)
        - np.min(yd)
    )


    prominence = max(
        prominence_fraction
        * dynamic_range,
        1e-15
    )


    maxima, _ = find_peaks(
        yd,
        prominence=prominence
    )


    minima, _ = find_peaks(
        -yd,
        prominence=prominence
    )


    extrema_indices = np.sort(
        np.concatenate([
            maxima,
            minima
        ])
    )


    x_ext = xd[
        extrema_indices
    ]

    y_ext = yd[
        extrema_indices
    ]


    left_mask = (
        x_ext < zero
    )

    right_mask = (
        x_ext > zero
    )


    if (
        not np.any(left_mask)
        or
        not np.any(right_mask)
    ):

        return {
            "Vpp": np.nan,
            "peak_separation": np.nan,
            "x_left": np.nan,
            "x_right": np.nan,
            "y_left": np.nan,
            "y_right": np.nan
        }


    # nearest local extremum to the left

    left_indices = np.where(
        left_mask
    )[0]

    il = left_indices[
        np.argmax(
            x_ext[left_indices]
        )
    ]


    # nearest local extremum to the right

    right_indices = np.where(
        right_mask
    )[0]

    ir = right_indices[
        np.argmin(
            x_ext[right_indices]
        )
    ]


    x_left = x_ext[il]
    x_right = x_ext[ir]

    y_left = y_ext[il]
    y_right = y_ext[ir]


    return {
        "Vpp":
            abs(
                y_right - y_left
            ),

        "peak_separation":
            x_right - x_left,

        "x_left":
            x_left,

        "x_right":
            x_right,

        "y_left":
            y_left,

        "y_right":
            y_right
    }


# ============================================================
# 21. Phase handling
# ============================================================

def unwrap_phase_mod_pi(
    phase_array
):

    phase_array = np.asarray(
        phase_array
    )


    # Physical mixer phase is equivalent under phi -> phi + pi.
    #
    # Therefore unwrap 2*phi and divide by two.

    return (
        0.5
        * np.unwrap(
            2.0 * phase_array
        )
    )


# ============================================================
# 22. Reference full spectrum
# ============================================================

Z_SCAN = np.zeros(
    len(DETUNING_SCAN),
    dtype=np.complex128
)


UPPER_SCAN = np.zeros_like(
    Z_SCAN
)

LOWER_SCAN = np.zeros_like(
    Z_SCAN
)


print()
print(
    "Calculating reference full MTS scan..."
)


for i, det in enumerate(
    DETUNING_SCAN
):

    print(
        f"{i+1:3d}/{len(DETUNING_SCAN)}  "
        f"detuning = {det:+7.3f} MHz"
    )


    (
        Z_SCAN[i],
        _,
        UPPER_SCAN[i],
        LOWER_SCAN[i]

    ) = mts_complex_signal(
        det
    )


# ============================================================
# 23. Reference demodulation
# ============================================================

SLOPE_COMPLEX = complex_center_slope(
    DETUNING_SCAN,
    Z_SCAN
)


PHI_OPT = (
    -np.angle(
        SLOPE_COMPLEX
    )
)


V_MTS = demodulate(
    Z_SCAN,
    PHI_OPT
)


ZERO_MTS = refined_zero_crossing(
    DETUNING_SCAN,
    V_MTS
)


REF_FEATURE = nearest_extrema_metrics(
    DETUNING_SCAN,
    V_MTS,
    ZERO_MTS
)


# ============================================================
# Figure 1: complex RF quadratures
# ============================================================

quad_scale = max(
    np.max(
        np.abs(
            np.real(Z_SCAN)
        )
    ),
    np.max(
        np.abs(
            np.imag(Z_SCAN)
        )
    )
)


plt.figure(figsize=(8, 5))


plt.plot(
    DETUNING_SCAN,
    np.real(Z_SCAN) / quad_scale,
    label="Re(Z)"
)


plt.plot(
    DETUNING_SCAN,
    np.imag(Z_SCAN) / quad_scale,
    label="Im(Z)"
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
    r"Detuning from $F=2\rightarrow F'=3$ (MHz)"
)

plt.ylabel(
    "Normalized RF quadrature"
)

plt.title(
    "Zeeman-resolved complex MTS response"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# Figure 2: reference MTS error signal
# ============================================================

V_SCALE = np.max(
    np.abs(
        V_MTS
    )
)


plt.figure(figsize=(8, 5))


plt.plot(
    DETUNING_SCAN,
    V_MTS / V_SCALE
)


plt.axhline(
    0,
    linewidth=0.8
)

plt.axvline(
    ZERO_MTS,
    linestyle="--",
    linewidth=0.8
)


plt.xlabel(
    "Laser detuning (MHz)"
)

plt.ylabel(
    "Normalized MTS signal"
)

plt.title(
    rf"MTS error signal: "
    rf"$f_m={FM_MHZ}$ MHz, "
    rf"$\beta={BETA}$"
)

plt.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# Figure 3: demodulation-phase dependence
# ============================================================

plt.figure(figsize=(8, 5))


for phase_deg in [
    0,
    30,
    60,
    90
]:

    V = demodulate(
        Z_SCAN,
        np.radians(
            phase_deg
        )
    )


    plt.plot(
        DETUNING_SCAN,
        V / V_SCALE,
        label=f"{phase_deg} deg"
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
    "MTS signal (common normalization)"
)

plt.title(
    "MTS demodulation-phase dependence"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# Figure 4: generated sidebands
# ============================================================

sideband_scale = max(
    np.max(
        np.abs(UPPER_SCAN)
    ),
    np.max(
        np.abs(LOWER_SCAN)
    )
)


plt.figure(figsize=(8, 5))


plt.plot(
    DETUNING_SCAN,
    np.abs(UPPER_SCAN) / sideband_scale,
    label=r"$\omega+\Omega_m$"
)


plt.plot(
    DETUNING_SCAN,
    np.abs(LOWER_SCAN) / sideband_scale,
    label=r"$\omega-\Omega_m$"
)


plt.xlabel(
    "Laser detuning (MHz)"
)

plt.ylabel(
    "Generated sideband magnitude"
)

plt.title(
    "Probe-direction generated sidebands"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# 24. Pump-power scan
# ============================================================

pump_results = []


print()
print(
    "Calculating pump-power scans..."
)


for Ip in PUMP_SCAN:

    print()
    print(
        f"Pump intensity = {Ip:.3f} mW/cm^2"
    )


    Zp = np.zeros(
        len(PUMP_DETUNING_SCAN),
        dtype=np.complex128
    )


    for i, det in enumerate(
        PUMP_DETUNING_SCAN
    ):

        print(
            f"  {i+1:3d}/"
            f"{len(PUMP_DETUNING_SCAN)}  "
            f"Delta = {det:+6.2f} MHz"
        )


        Zp[i], _, _, _ = mts_complex_signal(
            det,
            pump_intensity=Ip
        )


    # --------------------------------------------------------
    # Complex center slope
    # --------------------------------------------------------

    slope_complex = complex_center_slope(
        PUMP_DETUNING_SCAN,
        Zp
    )


    slope_opt = abs(
        slope_complex
    )


    # optimum phase for this pump power

    phi_opt = (
        -np.angle(
            slope_complex
        )
    )


    # optimized-phase signal

    V_opt = demodulate(
        Zp,
        phi_opt
    )


    # experimentally relevant:
    # keep the phase fixed at the 3 mW/cm^2 operating point

    V_fixed = demodulate(
        Zp,
        PHI_OPT
    )


    slope_fixed = abs(
        np.real(
            slope_complex
            * np.exp(
                1j * PHI_OPT
            )
        )
    )


    # --------------------------------------------------------
    # Refined zero crossings
    # --------------------------------------------------------

    zero_opt = refined_zero_crossing(
        PUMP_DETUNING_SCAN,
        V_opt
    )


    zero_fixed = refined_zero_crossing(
        PUMP_DETUNING_SCAN,
        V_fixed
    )


    # --------------------------------------------------------
    # Real nearest-extrema metrics
    # --------------------------------------------------------

    feature_opt = nearest_extrema_metrics(
        PUMP_DETUNING_SCAN,
        V_opt,
        zero_opt
    )


    feature_fixed = nearest_extrema_metrics(
        PUMP_DETUNING_SCAN,
        V_fixed,
        zero_fixed
    )


    pump_results.append({

        "I":
            Ip,

        "Z":
            Zp,

        "V_opt":
            V_opt,

        "V_fixed":
            V_fixed,

        "slope_complex":
            slope_complex,

        "slope_opt":
            slope_opt,

        "slope_fixed":
            slope_fixed,

        "phi_raw":
            phi_opt,

        "zero_opt":
            zero_opt,

        "zero_fixed":
            zero_fixed,

        "Vpp_opt":
            feature_opt["Vpp"],

        "Vpp_fixed":
            feature_fixed["Vpp"],

        "peak_sep_opt":
            feature_opt[
                "peak_separation"
            ],

        "peak_sep_fixed":
            feature_fixed[
                "peak_separation"
            ],

        "x_left_opt":
            feature_opt["x_left"],

        "x_right_opt":
            feature_opt["x_right"]
    })


# ============================================================
# 25. Convert results to arrays
# ============================================================

SLOPE_OPT_SCAN = np.array([
    r["slope_opt"]
    for r in pump_results
])


SLOPE_FIXED_SCAN = np.array([
    r["slope_fixed"]
    for r in pump_results
])


VPP_OPT_SCAN = np.array([
    r["Vpp_opt"]
    for r in pump_results
])


VPP_FIXED_SCAN = np.array([
    r["Vpp_fixed"]
    for r in pump_results
])


ZERO_OPT_SCAN = np.array([
    r["zero_opt"]
    for r in pump_results
])


ZERO_FIXED_SCAN = np.array([
    r["zero_fixed"]
    for r in pump_results
])


PEAK_SEP_OPT_SCAN = np.array([
    r["peak_sep_opt"]
    for r in pump_results
])


PHI_RAW_SCAN = np.array([
    r["phi_raw"]
    for r in pump_results
])


# ============================================================
# 26. Continuous phase unwrap
# ============================================================

PHI_UNWRAPPED = unwrap_phase_mod_pi(
    PHI_RAW_SCAN
)


# Shift the whole curve by n*pi so that the 3 mW point
# lies closest to the reference PHI_OPT.

reference_index = np.argmin(
    np.abs(
        PUMP_SCAN
        - PUMP_INTENSITY
    )
)


n_shift = np.round(
    (
        PHI_OPT
        - PHI_UNWRAPPED[
            reference_index
        ]
    )
    / np.pi
)


PHI_UNWRAPPED += (
    n_shift * np.pi
)


PHI_UNWRAPPED_DEG = np.degrees(
    PHI_UNWRAPPED
)


# ============================================================
# 27. Low-power efficiency metrics
# ============================================================

SLOPE_EFF_OPT = (
    SLOPE_OPT_SCAN
    / PUMP_SCAN
)


SLOPE_EFF_FIXED = (
    SLOPE_FIXED_SCAN
    / PUMP_SCAN
)


VPP_EFF_OPT = (
    VPP_OPT_SCAN
    / PUMP_SCAN
)


VPP_EFF_FIXED = (
    VPP_FIXED_SCAN
    / PUMP_SCAN
)


# ============================================================
# Figure 5:
# pump-dependent central MTS lineshapes
# ============================================================

power_line_scale = max(
    np.max(
        np.abs(
            r["V_opt"]
        )
    )
    for r in pump_results
)


plt.figure(figsize=(8, 5))


for r in pump_results:

    plt.plot(
        PUMP_DETUNING_SCAN,
        r["V_opt"] / power_line_scale,
        label=(
            f"{r['I']:g} "
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
    "MTS signal (common normalization)"
)

plt.title(
    "Pump-dependent central MTS lineshapes"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# Figure 6:
# absolute MTS performance
# ============================================================

plt.figure(figsize=(8, 5))


plt.plot(
    PUMP_SCAN,
    SLOPE_OPT_SCAN
    / np.nanmax(SLOPE_OPT_SCAN),
    marker="o",
    label="Optimized-phase center slope"
)


plt.plot(
    PUMP_SCAN,
    SLOPE_FIXED_SCAN
    / np.nanmax(SLOPE_FIXED_SCAN),
    marker="s",
    label="Fixed-phase center slope"
)


plt.plot(
    PUMP_SCAN,
    VPP_OPT_SCAN
    / np.nanmax(VPP_OPT_SCAN),
    marker="^",
    label="Nearest-extrema Vpp"
)


plt.xscale(
    "log"
)


plt.xlabel(
    r"Pump intensity (mW/cm$^2$)"
)

plt.ylabel(
    "Normalized metric"
)

plt.title(
    "Absolute MTS performance versus pump power"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# Figure 7:
# true nearest-extrema separation + refined zero
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(8, 5)
)


line1 = ax1.plot(
    PUMP_SCAN,
    PEAK_SEP_OPT_SCAN,
    marker="o",
    label="Nearest-extrema separation"
)


ax1.set_xscale(
    "log"
)

ax1.set_xlabel(
    r"Pump intensity (mW/cm$^2$)"
)

ax1.set_ylabel(
    "Nearest-extrema separation (MHz)"
)


ax2 = ax1.twinx()


line2 = ax2.plot(
    PUMP_SCAN,
    ZERO_FIXED_SCAN,
    marker="s",
    label="Fixed-phase lock zero"
)


ax2.axhline(
    0,
    linestyle="--",
    linewidth=0.8
)


ax2.set_ylabel(
    "Zero crossing shift (MHz)"
)


lines = line1 + line2

labels = [
    line.get_label()
    for line in lines
]


ax1.legend(
    lines,
    labels,
    loc="best"
)


plt.title(
    "Central MTS feature width and lock point"
)

ax1.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# Figure 8:
# continuous optimum demodulation phase
# ============================================================

plt.figure(figsize=(8, 5))


plt.plot(
    PUMP_SCAN,
    PHI_UNWRAPPED_DEG,
    marker="o"
)


plt.axvline(
    PUMP_INTENSITY,
    linestyle="--",
    linewidth=0.8,
    label="Reference pump"
)


plt.xscale(
    "log"
)


plt.xlabel(
    r"Pump intensity (mW/cm$^2$)"
)

plt.ylabel(
    "Continuous optimum phase (deg)"
)

plt.title(
    "Pump dependence of optimum demodulation phase"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# Figure 9:
# low-power efficiency
# ============================================================

plt.figure(figsize=(8, 5))


plt.plot(
    PUMP_SCAN,
    SLOPE_EFF_FIXED
    / np.nanmax(SLOPE_EFF_FIXED),
    marker="o",
    label=r"$K_\nu/I_p$"
)


plt.plot(
    PUMP_SCAN,
    VPP_EFF_FIXED
    / np.nanmax(VPP_EFF_FIXED),
    marker="s",
    label=r"$V_{\rm pp}/I_p$"
)


plt.xscale(
    "log"
)

plt.yscale(
    "log"
)


plt.xlabel(
    r"Pump intensity (mW/cm$^2$)"
)

plt.ylabel(
    "Normalized power efficiency"
)

plt.title(
    "Low-power MTS efficiency"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# 28. Weak-probe scaling
# ============================================================

PROBE_SCAN = np.array([
    0.01,
    0.03,
    0.10,
    0.30
])


TEST_DETUNING = 3.0


PROBE_SIGNAL = []


print()
print(
    "Checking weak-probe scaling..."
)


for Is in PROBE_SCAN:

    print(
        f"probe I = {Is:.3f} mW/cm^2"
    )


    Z, _, _, _ = mts_complex_signal(
        TEST_DETUNING,
        probe_intensity=Is
    )


    PROBE_SIGNAL.append(
        abs(Z)
    )


PROBE_SIGNAL = np.asarray(
    PROBE_SIGNAL
)


# ============================================================
# Figure 10:
# weak probe linearity
# ============================================================

plt.figure(figsize=(8, 5))


plt.plot(
    PROBE_SCAN,
    PROBE_SIGNAL,
    marker="o",
    label="Calculated"
)


reference_line = (
    PROBE_SIGNAL[0]
    * PROBE_SCAN
    / PROBE_SCAN[0]
)


plt.plot(
    PROBE_SCAN,
    reference_line,
    linestyle="--",
    label="Linear in probe intensity"
)


plt.xlabel(
    r"Probe intensity (mW/cm$^2$)"
)

plt.ylabel(
    "MTS heterodyne magnitude (arb. units)"
)

plt.title(
    "Weak-probe linearity check"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ============================================================
# 29. beta = 0 null test
# ============================================================

Z_BETA_ZERO, _, _, _ = mts_complex_signal(
    TEST_DETUNING,
    beta=0.0
)


Z_BETA_NORMAL, _, _, _ = mts_complex_signal(
    TEST_DETUNING,
    beta=BETA
)


NULL_RATIO = (
    abs(Z_BETA_ZERO)
    /
    max(
        abs(Z_BETA_NORMAL),
        1e-30
    )
)


# ============================================================
# 30. Numerical convergence checks
# ============================================================

def quick_complex_slope(
    n_theta,
    n_settle,
    step=0.5
):

    Zm, _, _, _ = mts_complex_signal(
        -step,
        n_theta=n_theta,
        n_settle=n_settle
    )


    Zp, _, _, _ = mts_complex_signal(
        +step,
        n_theta=n_theta,
        n_settle=n_settle
    )


    return (
        Zp - Zm
    ) / (
        2.0 * step
    )


THETA_ERROR = np.nan
SETTLE_ERROR = np.nan


if RUN_CONVERGENCE_CHECK:

    print()
    print(
        "Running convergence checks..."
    )


    slope_base = quick_complex_slope(
        N_THETA,
        N_SETTLE_PERIODS
    )


    slope_theta = quick_complex_slope(
        2 * N_THETA,
        N_SETTLE_PERIODS
    )


    slope_settle = quick_complex_slope(
        N_THETA,
        40
    )


    denominator = max(
        abs(slope_base),
        1e-30
    )


    THETA_ERROR = (
        abs(
            slope_theta
            - slope_base
        )
        / denominator
    )


    SETTLE_ERROR = (
        abs(
            slope_settle
            - slope_base
        )
        / denominator
    )


# ============================================================
# 31. Print diagnostics
# ============================================================

print()
print(
    "======================================================"
)

print(
    "Stage D-2B++ low-power quantitative MTS validation"
)

print(
    "======================================================"
)


print()
print(
    "Atomic model:"
)

print(
    f"Fg = {Fg}, Fe = {Fe}"
)

print(
    f"Hilbert dimension = {N}"
)

print(
    "Decay consistency norm =",
    DECAY_CHECK_NORM
)


print()
print(
    "Reference MTS parameters:"
)

print(
    "fm =",
    FM_MHZ,
    "MHz"
)

print(
    "beta =",
    BETA
)

print(
    "J0 =",
    J0
)

print(
    "J1 =",
    J1
)

print(
    "pump =",
    PUMP_INTENSITY,
    "mW/cm^2"
)

print(
    "probe =",
    PROBE_INTENSITY,
    "mW/cm^2"
)


print()
print(
    "Reference MTS result:"
)

print(
    "complex center slope =",
    SLOPE_COMPLEX
)

print(
    "center slope magnitude =",
    abs(SLOPE_COMPLEX)
)

print(
    "optimum demod phase =",
    np.degrees(PHI_OPT),
    "deg (mod 180 deg)"
)

print(
    "refined zero crossing =",
    ZERO_MTS,
    "MHz"
)

print(
    "nearest-extrema Vpp =",
    REF_FEATURE["Vpp"]
)

print(
    "nearest-extrema separation =",
    REF_FEATURE["peak_separation"],
    "MHz"
)


# ============================================================
# Pump table
# ============================================================

print()
print(
    "Pump-power quantitative scan:"
)

print()

print(
    f"{'I_pump':>8} "
    f"{'SlopeOpt':>11} "
    f"{'SlopeFix':>11} "
    f"{'VppOpt':>11} "
    f"{'VppFix':>11} "
    f"{'Sep':>8} "
    f"{'ZeroFix':>10} "
    f"{'PhiUnwrap':>10}"
)


for i, r in enumerate(
    pump_results
):

    print(
        f"{r['I']:8.3f} "
        f"{r['slope_opt']:11.4e} "
        f"{r['slope_fixed']:11.4e} "
        f"{r['Vpp_opt']:11.4e} "
        f"{r['Vpp_fixed']:11.4e} "
        f"{r['peak_sep_opt']:8.3f} "
        f"{r['zero_fixed']:10.6f} "
        f"{PHI_UNWRAPPED_DEG[i]:10.3f}"
    )


# ============================================================
# Low-power efficiency table
# ============================================================

print()
print(
    "Low-power efficiency:"
)

print()

print(
    f"{'I_pump':>8} "
    f"{'SlopeFix/I':>14} "
    f"{'VppFix/I':>14}"
)


for i, Ip in enumerate(
    PUMP_SCAN
):

    print(
        f"{Ip:8.3f} "
        f"{SLOPE_EFF_FIXED[i]:14.6e} "
        f"{VPP_EFF_FIXED[i]:14.6e}"
    )


# ============================================================
# Weak probe
# ============================================================

print()
print(
    "Weak-probe scaling:"
)

print(
    f"{'I_probe':>10} "
    f"{'signal':>14} "
    f"{'signal/I':>14}"
)


for Is, sig in zip(
    PROBE_SCAN,
    PROBE_SIGNAL
):

    print(
        f"{Is:10.4f} "
        f"{sig:14.6e} "
        f"{sig/Is:14.6e}"
    )


print()
print(
    "beta=0 null-test ratio =",
    NULL_RATIO
)


# ============================================================
# Convergence
# ============================================================

if RUN_CONVERGENCE_CHECK:

    print()
    print(
        "Numerical convergence:"
    )

    print(
        f"N_theta "
        f"{N_THETA} -> {2*N_THETA} "
        f"relative complex-slope change = "
        f"{THETA_ERROR:.6e}"
    )

    print(
        f"N_settle "
        f"{N_SETTLE_PERIODS} -> 40 "
        f"relative complex-slope change = "
        f"{SETTLE_ERROR:.6e}"
    )


# ============================================================
# Simple physical interpretation
# ============================================================

idx_abs_slope = np.nanargmax(
    SLOPE_FIXED_SCAN
)


idx_slope_eff = np.nanargmax(
    SLOPE_EFF_FIXED
)


idx_vpp_eff = np.nanargmax(
    VPP_EFF_FIXED
)


print()
print(
    "Interpretation summary:"
)


print(
    "Largest absolute fixed-phase slope "
    "within scanned range at I =",
    PUMP_SCAN[idx_abs_slope],
    "mW/cm^2"
)


print(
    "Highest slope-per-power efficiency at I =",
    PUMP_SCAN[idx_slope_eff],
    "mW/cm^2"
)


print(
    "Highest Vpp-per-power efficiency at I =",
    PUMP_SCAN[idx_vpp_eff],
    "mW/cm^2"
)


if idx_abs_slope == len(PUMP_SCAN) - 1:

    print(
        "NOTE: absolute MTS slope is still increasing "
        "at the upper scan boundary."
    )

    print(
        "This does NOT imply that the upper-bound "
        "pump power is the optimum low-power operating point."
    )


print()
print(
    "All finite =",
    (
        np.all(
            np.isfinite(
                Z_SCAN
            )
        )

        and

        np.all(
            np.isfinite(
                SLOPE_OPT_SCAN
            )
        )
    )
)


print(
    "======================================================"
)


plt.show()