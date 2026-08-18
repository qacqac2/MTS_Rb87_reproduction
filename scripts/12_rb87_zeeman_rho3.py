"""
12_rb87_zeeman_rho3.py

Stage F1
Real 87Rb D2 closed transition:

    Fg = 2  ->  Fe = 3

Zeeman-resolved explicit third-order MTS solver.

Physics chain:

    real CG coefficients
        ->
    12-state Liouvillian
        ->
    rho^(0)
        ->
    rho^(1)
        ->
    rho^(2)
        ->
    rho^(3)
        ->
    probe-direction generated sidebands
        ->
    heterodyne Z
        ->
    demodulated MTS error signal

This is the real-Rb replacement for the generic
three-level Stage-09 engine.

Doppler averaging is deliberately NOT performed here.
The function response_at_detuning(..., kv=...)
already supports the propagation/Doppler label and
will be wrapped by Stage F2 after this core passes.
"""


import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass

from scipy.special import jv
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from sympy.physics.wigner import clebsch_gordan
from sympy import S


# ============================================================
# 0. Calculation mode
# ============================================================

FAST_MODE = False

RUN_SCALING_CHECK = True


# ============================================================
# 1. Real 87Rb D2 parameters
# ============================================================

# Natural linewidth:
#
# Gamma / (2 pi) = 6.065 MHz
#
# All internal frequencies below are normalized to Gamma.

GAMMA_MHZ = 6.065

FM_MHZ = 12.5

FM = (
    FM_MHZ
    /
    GAMMA_MHZ
)

BETA = 0.28


# ------------------------------------------------------------
# Weak-field Rabi amplitudes
#
# IMPORTANT:
# Keep the first explicit rho^(3) real-Rb validation
# in the perturbative regime.
#
# Strong experimental powers will be added only after
# rho^(3) vs full-master validation is repeated.
# ------------------------------------------------------------

OMEGA_PUMP = 0.15

OMEGA_PROBE = 0.03


# ============================================================
# 2. Hyperfine / Zeeman structure
# ============================================================

FG = 2

FE = 3


MG = np.arange(
    -FG,
    FG + 1,
    dtype=int
)

ME = np.arange(
    -FE,
    FE + 1,
    dtype=int
)


NG = len(MG)
NE = len(ME)

N = NG + NE

DIM = N * N


# ------------------------------------------------------------
# Polarization
#
# q = 0 : pi
# q = +1: sigma+
# q = -1: sigma-
#
# Keep q=0 first because this matches the previous
# Zeeman-resolved validation stage.
# ------------------------------------------------------------

POL_Q = 0


# ============================================================
# 3. Optional magnetic field
# ============================================================

B_GAUSS = 0.0


# approximate hyperfine Landé factors

G_F_GROUND = 0.5

G_F_EXCITED = 2.0 / 3.0


MU_B_OVER_H_MHZ_PER_G = 1.3996246


ZEEMAN_SCALE = (
    MU_B_OVER_H_MHZ_PER_G
    *
    B_GAUSS
    /
    GAMMA_MHZ
)


# ============================================================
# 4. Transit / ground-state relaxation
# ============================================================

# Without a weak ground-state relaxation mechanism,
# the field-free Liouvillian has several degenerate
# stationary ground-state distributions.
#
# A small transit/mixing rate makes rho^(0) unique:
#
#     rho_g = I_g / (2F+1)
#
# and mimics fresh thermal atoms entering the laser beam.

GAMMA_TRANSIT = 0.02


# ============================================================
# 5. Scan
# ============================================================

if FAST_MODE:

    DELTA_SCAN = np.linspace(
        -5.0,
        +5.0,
        61
    )

else:

    DELTA_SCAN = np.linspace(
        -6.0,
        +6.0,
        121
    )


# ============================================================
# 6. State indexing
# ============================================================

def g_index(m):

    return int(
        m + FG
    )


def e_index(m):

    return int(
        NG
        +
        m
        +
        FE
    )


def ket(i):

    x = np.zeros(
        N,
        dtype=complex
    )

    x[i] = 1.0

    return x


def projector(i, j):

    return np.outer(
        ket(i),
        ket(j).conj()
    )


I_N = np.eye(
    N,
    dtype=complex
)


# ============================================================
# 7. CG coefficients
# ============================================================

def cg(
    mg,
    q,
    me
):
    """
    <Fg mg; 1 q | Fe me>
    """

    if (
        me
        !=
        mg + q
    ):

        return 0.0


    if (
        mg < -FG
        or mg > FG
        or me < -FE
        or me > FE
    ):

        return 0.0


    return float(

        clebsch_gordan(

            S(FG),
            S(1),
            S(FE),

            S(int(mg)),
            S(int(q)),
            S(int(me))
        )
    )


# ============================================================
# 8. Dipole operators
# ============================================================

def dipole_raise(
    q
):
    """
    Dimensionless absorption operator

        D_q^(+)
        =
        sum C |e><g|
    """

    D = np.zeros(
        (N, N),
        dtype=complex
    )


    for mg in MG:

        me = (
            mg + q
        )


        if (
            me < -FE
            or me > FE
        ):

            continue


        c = cg(
            mg,
            q,
            me
        )


        if abs(c) < 1e-15:
            continue


        D[
            e_index(me),
            g_index(mg)
        ] = c


    return D


D_RAISE = {

    q:
    dipole_raise(q)

    for q in (
        -1,
        0,
        +1
    )
}


D_LOWER = {

    q:
    D_RAISE[q].conj().T

    for q in (
        -1,
        0,
        +1
    )
}


D_DRIVE = (
    D_RAISE[POL_Q]
)


D_DRIVE_DAG = (
    D_DRIVE.conj().T
)


# ============================================================
# 9. Excited-state projector
# ============================================================

P_E = np.zeros(
    (N, N),
    dtype=complex
)


for me in ME:

    i = e_index(me)

    P_E[i, i] = 1.0


# ============================================================
# 10. Collapse operators
# ============================================================

C_OPS_SPONT = []


# Angular-momentum spontaneous emission:
#
# one jump operator for each emitted spherical polarization q.

for q in (
    -1,
    0,
    +1
):

    C_OPS_SPONT.append(

        D_LOWER[q]
    )


# ------------------------------------------------------------
# Check:
#
# sum C_q^dag C_q = P_e
# ------------------------------------------------------------

DECAY_CHECK = np.zeros(
    (N, N),
    dtype=complex
)


for C in C_OPS_SPONT:

    DECAY_CHECK += (
        C.conj().T
        @
        C
    )


DECAY_ERROR = np.linalg.norm(
    DECAY_CHECK
    -
    P_E
)


# ------------------------------------------------------------
# Ground-state transit mixing
#
# For every ground state j:
#
# total rate out of j = GAMMA_TRANSIT.
# ------------------------------------------------------------

C_OPS_TRANSIT = []


if GAMMA_TRANSIT > 0:

    branch_rate = (
        GAMMA_TRANSIT
        /
        (NG - 1)
    )


    for mg_from in MG:

        for mg_to in MG:

            if (
                mg_to
                ==
                mg_from
            ):

                continue


            C = (

                np.sqrt(
                    branch_rate
                )

                *
                projector(
                    g_index(mg_to),
                    g_index(mg_from)
                )
            )


            C_OPS_TRANSIT.append(
                C
            )


C_OPS = (
    C_OPS_SPONT
    +
    C_OPS_TRANSIT
)


# ============================================================
# 11. Vectorization
# ============================================================

def vec(
    rho
):

    return np.asarray(
        rho,
        dtype=complex
    ).reshape(
        DIM,
        order="F"
    )


def mat(
    v
):

    return np.asarray(
        v,
        dtype=complex
    ).reshape(
        (N, N),
        order="F"
    )


TRACE_ROW = np.zeros(
    DIM,
    dtype=complex
)


for i in range(N):

    TRACE_ROW[
        i
        +
        i * N
    ] = 1.0


# ============================================================
# 12. Superoperators
# ============================================================

def commutator_superoperator(
    H
):
    """
        -i [H, rho]
    """

    return -1j * (

        np.kron(
            I_N,
            H
        )

        -

        np.kron(
            H.T,
            I_N
        )
    )


def lindblad_superoperator(
    C
):

    CdC = (
        C.conj().T
        @
        C
    )


    return (

        np.kron(
            C.conj(),
            C
        )

        -

        0.5
        *
        np.kron(
            I_N,
            CdC
        )

        -

        0.5
        *
        np.kron(
            CdC.T,
            I_N
        )
    )


# ============================================================
# 13. Field-free Liouvillian
# ============================================================

def build_L0(
    delta
):
    """
    delta:
        carrier detuning from the
        F=2 -> F'=3 hyperfine line,
        normalized by Gamma.

    Rotating frame:

        ground:
            Zeeman shift

        excited:
            -delta + Zeeman shift
    """

    H0 = np.zeros(
        (N, N),
        dtype=complex
    )


    # --------------------------------------------------------
    # ground Zeeman energies
    # --------------------------------------------------------

    for mg in MG:

        H0[
            g_index(mg),
            g_index(mg)
        ] = (

            G_F_GROUND
            *
            ZEEMAN_SCALE
            *
            mg
        )


    # --------------------------------------------------------
    # excited Zeeman energies
    # --------------------------------------------------------

    for me in ME:

        H0[
            e_index(me),
            e_index(me)
        ] = (

            -delta

            +

            G_F_EXCITED
            *
            ZEEMAN_SCALE
            *
            me
        )


    L = (
        commutator_superoperator(
            H0
        )
    )


    for C in C_OPS:

        L += (
            lindblad_superoperator(
                C
            )
        )


    return L


# ============================================================
# 14. Zeroth-order state
# ============================================================

RHO0 = np.zeros(
    (N, N),
    dtype=complex
)


for mg in MG:

    RHO0[
        g_index(mg),
        g_index(mg)
    ] = (
        1.0
        /
        NG
    )


# ============================================================
# 15. Optical modes
# ============================================================

@dataclass(frozen=True)
class FieldMode:

    name: str

    # modulation harmonic
    n: int

    # propagation sign:
    #
    # pump  +1
    # probe -1
    k: int

    amp: complex


def make_field_modes(
    beta=BETA,
    omega_pump=OMEGA_PUMP,
    omega_probe=OMEGA_PROBE,
    field_scale=1.0
):

    return [

        FieldMode(

            "probe",

            0,

            -1,

            field_scale
            *
            omega_probe
        ),


        FieldMode(

            "pump_-1",

            -1,

            +1,

            field_scale
            *
            omega_pump
            *
            jv(
                -1,
                beta
            )
        ),


        FieldMode(

            "pump_0",

            0,

            +1,

            field_scale
            *
            omega_pump
            *
            jv(
                0,
                beta
            )
        ),


        FieldMode(

            "pump_+1",

            +1,

            +1,

            field_scale
            *
            omega_pump
            *
            jv(
                +1,
                beta
            )
        )
    ]


# ============================================================
# 16. Interaction branches
# ============================================================

@dataclass(frozen=True)
class Branch:

    label: str

    dn: int

    dk: int

    M: np.ndarray


def make_branches(
    modes
):

    branches = []


    for mode in modes:


        # positive-frequency interaction

        H_plus = (

            -0.5

            *
            mode.amp

            *
            D_DRIVE
        )


        # Hermitian-conjugate interaction

        H_minus = (

            -0.5

            *
            np.conj(
                mode.amp
            )

            *
            D_DRIVE_DAG
        )


        branches.append(

            Branch(

                mode.name
                +
                "[+]",

                mode.n,

                mode.k,

                commutator_superoperator(
                    H_plus
                )
            )
        )


        branches.append(

            Branch(

                mode.name
                +
                "[-]",

                -mode.n,

                -mode.k,

                commutator_superoperator(
                    H_minus
                )
            )
        )


    return branches


# ============================================================
# 17. Solve one perturbative Fourier/spatial component
# ============================================================

def solve_correction(
    L0,
    slow_frequency,
    source
):

    A = (

        -1j
        *
        slow_frequency
        *
        np.eye(
            DIM,
            dtype=complex
        )

        -

        L0
    )


    b = vec(
        source
    )


    # q=k=0 block:
    # impose zero trace for every perturbative correction.

    if (
        abs(
            slow_frequency
        )
        <
        1e-13
    ):

        A = A.copy()

        b = b.copy()


        A[0, :] = (
            TRACE_ROW
        )

        b[0] = 0.0


    solution = np.linalg.solve(
        A,
        b
    )


    return mat(
        solution
    )


# ============================================================
# 18. Explicit rho^(1), rho^(2), rho^(3)
# ============================================================

def perturbative_orders(
    delta,
    kv=0.0,
    beta=BETA,
    omega_pump=OMEGA_PUMP,
    omega_probe=OMEGA_PROBE,
    field_scale=1.0
):
    """
    Dictionary key:

        (q, k)

    q:
        modulation harmonic

    k:
        accumulated propagation label.

    Moving atom frequency:

        Omega_slow
        =
        q * FM
        -
        k * kv
    """

    L0 = build_L0(
        delta
    )


    modes = make_field_modes(

        beta=beta,

        omega_pump=omega_pump,

        omega_probe=omega_probe,

        field_scale=field_scale
    )


    branches = make_branches(
        modes
    )


    orders = {

        0: {

            (0, 0):
            RHO0.copy()
        }
    }


    for order in (
        1,
        2,
        3
    ):


        source_dict = {}


        for (
            q0,
            k0

        ), rho_previous in orders[
            order - 1
        ].items():


            rho_vec = vec(
                rho_previous
            )


            for branch in branches:


                source = mat(

                    branch.M
                    @
                    rho_vec
                )


                if (
                    np.linalg.norm(
                        source
                    )
                    <
                    1e-18
                ):

                    continue


                key = (

                    q0
                    +
                    branch.dn,

                    k0
                    +
                    branch.dk
                )


                if key not in source_dict:

                    source_dict[key] = (
                        np.zeros(
                            (N, N),
                            dtype=complex
                        )
                    )


                source_dict[key] += (
                    source
                )


        current = {}


        for (
            q,
            k

        ), source in source_dict.items():


            slow_frequency = (

                q
                *
                FM

                -

                k
                *
                kv
            )


            current[(q, k)] = (
                solve_correction(

                    L0,

                    slow_frequency,

                    source
                )
            )


        orders[order] = current


    return orders


# ============================================================
# 19. Optical polarization
# ============================================================

def positive_polarization(
    rho
):
    """
    pi-polarized positive-frequency atomic polarization.

    P^(+) ~ Tr[D_+ rho]
    """

    return np.trace(

        D_DRIVE
        @
        rho
    )


# ============================================================
# 20. Full MTS response
# ============================================================

def response_at_detuning(
    delta,
    kv=0.0,
    beta=BETA,
    omega_pump=OMEGA_PUMP,
    omega_probe=OMEGA_PROBE,
    field_scale=1.0
):

    orders = perturbative_orders(

        delta=delta,

        kv=kv,

        beta=beta,

        omega_pump=omega_pump,

        omega_probe=omega_probe,

        field_scale=field_scale
    )


    rho3 = orders[3]


    zero = np.zeros(
        (N, N),
        dtype=complex
    )


    # --------------------------------------------------------
    # probe-direction generated sidebands
    # --------------------------------------------------------

    rho_plus = rho3.get(
        (+1, -1),
        zero
    )


    rho_minus = rho3.get(
        (-1, -1),
        zero
    )


    P_plus = positive_polarization(
        rho_plus
    )


    P_minus = positive_polarization(
        rho_minus
    )


    # Maxwell thin-sample proportionality

    E_plus = (
        -1j
        *
        P_plus
    )


    E_minus = (
        -1j
        *
        P_minus
    )


    E_c = (
        field_scale
        *
        omega_probe
    )


    Z = (

        E_plus
        *
        np.conj(
            E_c
        )

        +

        np.conj(
            E_minus
        )
        *
        E_c
    )


    return {

        "P_plus":
        P_plus,

        "P_minus":
        P_minus,

        "Z":
        Z,

        "orders":
        orders
    }


# ============================================================
# 21. Analysis
# ============================================================

def complex_center_slope(
    x,
    Z,
    window=0.7
):

    mask = (
        np.abs(x)
        <=
        window
    )


    xx = x[mask]

    ZZ = Z[mask]


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
        +
        1j
        *
        ci[-2]
    )


def refined_zero(
    x,
    y,
    window=1.5
):

    interp = PchipInterpolator(
        x,
        y
    )


    dense = np.linspace(
        -window,
        +window,
        6001
    )


    yy = interp(
        dense
    )


    candidates = []


    for i in range(
        len(dense) - 1
    ):

        if (
            yy[i]
            ==
            0
        ):

            candidates.append(
                dense[i]
            )


        elif (
            yy[i]
            *
            yy[i + 1]
            <
            0
        ):

            root = brentq(

                interp,

                dense[i],

                dense[i + 1]
            )


            candidates.append(
                root
            )


    if not candidates:

        return np.nan


    return min(
        candidates,
        key=abs
    )


def extrema_metrics(
    x,
    y,
    zero
):

    interp = PchipInterpolator(
        x,
        y
    )


    dense = np.linspace(
        x.min(),
        x.max(),
        12001
    )


    yy = interp(
        dense
    )


    prominence = (
        0.01
        *
        (
            yy.max()
            -
            yy.min()
        )
    )


    maxima, _ = find_peaks(
        yy,
        prominence=prominence
    )


    minima, _ = find_peaks(
        -yy,
        prominence=prominence
    )


    ext = np.sort(
        np.concatenate(
            [
                maxima,
                minima
            ]
        )
    )


    left = ext[
        dense[ext] < zero
    ]


    right = ext[
        dense[ext] > zero
    ]


    if (
        len(left) == 0
        or
        len(right) == 0
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )


    il = left[-1]

    ir = right[0]


    Vpp = abs(
        yy[ir]
        -
        yy[il]
    )


    sep = (
        dense[ir]
        -
        dense[il]
    )


    return (
        Vpp,
        sep,
        dense[il],
        dense[ir]
    )


# ============================================================
# 22. Main
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "Stage F1: real 87Rb F=2 -> F'=3 "
        "Zeeman-resolved rho^(3) MTS"
    )

    print(
        "=" * 72
    )


    print()
    print(
        "Hilbert dimension =",
        N
    )

    print(
        "Liouville dimension =",
        DIM
    )


    print()
    print(
        "87Rb parameters:"
    )

    print(
        "Gamma/(2pi) =",
        GAMMA_MHZ,
        "MHz"
    )

    print(
        "fm =",
        FM_MHZ,
        "MHz"
    )

    print(
        "fm/Gamma =",
        FM
    )

    print(
        "beta =",
        BETA
    )

    print(
        "pump Rabi/Gamma =",
        OMEGA_PUMP
    )

    print(
        "probe Rabi/Gamma =",
        OMEGA_PROBE
    )

    print(
        "polarization q =",
        POL_Q
    )

    print(
        "B =",
        B_GAUSS,
        "G"
    )

    print(
        "ground transit rate/Gamma =",
        GAMMA_TRANSIT
    )


    # --------------------------------------------------------
    # CG table
    # --------------------------------------------------------

    print()
    print(
        "pi-transition CG coefficients:"
    )


    for mg in MG:

        me = mg

        c = cg(
            mg,
            POL_Q,
            me
        )

        print(
            f"m_g={mg:+d}"
            f" -> "
            f"m_e={me:+d}"
            f"   CG={c:+.8f}"
            f"   strength={c*c:.8f}"
        )


    print()
    print(
        "spontaneous-decay consistency norm =",
        DECAY_ERROR
    )


    # --------------------------------------------------------
    # rho0 check
    # --------------------------------------------------------

    L0_center = build_L0(
        0.0
    )


    rho0_residual = np.linalg.norm(

        L0_center
        @
        vec(RHO0)
    )


    print(
        "rho0 steady-state residual =",
        rho0_residual
    )


    # --------------------------------------------------------
    # scan
    # --------------------------------------------------------

    print()
    print(
        "Calculating real-Rb rho^(3) scan..."
    )


    results = []


    for i, delta in enumerate(
        DELTA_SCAN
    ):

        print(
            f"{i+1:3d}/{len(DELTA_SCAN)}"
            f"   Delta/Gamma={delta:+.4f}"
        )


        results.append(

            response_at_detuning(
                delta
            )
        )


    P_PLUS = np.array(
        [
            r["P_plus"]
            for r in results
        ]
    )


    P_MINUS = np.array(
        [
            r["P_minus"]
            for r in results
        ]
    )


    Z = np.array(
        [
            r["Z"]
            for r in results
        ]
    )


    # --------------------------------------------------------
    # optimal demodulation
    # --------------------------------------------------------

    slope_complex = (
        complex_center_slope(
            DELTA_SCAN,
            Z
        )
    )


    phi_opt = np.angle(
        slope_complex
    )


    S_OPT = np.real(

        Z
        *
        np.exp(
            -1j
            *
            phi_opt
        )
    )


    zero = refined_zero(
        DELTA_SCAN,
        S_OPT
    )


    (
        Vpp,
        separation,
        x_left,
        x_right

    ) = extrema_metrics(
        DELTA_SCAN,
        S_OPT,
        zero
    )


    print()
    print(
        "MTS result:"
    )

    print(
        "complex center slope =",
        slope_complex
    )

    print(
        "|center slope| =",
        abs(
            slope_complex
        )
    )

    print(
        "optimum demodulation phase =",
        np.degrees(
            phi_opt
        ),
        "deg"
    )

    print(
        "zero crossing =",
        zero
    )

    print(
        "nearest-extrema Vpp =",
        Vpp
    )

    print(
        "nearest-extrema separation =",
        separation
    )


    # --------------------------------------------------------
    # beta=0 null test
    # --------------------------------------------------------

    if np.isfinite(
        x_right
    ):

        test_delta = x_right

    else:

        test_delta = 1.0


    ref = response_at_detuning(
        test_delta,
        beta=BETA
    )


    null = response_at_detuning(
        test_delta,
        beta=0.0
    )


    null_ratio = (

        abs(
            null["Z"]
        )

        /

        max(
            abs(
                ref["Z"]
            ),
            1e-300
        )
    )


    print()
    print(
        "Structural validation:"
    )

    print(
        "test detuning =",
        test_delta
    )

    print(
        "beta=0 null-test ratio =",
        null_ratio
    )


    # --------------------------------------------------------
    # third-order scaling
    # --------------------------------------------------------

    scales = np.array(
        [
            0.50,
            0.75,
            1.00
        ]
    )


    P_scale = []

    Z_scale = []


    if RUN_SCALING_CHECK:

        for s in scales:

            rr = response_at_detuning(

                test_delta,

                field_scale=s
            )


            P_scale.append(

                max(

                    abs(
                        rr["P_plus"]
                    ),

                    abs(
                        rr["P_minus"]
                    )
                )
            )


            Z_scale.append(
                abs(
                    rr["Z"]
                )
            )


        P_scale = np.asarray(
            P_scale
        )

        Z_scale = np.asarray(
            Z_scale
        )


        print()
        print(
            "Third-order scaling:"
        )

        print(
            f"{'s':>8}"
            f"{'|P3|/s^3':>20}"
            f"{'|Z|/s^4':>20}"
        )


        for (
            s,
            p,
            z

        ) in zip(
            scales,
            P_scale,
            Z_scale
        ):

            print(
                f"{s:8.3f}"
                f"{p/s**3:20.10e}"
                f"{z/s**4:20.10e}"
            )


    # --------------------------------------------------------
    # symmetry
    # --------------------------------------------------------

    symmetry_error = (

        np.linalg.norm(
            S_OPT
            +
            S_OPT[::-1]
        )

        /

        max(
            np.linalg.norm(
                S_OPT
            ),
            1e-30
        )
    )


    print()
    print(
        "odd-symmetry error =",
        symmetry_error
    )


    print(
        "All finite =",
        (
            np.all(
                np.isfinite(
                    P_PLUS
                )
            )

            and

            np.all(
                np.isfinite(
                    P_MINUS
                )
            )

            and

            np.all(
                np.isfinite(
                    Z
                )
            )
        )
    )


    # ========================================================
    # Figure 1
    # CG strengths
    # ========================================================

    strengths = np.array(
        [
            cg(
                mg,
                POL_Q,
                mg
            )**2

            for mg in MG
        ]
    )


    plt.figure(
        figsize=(7, 4.5)
    )


    plt.stem(
        MG,
        strengths
    )


    plt.xlabel(
        r"$m_F$"
    )

    plt.ylabel(
        r"$|C_{F m,1q}^{F'm'}|^2$"
    )

    plt.title(
        r"$^{87}$Rb "
        r"$F=2\rightarrow F'=3$ "
        r"$\pi$-transition strengths"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    # ========================================================
    # Figure 2
    # generated sidebands
    # ========================================================

    scale_P = max(

        np.max(
            np.abs(
                P_PLUS
            )
        ),

        np.max(
            np.abs(
                P_MINUS
            )
        ),

        1e-300
    )


    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        DELTA_SCAN,
        np.abs(
            P_PLUS
        )
        /
        scale_P,
        label=r"$\omega+\Omega_m$"
    )


    plt.plot(
        DELTA_SCAN,
        np.abs(
            P_MINUS
        )
        /
        scale_P,
        label=r"$\omega-\Omega_m$"
    )


    plt.axvline(
        0,
        linestyle="--",
        linewidth=0.8
    )


    plt.xlabel(
        r"Laser detuning $\Delta/\Gamma$"
    )

    plt.ylabel(
        "Generated third-order polarization"
    )

    plt.title(
        r"$^{87}$Rb Zeeman-resolved "
        r"$P_\pm^{(3)}$"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    # ========================================================
    # Figure 3
    # complex heterodyne
    # ========================================================

    scale_Z = max(

        np.max(
            np.abs(
                Z
            )
        ),

        1e-300
    )


    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        DELTA_SCAN,
        np.real(Z)
        /
        scale_Z,
        label=r"$\mathrm{Re}(Z)$"
    )


    plt.plot(
        DELTA_SCAN,
        np.imag(Z)
        /
        scale_Z,
        label=r"$\mathrm{Im}(Z)$"
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
        r"Laser detuning $\Delta/\Gamma$"
    )

    plt.ylabel(
        "Complex RF response"
    )

    plt.title(
        r"$^{87}$Rb Zeeman-resolved "
        "MTS heterodyne response"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    # ========================================================
    # Figure 4
    # error signal
    # ========================================================

    scale_S = max(

        np.max(
            np.abs(
                S_OPT
            )
        ),

        1e-300
    )


    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        DELTA_SCAN,
        S_OPT
        /
        scale_S,
        label=(
            f"phi = "
            f"{np.degrees(phi_opt):.1f} deg"
        )
    )


    plt.axhline(
        0,
        linewidth=0.8
    )


    plt.axvline(
        0,
        linestyle="--",
        linewidth=0.8,
        label="F=2 -> F'=3 resonance"
    )


    if np.isfinite(
        zero
    ):

        plt.axvline(
            zero,
            linestyle=":",
            linewidth=0.8,
            label=(
                f"lock zero = "
                f"{zero:.4g}"
            )
        )


    plt.xlabel(
        r"Laser detuning $\Delta/\Gamma$"
    )

    plt.ylabel(
        "Normalized MTS error signal"
    )

    plt.title(
        r"$^{87}$Rb "
        r"$F=2\rightarrow F'=3$ "
        "third-order MTS"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    # ========================================================
    # Figure 5
    # cubic scaling
    # ========================================================

    if RUN_SCALING_CHECK:

        plt.figure(
            figsize=(7, 5)
        )


        plt.loglog(
            scales,
            P_scale
            /
            P_scale[-1],
            "o-",
            label=r"$|P^{(3)}|$"
        )


        plt.loglog(
            scales,
            scales**3,
            "--",
            label=r"$s^3$ reference"
        )


        plt.xlabel(
            "Common field-amplitude scale s"
        )

        plt.ylabel(
            "Relative third-order polarization"
        )

        plt.title(
            "Real-Rb third-order scaling"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()


    plt.show()


if __name__ == "__main__":

    main()