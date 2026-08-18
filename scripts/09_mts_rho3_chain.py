import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.special import jv
from scipy.signal import find_peaks


# ============================================================
# Stage D-1
#
# Explicit frequency-domain perturbative MTS chain:
#
# rho^(0)
#   -> rho^(1)
#   -> rho^(2)
#   -> rho^(3)
#   -> P^(3)
#   -> generated sidebands E_+, E_-
#   -> RF heterodyne envelope Z
#   -> lock-in demodulated MTS error signal
#
# Generic cascade:
#
#       |c>
#        |
#       |b>
#        |
#       |a>
#
# All frequencies are normalized to gamma_ref.
# ============================================================


# ============================================================
# 1. Parameters
# ============================================================

GAMMA_REF = 1.0

# 1993-style normalized parameters
FM = 0.60
BETA = 0.28

# (omega_ac / 2 - omega_ab) / gamma_ref
LEVEL_OFFSET = 20.0


# ------------------------------------------------------------
# Phenomenological relaxation assumptions
#
# These are model assumptions rather than uniquely fixed
# experimental values in the old generic three-level paper.
# ------------------------------------------------------------

GAMMA_B_POP = 1.0
GAMMA_C_POP = 1.0

GAMMA_AB = 1.0
GAMMA_BC = 1.0
GAMMA_AC = 1.0


# ------------------------------------------------------------
# Relative dipole matrix elements
# ------------------------------------------------------------

MU_AB = 1.0
MU_BC = 1.0


# ------------------------------------------------------------
# Weak-field normalized Rabi amplitudes
#
# We intentionally remain in a weak-field regime here because
# this script explicitly truncates the response at third order.
# ------------------------------------------------------------

OMEGA_PUMP = 0.15
OMEGA_PROBE = 0.03


# ------------------------------------------------------------
# Doppler parameter
#
# Stage D-1 deliberately sets kv = 0.
#
# Later we will wrap the same response function in the already
# verified Maxwell velocity average from Stage C.
# ------------------------------------------------------------

KV = 0.0


# ------------------------------------------------------------
# Scan coordinate
#
# Define
#
# x = Delta_ab - LEVEL_OFFSET
#
# so that:
#
# Delta_ab = LEVEL_OFFSET + x
#
# Delta_ac = 2 x
#
# Therefore x = 0 is the two-photon resonance.
# ------------------------------------------------------------

X = np.linspace(-4.0, 4.0, 401)


N_PATH_PRINT = 8


# ============================================================
# 2. Matrix utilities
# ============================================================

NLEV = 3
DIM = NLEV * NLEV


def vec(rho):
    """
    Column-major vectorization.
    """
    return np.asarray(
        rho,
        dtype=complex
    ).reshape(
        DIM,
        order="F"
    )


def mat(v):
    """
    Inverse of vec().
    """
    return np.asarray(
        v,
        dtype=complex
    ).reshape(
        (NLEV, NLEV),
        order="F"
    )


def dm_index(i, j):
    """
    Column-major index of rho_ij.
    """
    return i + j * NLEV


def commutator_superoperator(H):
    """
    Construct

        M(H) rho = -i [H, rho]

    in vectorized form.
    """

    I = np.eye(
        NLEV,
        dtype=complex
    )

    return -1j * (
        np.kron(I, H)
        -
        np.kron(H.T, I)
    )


# ============================================================
# 3. Unperturbed Liouvillian L0
# ============================================================

def build_L0(
        delta_ab,
        delta_ac
):
    """
    Phenomenological three-level cascade Liouvillian.

    Rotating-frame energies:

        eps_a = 0
        eps_b = -Delta_ab
        eps_c = -Delta_ac

    Then for example

        rho_ab

    contains approximately

        gamma_ab + i Delta_ab

    as its resonance denominator.

    Population relaxation is

        c -> b -> a.

    Coherence decay rates are inserted independently,
    following the phenomenological spirit of the 1993 model.
    """

    L = np.zeros(
        (DIM, DIM),
        dtype=complex
    )


    # --------------------------------------------------------
    # populations
    # --------------------------------------------------------

    iaa = dm_index(0, 0)
    ibb = dm_index(1, 1)
    icc = dm_index(2, 2)


    # b -> a
    L[iaa, ibb] += GAMMA_B_POP
    L[ibb, ibb] += -GAMMA_B_POP


    # c -> b
    L[ibb, icc] += GAMMA_C_POP
    L[icc, icc] += -GAMMA_C_POP


    # --------------------------------------------------------
    # rotating-frame energies
    # --------------------------------------------------------

    eps = np.array(
        [
            0.0,
            -delta_ab,
            -delta_ac
        ],
        dtype=float
    )


    gamma_coh = {

        (0, 1): GAMMA_AB,
        (1, 0): GAMMA_AB,

        (1, 2): GAMMA_BC,
        (2, 1): GAMMA_BC,

        (0, 2): GAMMA_AC,
        (2, 0): GAMMA_AC,
    }


    # --------------------------------------------------------
    # coherences
    # --------------------------------------------------------

    for i in range(NLEV):

        for j in range(NLEV):

            if i == j:
                continue

            idx = dm_index(i, j)

            L[idx, idx] += (
                -gamma_coh[(i, j)]
                -
                1j * (
                    eps[i]
                    -
                    eps[j]
                )
            )


    return L


# ============================================================
# 4. Solve one perturbative Fourier component
# ============================================================

def solve_correction(
        L0,
        slow_frequency,
        source
):
    """
    Convention:

        rho_qk(t)
        ~ exp(-i * slow_frequency * t)

    therefore

        (-i Omega I - L0) rho = source.

    All perturbative corrections n >= 1 obey

        Tr[rho^(n)] = 0.

    The trace equation is appended explicitly.

    This is especially useful for the q = 0 population block,
    where L0 itself contains the steady-state null eigenvalue.
    """

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


    b = vec(source)


    # Trace constraint
    tr_row = np.zeros(
        DIM,
        dtype=complex
    )


    for i in range(NLEV):

        tr_row[
            dm_index(i, i)
        ] = 1.0


    A_aug = np.vstack(
        [
            A,
            tr_row[None, :]
        ]
    )


    b_aug = np.concatenate(
        [
            b,
            np.array(
                [0.0 + 0.0j]
            )
        ]
    )


    solution, *_ = np.linalg.lstsq(
        A_aug,
        b_aug,
        rcond=None
    )


    return mat(solution)


# ============================================================
# 5. Optical field modes
# ============================================================

@dataclass(frozen=True)
class FieldMode:

    name: str

    # modulation harmonic:
    # omega + n Omega_m
    n: int

    # propagation direction:
    #
    # pump  = +1
    # probe = -1
    ksign: int

    # normalized Rabi amplitude
    amp: complex


def make_field_modes(
        beta=BETA,
        omega_pump=OMEGA_PUMP,
        omega_probe=OMEGA_PROBE,
        field_scale=1.0
):
    """
    Pump:

        phase modulated

        n = -1, 0, +1.

    Probe:

        unmodulated,
        counterpropagating.

    Note carefully:

        J_-1(beta) = -J_1(beta).

    Therefore the sign of the negative sideband must NOT
    be manually discarded.
    """

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
            jv(-1, beta)
        ),

        FieldMode(
            "pump_0",
            0,
            +1,
            field_scale
            *
            omega_pump
            *
            jv(0, beta)
        ),

        FieldMode(
            "pump_+1",
            +1,
            +1,
            field_scale
            *
            omega_pump
            *
            jv(+1, beta)
        ),
    ]


# ============================================================
# 6. Interaction branches
# ============================================================

@dataclass(frozen=True)
class InteractionBranch:

    label: str

    dn: int
    dk: int

    M: np.ndarray


def make_interaction_branches(
        modes
):
    """
    Under RWA,

        H_I
        =
        -1/2 [
            Omega(t) S_+
            +
            Omega*(t) S_-
        ].

    A positive-frequency field interaction carries:

        (+n, +k),

    while the Hermitian-conjugate interaction carries:

        (-n, -k).
    """


    S_plus = np.zeros(
        (NLEV, NLEV),
        dtype=complex
    )


    # a -> b
    S_plus[1, 0] = MU_AB


    # b -> c
    S_plus[2, 1] = MU_BC


    S_minus = S_plus.conj().T


    branches = []


    for mode in modes:


        # positive-frequency interaction
        H_plus = (
            -0.5
            *
            mode.amp
            *
            S_plus
        )


        # complex-conjugate interaction
        H_minus = (
            -0.5
            *
            np.conj(mode.amp)
            *
            S_minus
        )


        branches.append(

            InteractionBranch(

                mode.name + "[+]",

                mode.n,

                mode.ksign,

                commutator_superoperator(
                    H_plus
                )
            )
        )


        branches.append(

            InteractionBranch(

                mode.name + "[-]",

                -mode.n,

                -mode.ksign,

                commutator_superoperator(
                    H_minus
                )
            )
        )


    return branches


# ============================================================
# 7. Explicit perturbative recursion
# ============================================================

def perturbative_orders(
        x,
        beta=BETA,
        omega_pump=OMEGA_PUMP,
        omega_probe=OMEGA_PROBE,
        field_scale=1.0,
        kv=KV
):
    """
    Explicitly generate

        rho^(1)
        rho^(2)
        rho^(3).

    Dictionary key:

        (q, k)

    q:
        modulation harmonic

    k:
        accumulated propagation-direction index.

    For a moving atom:

        slow frequency
        =
        q FM - k kv.

    At third order the MTS-generated sidebands
    propagating in the PROBE direction are

        (q = +1, k = -1)

    and

        (q = -1, k = -1).
    """


    delta_ab = (
        LEVEL_OFFSET
        +
        x
    )


    # two-photon detuning
    delta_ac = (
        2.0
        *
        x
    )


    L0 = build_L0(
        delta_ab,
        delta_ac
    )


    modes = make_field_modes(

        beta=beta,

        omega_pump=omega_pump,

        omega_probe=omega_probe,

        field_scale=field_scale
    )


    branches = make_interaction_branches(
        modes
    )


    # --------------------------------------------------------
    # zeroth order
    # --------------------------------------------------------

    rho0 = np.zeros(
        (NLEV, NLEV),
        dtype=complex
    )


    rho0[0, 0] = 1.0


    orders = {

        0: {

            (0, 0):

            rho0
        }
    }


    # --------------------------------------------------------
    # first, second, third order
    # --------------------------------------------------------

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
                    np.linalg.norm(source)
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
                            (
                                NLEV,
                                NLEV
                            ),
                            dtype=complex
                        )
                    )


                source_dict[key] += source


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
# 8. rho^(3) -> P^(3)
# ============================================================

def positive_frequency_polarization(
        rho
):
    """
    Normalized positive-frequency polarization:

        P^(+)
        ~
        mu_ab rho_ba
        +
        mu_bc rho_cb.

    Absolute factors such as

        N,
        absolute mu,
        epsilon_0,
        cell length

    are deliberately omitted.

    They multiply every normalized line shape by a common
    constant and are not fixed uniquely by the generic paper.
    """

    return (

        MU_AB
        *
        rho[1, 0]

        +

        MU_BC
        *
        rho[2, 1]
    )


# ============================================================
# 9. P^(3) -> E_+ / E_- -> RF heterodyne
# ============================================================

def response_at_x(
        x,
        beta=BETA,
        omega_pump=OMEGA_PUMP,
        omega_probe=OMEGA_PROBE,
        field_scale=1.0,
        kv=KV
):


    orders = perturbative_orders(

        x=x,

        beta=beta,

        omega_pump=omega_pump,

        omega_probe=omega_probe,

        field_scale=field_scale,

        kv=kv
    )


    rho3 = orders[3]


    zero_matrix = np.zeros(
        (NLEV, NLEV),
        dtype=complex
    )


    # --------------------------------------------------------
    # phase-matched probe-direction generated sidebands
    # --------------------------------------------------------

    rho_plus = rho3.get(
        (+1, -1),
        zero_matrix
    )


    rho_minus = rho3.get(
        (-1, -1),
        zero_matrix
    )


    P_plus = (
        positive_frequency_polarization(
            rho_plus
        )
    )


    P_minus = (
        positive_frequency_polarization(
            rho_minus
        )
    )


    # --------------------------------------------------------
    # Thin-sample Maxwell propagation:
    #
    # E_r ~ -i C P^(3)
    #
    # common constant C is omitted.
    # --------------------------------------------------------

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


    # probe carrier = heterodyne local optical oscillator
    E_c = (
        field_scale
        *
        omega_probe
    )


    # --------------------------------------------------------
    # RF component at Omega_m
    #
    # Z =
    # E_(omega+Omega_m) E_c*
    # +
    # E_(omega-Omega_m)* E_c
    # --------------------------------------------------------

    Z = (

        E_plus
        *
        np.conj(E_c)

        +

        np.conj(E_minus)
        *
        E_c
    )


    return {

        "orders":
        orders,

        "rho3_plus":
        rho_plus,

        "rho3_minus":
        rho_minus,

        "P_plus":
        P_plus,

        "P_minus":
        P_minus,

        "E_plus":
        E_plus,

        "E_minus":
        E_minus,

        "Z":
        Z,
    }


# ============================================================
# 10. Lock-point / peak diagnostics
# ============================================================

def zero_crossing_near(
        x,
        y,
        center=0.0,
        window=1.5
):


    mask = (
        np.abs(
            x
            -
            center
        )
        <=
        window
    )


    xx = x[mask]
    yy = y[mask]


    candidates = []


    for i in range(
        len(xx) - 1
    ):


        if (
            yy[i]
            *
            yy[i + 1]
            <
            0
        ):


            x0 = (

                xx[i]

                -

                yy[i]
                *
                (
                    xx[i + 1]
                    -
                    xx[i]
                )
                /
                (
                    yy[i + 1]
                    -
                    yy[i]
                )
            )


            candidates.append(
                x0
            )


    if not candidates:

        j = np.argmin(
            np.abs(yy)
        )

        return float(
            xx[j]
        )


    return float(

        min(

            candidates,

            key=lambda z:
            abs(
                z
                -
                center
            )
        )
    )


def nearest_extrema_metrics(
        x,
        y,
        zero
):


    peaks, _ = find_peaks(
        y
    )


    troughs, _ = find_peaks(
        -y
    )


    extrema = np.sort(

        np.unique(

            np.concatenate(
                [
                    peaks,
                    troughs
                ]
            )
        )
    )


    left = [

        i
        for i in extrema

        if x[i]
        <
        zero
    ]


    right = [

        i
        for i in extrema

        if x[i]
        >
        zero
    ]


    if (
        not left
        or
        not right
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )


    i_left = left[-1]
    i_right = right[0]


    Vpp = abs(

        y[i_right]
        -
        y[i_left]
    )


    separation = (

        x[i_right]
        -
        x[i_left]
    )


    return (

        float(Vpp),

        float(separation),

        float(x[i_left]),

        float(x[i_right])
    )


# ============================================================
# 11. Explicit individual FWM-path decomposition
# ============================================================

def pathway_decomposition(
        x,
        target,
        beta=BETA,
        omega_pump=OMEGA_PUMP,
        omega_probe=OMEGA_PROBE,
        kv=KV
):
    """
    Unlike perturbative_orders(), this routine does NOT merge
    paths that end at the same (q,k).

    Every three-interaction sequence is propagated separately.

    This lets us explicitly inspect terms corresponding to

        E_l E_m* E_n

    and verify which ones finally satisfy the target
    frequency/wavevector condition.
    """


    delta_ab = (
        LEVEL_OFFSET
        +
        x
    )


    delta_ac = (
        2.0
        *
        x
    )


    L0 = build_L0(
        delta_ab,
        delta_ac
    )


    modes = make_field_modes(

        beta=beta,

        omega_pump=omega_pump,

        omega_probe=omega_probe
    )


    branches = make_interaction_branches(
        modes
    )


    rho0 = np.zeros(
        (NLEV, NLEV),
        dtype=complex
    )


    rho0[0, 0] = 1.0


    states = {

        (0, 0):
        [
            (
                [],
                rho0
            )
        ]
    }


    for order in (
        1,
        2,
        3
    ):


        new_states = {}


        for (
            q0,
            k0
        ), contributions in states.items():


            for (
                path,
                rho_previous
            ) in contributions:


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
                        np.linalg.norm(source)
                        <
                        1e-18
                    ):
                        continue


                    q = (
                        q0
                        +
                        branch.dn
                    )


                    k = (
                        k0
                        +
                        branch.dk
                    )


                    slow_frequency = (

                        q
                        *
                        FM

                        -

                        k
                        *
                        kv
                    )


                    rho_new = (
                        solve_correction(

                            L0,

                            slow_frequency,

                            source
                        )
                    )


                    if (
                        np.linalg.norm(rho_new)
                        <
                        1e-18
                    ):
                        continue


                    new_states.setdefault(

                        (q, k),

                        []

                    ).append(

                        (
                            path
                            +
                            [
                                branch.label
                            ],

                            rho_new
                        )
                    )


        states = new_states


    entries = []


    for (
        path,
        rho
    ) in states.get(
        target,
        []
    ):


        P = (
            positive_frequency_polarization(
                rho
            )
        )


        entries.append(
            (
                abs(P),
                path,
                P
            )
        )


    entries.sort(
        key=lambda item:
        item[0],
        reverse=True
    )


    return entries


# ============================================================
# 12. Main program
# ============================================================

def main():


    print(
        "=" * 72
    )


    print(
        "Stage D-1: explicit rho^(3) -> P^(3) "
        "-> sidebands -> heterodyne"
    )


    print(
        "=" * 72
    )


    # --------------------------------------------------------
    # parameter report
    # --------------------------------------------------------

    print(
        "\nNormalized parameters:"
    )


    print(
        f"fm/gamma_ref = {FM}"
    )


    print(
        f"beta = {BETA}"
    )


    print(
        "(omega_ac/2 - omega_ab)"
        f"/gamma_ref = {LEVEL_OFFSET}"
    )


    print(
        f"gamma_ab/gamma_ref = {GAMMA_AB}"
    )


    print(
        f"gamma_bc/gamma_ref = {GAMMA_BC}"
    )


    print(
        f"gamma_ac/gamma_ref = {GAMMA_AC}"
    )


    print(
        f"Omega_pump/gamma_ref = {OMEGA_PUMP}"
    )


    print(
        f"Omega_probe/gamma_ref = {OMEGA_PROBE}"
    )


    print(
        f"kv/gamma_ref = {KV}"
    )


    print(
        "\nx = Delta_ab/gamma_ref - LEVEL_OFFSET"
    )


    print(
        "Delta_ac/gamma_ref = 2*x"
    )


    print(
        "two-photon resonance: x = 0"
    )


    # --------------------------------------------------------
    # field spectrum
    # --------------------------------------------------------

    print(
        "\nOptical modes:"
    )


    modes = make_field_modes()


    for m in modes:

        print(

            f"{m.name:10s}  "

            f"n={m.n:+d}  "

            f"k={m.ksign:+d}  "

            f"amp="
            f"{m.amp.real:+.8f}"
            f"{m.amp.imag:+.8f}j"
        )


    # --------------------------------------------------------
    # detuning scan
    # --------------------------------------------------------

    print(
        "\nCalculating explicit third-order scan..."
    )


    results = [

        response_at_x(x)

        for x in X
    ]


    P_plus = np.array(
        [
            r["P_plus"]
            for r in results
        ]
    )


    P_minus = np.array(
        [
            r["P_minus"]
            for r in results
        ]
    )


    E_plus = np.array(
        [
            r["E_plus"]
            for r in results
        ]
    )


    E_minus = np.array(
        [
            r["E_minus"]
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
    # optimum demodulation phase
    # --------------------------------------------------------

    dZdx = np.gradient(
        Z,
        X
    )


    i0 = int(
        np.argmin(
            np.abs(X)
        )
    )


    complex_slope = dZdx[
        i0
    ]


    phi_opt = np.angle(
        complex_slope
    )


    phi_opt_deg = np.degrees(
        phi_opt
    )


    S_opt = np.real(

        Z
        *
        np.exp(
            -1j
            *
            phi_opt
        )
    )


    center_slope = (
        np.gradient(
            S_opt,
            X
        )[i0]
    )


    # --------------------------------------------------------
    # lock point and central extrema
    # --------------------------------------------------------

    zero = zero_crossing_near(

        X,

        S_opt,

        center=0.0,

        window=1.5
    )


    (
        Vpp,
        peak_sep,
        x_left,
        x_right
    ) = nearest_extrema_metrics(

        X,
        S_opt,
        zero
    )


    print(
        "\nThird-order heterodyne result:"
    )


    print(
        "complex central slope =",
        complex_slope
    )


    print(
        "|complex central slope| =",
        abs(complex_slope)
    )


    print(
        "optimum demodulation phase =",
        phi_opt_deg,
        "deg"
    )


    print(
        "demodulated center slope =",
        center_slope
    )


    print(
        "refined zero crossing =",
        zero
    )


    print(
        "nearest-extrema Vpp =",
        Vpp
    )


    print(
        "nearest-extrema separation =",
        peak_sep
    )


    print(
        "left/right extrema =",
        x_left,
        x_right
    )


    # ========================================================
    # 13. Structural tests
    # ========================================================

    if np.isfinite(
        x_right
    ):

        x_test = x_right

    else:

        x_test = 0.3


    reference = response_at_x(
        x_test
    )


    ref_Z = abs(
        reference["Z"]
    )


    # --------------------------------------------------------
    # beta = 0 test
    #
    # No modulation sidebands -> no transferred modulation
    # at q = +/-1.
    # --------------------------------------------------------

    beta_zero = response_at_x(

        x_test,

        beta=0.0
    )


    null_ratio = (

        abs(
            beta_zero["Z"]
        )

        /

        max(
            ref_Z,
            1e-300
        )
    )


    print(
        "\nStructural validation:"
    )


    print(
        f"test detuning x = {x_test}"
    )


    print(
        "beta=0 null-test ratio =",
        null_ratio
    )


    # --------------------------------------------------------
    # all-field amplitude scaling
    #
    # rho^(3) / P^(3) must scale as s^3.
    #
    # Heterodyne adds another optical carrier field:
    #
    # Z ~ s^4.
    # --------------------------------------------------------

    scales = np.array(
        [
            0.4,
            0.6,
            0.8,
            1.0
        ]
    )


    P3_values = []

    Z_values = []


    for s in scales:


        rr = response_at_x(

            x_test,

            field_scale=s
        )


        P3_values.append(

            max(
                abs(
                    rr["P_plus"]
                ),

                abs(
                    rr["P_minus"]
                )
            )
        )


        Z_values.append(

            abs(
                rr["Z"]
            )
        )


    P3_values = np.array(
        P3_values
    )


    Z_values = np.array(
        Z_values
    )


    print(
        "\nAll-field amplitude scaling:"
    )


    print(
        "   s        |P3|/s^3             |Z|/s^4"
    )


    for (
        s,
        p,
        z
    ) in zip(
        scales,
        P3_values,
        Z_values
    ):


        print(

            f"{s:6.3f}   "

            f"{p/s**3: .12e}   "

            f"{z/s**4: .12e}"
        )


    # --------------------------------------------------------
    # pump/probe INTENSITY scaling
    #
    # Omega ~ sqrt(I).
    #
    # Strict third-order heterodyne response therefore obeys:
    #
    # Z ~ I_pump * I_probe
    # --------------------------------------------------------

    intensity_scale = np.array(
        [
            0.25,
            0.50,
            1.00,
            2.00,
            4.00
        ]
    )


    pump_signal = []

    probe_signal = []


    for r in intensity_scale:


        pump_signal.append(

            abs(

                response_at_x(

                    x_test,

                    omega_pump=
                    OMEGA_PUMP
                    *
                    np.sqrt(r),

                    omega_probe=
                    OMEGA_PROBE

                )["Z"]
            )
        )


        probe_signal.append(

            abs(

                response_at_x(

                    x_test,

                    omega_pump=
                    OMEGA_PUMP,

                    omega_probe=
                    OMEGA_PROBE
                    *
                    np.sqrt(r)

                )["Z"]
            )
        )


    pump_signal = np.array(
        pump_signal
    )


    probe_signal = np.array(
        probe_signal
    )


    print(
        "\nStrict third-order intensity scaling:"
    )


    print(
        " I/I0         Zpump/(I/I0)       Zprobe/(I/I0)"
    )


    for (
        r,
        zp,
        zs
    ) in zip(
        intensity_scale,
        pump_signal,
        probe_signal
    ):


        print(

            f"{r:7.3f}   "

            f"{zp/r: .12e}   "

            f"{zs/r: .12e}"
        )


    # ========================================================
    # 14. Explicit FWM pathway decomposition
    # ========================================================

    print(
        "\nDominant third-order FWM pathways:"
    )


    targets = [

        (
            (+1, -1),
            "omega + Omega_m"
        ),

        (
            (-1, -1),
            "omega - Omega_m"
        )
    ]


    for (
        target,
        name
    ) in targets:


        entries = pathway_decomposition(

            x_test,

            target=target
        )


        print(
            "\nTarget:",
            name
        )


        print(
            "(q,k) =",
            target
        )


        print(
            "number of nonzero paths =",
            len(entries)
        )


        for rank, (
            magnitude,
            path,
            P
        ) in enumerate(

            entries[
                :N_PATH_PRINT
            ],

            start=1
        ):


            path_text = (
                " -> ".join(path)
            )


            print(

                f"{rank:2d}. "

                f"|P_path|="
                f"{magnitude:.6e}"

                f"   P="
                f"{P.real:+.4e}"
                f"{P.imag:+.4e}j"
            )


            print(
                "    ",
                path_text
            )


    # ========================================================
    # 15. Numerical sanity
    # ========================================================

    all_finite = (

        np.all(
            np.isfinite(
                P_plus
            )
        )

        and

        np.all(
            np.isfinite(
                P_minus
            )
        )

        and

        np.all(
            np.isfinite(
                Z
            )
        )
    )


    print(
        "\nAll finite =",
        all_finite
    )


    # ========================================================
    # 16. Figures
    #
    # IMPORTANT:
    #
    # all paired curves use COMMON normalization.
    # ========================================================


    # --------------------------------------------------------
    # Figure 1
    # Generated third-order probe-direction sidebands
    # --------------------------------------------------------

    common_sideband_norm = max(

        np.max(
            np.abs(
                E_plus
            )
        ),

        np.max(
            np.abs(
                E_minus
            )
        ),

        1e-300
    )


    plt.figure()


    plt.plot(

        X,

        np.abs(E_plus)
        /
        common_sideband_norm,

        label=
        r"$\omega+\Omega_m$"
    )


    plt.plot(

        X,

        np.abs(E_minus)
        /
        common_sideband_norm,

        label=
        r"$\omega-\Omega_m$"
    )


    plt.axvline(
        0.0,
        linestyle="--"
    )


    plt.xlabel(
        r"Two-photon-centered detuning "
        r"$x/\gamma_{\rm ref}$"
    )


    plt.ylabel(
        "Generated sideband magnitude "
        "(common normalization)"
    )


    plt.title(
        r"Probe-direction sidebands from $\rho^{(3)}$"
    )


    plt.legend()

    plt.tight_layout()


    # --------------------------------------------------------
    # Figure 2
    # Complex RF heterodyne envelope
    # --------------------------------------------------------

    common_Z_norm = max(

        np.max(
            np.abs(
                np.real(Z)
            )
        ),

        np.max(
            np.abs(
                np.imag(Z)
            )
        ),

        1e-300
    )


    plt.figure()


    plt.plot(

        X,

        np.real(Z)
        /
        common_Z_norm,

        label=r"$\mathrm{Re}(Z)$"
    )


    plt.plot(

        X,

        np.imag(Z)
        /
        common_Z_norm,

        label=r"$\mathrm{Im}(Z)$"
    )


    plt.axhline(
        0.0,
        linewidth=0.8
    )


    plt.axvline(
        0.0,
        linestyle="--"
    )


    plt.xlabel(
        r"Two-photon-centered detuning "
        r"$x/\gamma_{\rm ref}$"
    )


    plt.ylabel(
        "RF quadrature "
        "(common normalization)"
    )


    plt.title(
        "Complex MTS heterodyne envelope"
    )


    plt.legend()

    plt.tight_layout()


    # --------------------------------------------------------
    # Figure 3
    # Optimum-phase error signal
    # --------------------------------------------------------

    error_norm = max(

        np.max(
            np.abs(
                S_opt
            )
        ),

        1e-300
    )


    plt.figure()


    plt.plot(

        X,

        S_opt
        /
        error_norm,

        label=
        f"optimum phase = "
        f"{phi_opt_deg:.1f} deg"
    )


    plt.axhline(
        0.0,
        linewidth=0.8
    )


    plt.axvline(

        0.0,

        linestyle="--",

        label=
        "two-photon resonance"
    )


    plt.axvline(

        zero,

        linestyle=":",

        label=
        f"lock zero = {zero:.4g}"
    )


    plt.xlabel(
        r"Two-photon-centered detuning "
        r"$x/\gamma_{\rm ref}$"
    )


    plt.ylabel(
        "Demodulated MTS signal"
    )


    plt.title(
        "MTS error signal from explicit third-order chain"
    )


    plt.legend()

    plt.tight_layout()


    # --------------------------------------------------------
    # Figure 4
    # Demodulation phase
    # --------------------------------------------------------

    phases_deg = [
        0,
        30,
        60,
        90
    ]


    phase_curves = []


    for phase_deg in phases_deg:


        phase = np.deg2rad(
            phase_deg
        )


        curve = np.real(

            Z

            *

            np.exp(
                -1j
                *
                phase
            )
        )


        phase_curves.append(
            curve
        )


    common_phase_norm = max(

        max(

            np.max(
                np.abs(curve)
            )

            for curve
            in phase_curves
        ),

        1e-300
    )


    plt.figure()


    for (
        phase_deg,
        curve
    ) in zip(

        phases_deg,
        phase_curves
    ):


        plt.plot(

            X,

            curve
            /
            common_phase_norm,

            label=
            f"{phase_deg} deg"
        )


    plt.axhline(
        0.0,
        linewidth=0.8
    )


    plt.axvline(
        0.0,
        linestyle="--"
    )


    plt.xlabel(
        r"Two-photon-centered detuning "
        r"$x/\gamma_{\rm ref}$"
    )


    plt.ylabel(
        "MTS signal "
        "(common normalization)"
    )


    plt.title(
        "Demodulation-phase dependence"
    )


    plt.legend()

    plt.tight_layout()


    # --------------------------------------------------------
    # Figure 5
    # Strict perturbative intensity scaling
    # --------------------------------------------------------

    plt.figure()


    plt.loglog(

        intensity_scale,

        pump_signal
        /
        pump_signal[2],

        "o-",

        label=
        "pump scan"
    )


    plt.loglog(

        intensity_scale,

        probe_signal
        /
        probe_signal[2],

        "s-",

        label=
        "probe scan"
    )


    plt.loglog(

        intensity_scale,

        intensity_scale,

        "--",

        label=
        r"linear reference $\propto I$"
    )


    plt.xlabel(
        r"Relative intensity $I/I_0$"
    )


    plt.ylabel(
        "Relative heterodyne magnitude"
    )


    plt.title(
        "Strict third-order intensity scaling"
    )


    plt.legend()

    plt.tight_layout()


    plt.show()

# ============================================================
# Public API for later stages
#
# These wrappers DO NOT change the validated Stage-09 physics.
# They only expose clean interfaces for Stage E and later.
# ============================================================

def rho3_sidebands(
    x,
    beta=BETA,
    omega_pump=OMEGA_PUMP,
    omega_probe=OMEGA_PROBE,
    field_scale=1.0,
    kv=KV
):
    """
    Return the probe-direction third-order polarizations

        P_plus  : omega + Omega_m
        P_minus : omega - Omega_m

    Parameters
    ----------
    x :
        two-photon-centered detuning / gamma_ref

    kv :
        normalized Doppler shift

            kv / gamma_ref

        Sign and propagation-direction dependence are handled
        internally through the (q, k) perturbative labels.
    """

    result = response_at_x(
        x=x,
        beta=beta,
        omega_pump=omega_pump,
        omega_probe=omega_probe,
        field_scale=field_scale,
        kv=kv
    )

    return (
        result["P_plus"],
        result["P_minus"]
    )


def generated_fields_from_polarization(
    P_plus,
    P_minus
):
    """
    Thin-sample Maxwell propagation used throughout Stage 09:

        E_generated ~ -i P^(3)

    Common dimensional constants are omitted.
    """

    E_plus = -1j * P_plus
    E_minus = -1j * P_minus

    return (
        E_plus,
        E_minus
    )


def heterodyne_from_polarization(
    P_plus,
    P_minus,
    omega_probe=OMEGA_PROBE,
    field_scale=1.0
):
    """
    Construct the complex RF heterodyne envelope

        Z =
          E_plus E_c*
          +
          E_minus* E_c

    with

        E_c = probe carrier.
    """

    E_plus, E_minus = (
        generated_fields_from_polarization(
            P_plus,
            P_minus
        )
    )

    E_c = (
        field_scale
        * omega_probe
    )

    Z = (
        E_plus * np.conj(E_c)
        +
        np.conj(E_minus) * E_c
    )

    return Z


def mts_response(
    x,
    beta=BETA,
    omega_pump=OMEGA_PUMP,
    omega_probe=OMEGA_PROBE,
    field_scale=1.0,
    kv=KV
):
    """
    Lightweight public MTS response interface.
    """

    P_plus, P_minus = rho3_sidebands(
        x=x,
        beta=beta,
        omega_pump=omega_pump,
        omega_probe=omega_probe,
        field_scale=field_scale,
        kv=kv
    )

    Z = heterodyne_from_polarization(
        P_plus,
        P_minus,
        omega_probe=omega_probe,
        field_scale=field_scale
    )

    return {
        "P_plus": P_plus,
        "P_minus": P_minus,
        "Z": Z
    }


MODEL_PARAMETERS = {
    "FM": FM,
    "BETA": BETA,
    "LEVEL_OFFSET": LEVEL_OFFSET,
    "GAMMA_AB": GAMMA_AB,
    "GAMMA_BC": GAMMA_BC,
    "GAMMA_AC": GAMMA_AC,
    "OMEGA_PUMP": OMEGA_PUMP,
    "OMEGA_PROBE": OMEGA_PROBE
}

if __name__ == "__main__":

    main()