"""
13_rb87_hyperfine_zeeman_mts.py

Stage F2:
87Rb D2 hyperfine + Zeeman resolved explicit rho^(3) MTS.

Included atomic manifolds
-------------------------
Ground:
    5S1/2 Fg = 1, 2

Excited:
    5P3/2 Fe = 1, 2, 3

Laser drive:
    Fg = 2 -> Fe = 1, 2, 3

Fg = 1 is retained explicitly as a dark hyperfine reservoir
because Fe = 1,2 can spontaneously decay into it.

Main chain
----------
hyperfine offsets
    ->
Wigner 3j / 6j dipole matrix
    ->
Zeeman-resolved Liouvillian
    ->
rho^(0) -> rho^(1) -> rho^(2) -> rho^(3)
    ->
P_+^(3), P_-^(3)
    ->
heterodyne Z
    ->
lock-in MTS signal

Doppler averaging is NOT yet performed here.
The kv interface is retained so Stage F3 can directly wrap
this validated engine with the Stage-11 thermal integration.
"""

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass

from scipy.special import jv
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from scipy.sparse import (
    csc_matrix,
    eye as sparse_eye,
    kron as sparse_kron
)

from scipy.sparse.linalg import splu

from sympy import S
from sympy.physics.wigner import (
    wigner_3j,
    wigner_6j
)


# ============================================================
# 0. Calculation mode
# ============================================================

FAST_MODE = False

RUN_SCALING_CHECK = True


# ============================================================
# 1. Common MTS parameters
#
# Keep the same normalization as the previous project stages.
# ============================================================

GAMMA_MHZ = 6.065

FM_MHZ = 12.5

FM = (
    FM_MHZ
    /
    GAMMA_MHZ
)

BETA = 0.28


# Weak-field perturbative Rabi amplitudes

OMEGA_PUMP = 0.15

OMEGA_PROBE = 0.03


# pi polarization

POL_Q = 0


# ============================================================
# 2. 87Rb angular momenta
# ============================================================

I_NUC = S(3) / 2

JG = S(1) / 2

JE = S(3) / 2


FG_LIST = (
    1,
    2
)

FE_LIST = (
    1,
    2,
    3
)


# ============================================================
# 3. Hyperfine constants
# ============================================================

# 5P3/2

A_EXCITED_MHZ = 84.7185

B_EXCITED_MHZ = 12.4965


# 5S1/2 ground-state F=1 <-> F=2 interval

GROUND_HFS_MHZ = (
    6834.68261090429
)


# ============================================================
# 4. Excited-state hyperfine energies
# ============================================================

def excited_hfs_energy_mhz(
    F
):
    """
    5P3/2 hyperfine energy relative to the
    fine-structure center of gravity.

    Magnetic octupole term neglected.
    """

    I = 1.5

    J = 1.5


    K = (

        F * (F + 1)

        -

        I * (I + 1)

        -

        J * (J + 1)
    )


    term_A = (

        0.5

        *

        A_EXCITED_MHZ

        *

        K
    )


    term_B = (

        B_EXCITED_MHZ

        *

        (

            1.5
            *
            K
            *
            (K + 1)

            -

            2
            *
            I
            *
            (I + 1)
            *
            J
            *
            (J + 1)

        )

        /

        (

            4
            *
            I
            *
            (2 * I - 1)
            *
            J
            *
            (2 * J - 1)

        )
    )


    return (
        term_A
        +
        term_B
    )


E_HFS_MHZ = {

    F:
    excited_hfs_energy_mhz(F)

    for F in FE_LIST
}


# Reference:
#
# laser detuning = 0
#
# means Fg=2 -> Fe=3 resonance.

HF_OFFSET_MHZ = {

    F:

    E_HFS_MHZ[F]
    -
    E_HFS_MHZ[3]

    for F in FE_LIST
}


HF_OFFSET = {

    F:

    HF_OFFSET_MHZ[F]
    /
    GAMMA_MHZ

    for F in FE_LIST
}


# ============================================================
# 5. Magnetic field
# ============================================================

B_GAUSS = 0.0


MU_B_OVER_H_MHZ_PER_G = (
    1.39962449361
)


GJ_GROUND = (
    2.002331070
)

GJ_EXCITED = (
    1.33410
)

GI = (
    -0.0009951414
)


def gF(
    F,
    J,
    gJ
):

    I = float(I_NUC)

    J = float(J)


    return (

        gJ

        *

        (
            F * (F + 1)
            -
            I * (I + 1)
            +
            J * (J + 1)
        )

        /

        (
            2
            *
            F
            *
            (F + 1)
        )

        +

        GI

        *

        (
            F * (F + 1)
            +
            I * (I + 1)
            -
            J * (J + 1)
        )

        /

        (
            2
            *
            F
            *
            (F + 1)
        )
    )


GF_GROUND = {

    F:
    gF(
        F,
        JG,
        GJ_GROUND
    )

    for F in FG_LIST
}


GF_EXCITED = {

    F:
    gF(
        F,
        JE,
        GJ_EXCITED
    )

    for F in FE_LIST
}


ZEEMAN_UNIT = (

    MU_B_OVER_H_MHZ_PER_G

    *
    B_GAUSS

    /
    GAMMA_MHZ
)


# ============================================================
# 6. Ground-state transit relaxation
#
# All 8 magnetic ground states are weakly mixed.
#
# This models fresh thermal atoms entering/leaving the beam
# and makes rho^(0) unique.
# ============================================================

GAMMA_TRANSIT = 0.02


# ============================================================
# 7. Build state list
# ============================================================

STATES = []


for F in FG_LIST:

    for m in range(
        -F,
        F + 1
    ):

        STATES.append(
            (
                "g",
                F,
                m
            )
        )


for F in FE_LIST:

    for m in range(
        -F,
        F + 1
    ):

        STATES.append(
            (
                "e",
                F,
                m
            )
        )


INDEX = {

    state: i

    for i, state
    in enumerate(
        STATES
    )
}


N = len(
    STATES
)

DIM = (
    N
    *
    N
)


GROUND_STATES = [

    state

    for state
    in STATES

    if state[0] == "g"
]


EXCITED_STATES = [

    state

    for state
    in STATES

    if state[0] == "e"
]


I_N = sparse_eye(
    N,
    format="csc",
    dtype=complex
)


I_SUPER = sparse_eye(
    DIM,
    format="csc",
    dtype=complex
)


# ============================================================
# 8. Dipole matrix elements
#
# Direct Wigner-Eckart construction.
# ============================================================

def hyperfine_reduced(
    Fg,
    Fe
):
    """
    Hyperfine reduced dipole amplitude apart from
    the common electronic reduced matrix element.
    """

    phase = (

        (-1)

        **
        int(
            Fe
            +
            3
        )
    )


    six_j = float(

        wigner_6j(

            JG,
            JE,
            S(1),

            S(Fe),
            S(Fg),
            I_NUC
        )
    )


    return (

        phase

        *

        np.sqrt(

            (
                2 * Fe
                +
                1
            )

            *

            (
                2 * float(JG)
                +
                1
            )
        )

        *

        six_j
    )


def raw_emission_element(
    Fg,
    mg,
    Fe,
    me,
    q
):
    """
    <Fg mg | d_q | Fe me>

    Common fine-structure reduced dipole moment omitted.
    """

    if (
        mg
        !=
        me + q
    ):

        return 0.0


    if (
        abs(mg) > Fg

        or

        abs(me) > Fe
    ):

        return 0.0


    reduced = (
        hyperfine_reduced(
            Fg,
            Fe
        )
    )


    phase = (

        (-1)

        **
        int(
            Fe
            -
            1
            +
            mg
        )
    )


    three_j = float(

        wigner_3j(

            S(Fe),
            S(1),
            S(Fg),

            S(me),
            S(q),
            S(-mg)
        )
    )


    return (

        reduced

        *
        phase

        *
        np.sqrt(
            2 * Fg
            +
            1
        )

        *
        three_j
    )


# ------------------------------------------------------------
# Normalize so that every excited Zeeman state has
#
#     sum decay probability = 1
#
# in units Gamma = 1.
# ------------------------------------------------------------

RAW_DECAY_TOTAL = 0.0


_REFERENCE_FE = 3

_REFERENCE_ME = 0


for Fg in FG_LIST:

    for mg in range(
        -Fg,
        Fg + 1
    ):

        for q in (
            -1,
            0,
            +1
        ):

            a = raw_emission_element(

                Fg,
                mg,

                _REFERENCE_FE,
                _REFERENCE_ME,

                q
            )


            RAW_DECAY_TOTAL += (
                abs(a)**2
            )


DIPOLE_NORM = (

    1.0

    /

    np.sqrt(
        RAW_DECAY_TOTAL
    )
)


def emission_element(
    Fg,
    mg,
    Fe,
    me,
    q
):

    return (

        DIPOLE_NORM

        *

        raw_emission_element(

            Fg,
            mg,
            Fe,
            me,
            q
        )
    )


# ============================================================
# 9. Sparse matrix utility
# ============================================================

def sparse_operator(
    entries
):

    matrix = np.zeros(
        (
            N,
            N
        ),
        dtype=complex
    )


    for (
        i,
        j,
        value

    ) in entries:

        matrix[
            i,
            j
        ] += value


    return csc_matrix(
        matrix
    )


# ============================================================
# 10. Optical drive operator
#
# Only:
#
#     Fg = 2 -> Fe = 1,2,3
#
# is driven.
# ============================================================

DRIVE_ENTRIES = []


for Fe in FE_LIST:

    for me in range(
        -Fe,
        Fe + 1
    ):


        mg = (
            me
            -
            POL_Q
        )


        if (
            abs(mg)
            >
            2
        ):

            continue


        amp_emission = (
            emission_element(

                2,
                mg,

                Fe,
                me,

                POL_Q
            )
        )


        amp_absorption = (
            np.conj(
                amp_emission
            )
        )


        if (
            abs(
                amp_absorption
            )
            <
            1e-15
        ):

            continue


        DRIVE_ENTRIES.append(

            (

                INDEX[
                    (
                        "e",
                        Fe,
                        me
                    )
                ],

                INDEX[
                    (
                        "g",
                        2,
                        mg
                    )
                ],

                amp_absorption
            )
        )


D_DRIVE = sparse_operator(
    DRIVE_ENTRIES
)


D_DRIVE_DAG = (
    D_DRIVE.getH()
)


# ============================================================
# 11. Spontaneous-emission collapse operators
#
# Separate:
#
#     Fe
#     Fg
#     q
#
# channels.
#
# This avoids artificial coherence between photons belonging
# to spectrally different hyperfine decay channels.
# ============================================================

C_SPONT = []


for Fe in FE_LIST:

    for Fg in FG_LIST:

        for q in (
            -1,
            0,
            +1
        ):


            entries = []


            for me in range(
                -Fe,
                Fe + 1
            ):


                mg = (
                    me
                    +
                    q
                )


                if (
                    abs(mg)
                    >
                    Fg
                ):

                    continue


                amp = emission_element(

                    Fg,
                    mg,

                    Fe,
                    me,

                    q
                )


                if (
                    abs(amp)
                    <
                    1e-15
                ):

                    continue


                entries.append(

                    (

                        INDEX[
                            (
                                "g",
                                Fg,
                                mg
                            )
                        ],

                        INDEX[
                            (
                                "e",
                                Fe,
                                me
                            )
                        ],

                        amp
                    )
                )


            if entries:

                C_SPONT.append(
                    sparse_operator(
                        entries
                    )
                )


# ============================================================
# 12. Excited projector and decay check
# ============================================================

P_E = np.zeros(
    (
        N,
        N
    ),
    dtype=complex
)


for state in EXCITED_STATES:

    i = INDEX[
        state
    ]

    P_E[
        i,
        i
    ] = 1.0


DECAY_CHECK = np.zeros(
    (
        N,
        N
    ),
    dtype=complex
)


for C in C_SPONT:

    DECAY_CHECK += (

        C.getH()

        @

        C

    ).toarray()


DECAY_ERROR = (
    np.linalg.norm(

        DECAY_CHECK

        -

        P_E
    )
)


# ============================================================
# 13. Transit mixing
# ============================================================

C_TRANSIT = []


if (
    GAMMA_TRANSIT
    >
    0
):

    branch_rate = (

        GAMMA_TRANSIT

        /

        (
            len(
                GROUND_STATES
            )
            -
            1
        )
    )


    for state_from in GROUND_STATES:

        for state_to in GROUND_STATES:


            if (
                state_to
                ==
                state_from
            ):

                continue


            C = sparse_operator(

                [

                    (

                        INDEX[
                            state_to
                        ],

                        INDEX[
                            state_from
                        ],

                        np.sqrt(
                            branch_rate
                        )
                    )
                ]
            )


            C_TRANSIT.append(
                C
            )


C_OPS = (

    C_SPONT

    +

    C_TRANSIT
)


# ============================================================
# 14. Superoperators
# ============================================================

def commutator_superoperator(
    H
):

    return (

        -1j

        *

        (

            sparse_kron(
                I_N,
                H,
                format="csc"
            )

            -

            sparse_kron(
                H.T,
                I_N,
                format="csc"
            )
        )
    )


def lindblad_superoperator(
    C
):

    CdC = (

        C.getH()

        @

        C
    )


    return (

        sparse_kron(
            C.conjugate(),
            C,
            format="csc"
        )

        -

        0.5

        *

        sparse_kron(
            I_N,
            CdC,
            format="csc"
        )

        -

        0.5

        *

        sparse_kron(
            CdC.T,
            I_N,
            format="csc"
        )
    )


# ------------------------------------------------------------
# Dissipator is independent of detuning.
# ------------------------------------------------------------

L_DISS = csc_matrix(
    (
        DIM,
        DIM
    ),
    dtype=complex
)


for C in C_OPS:

    L_DISS += (
        lindblad_superoperator(
            C
        )
    )


# ============================================================
# 15. Unperturbed rho0
#
# At room temperature the ground hyperfine splitting is tiny
# compared with kBT, so use equal occupation per magnetic
# ground-state sublevel.
# ============================================================

RHO0 = np.zeros(
    (
        N,
        N
    ),
    dtype=complex
)


for state in GROUND_STATES:

    i = INDEX[
        state
    ]


    RHO0[
        i,
        i
    ] = (

        1.0

        /

        len(
            GROUND_STATES
        )
    )


RHO0_VEC = (

    RHO0.reshape(
        DIM,
        order="F"
    )
)


# ============================================================
# 16. Trace row
# ============================================================

TRACE_INDICES = [

    i
    +
    i * N

    for i in range(N)
]


# ============================================================
# 17. Field-free Hamiltonian / Liouvillian
# ============================================================

def build_L0(
    delta
):
    """
    delta:
        laser detuning from

            Fg=2 -> Fe=3

        normalized by Gamma.

    Excited Fe resonance occurs at

        delta = HF_OFFSET[Fe].
    """

    energies = np.zeros(
        N,
        dtype=float
    )


    # --------------------------------------------------------
    # ground manifolds
    # --------------------------------------------------------

    for Fg in FG_LIST:

        if (
            Fg == 2
        ):

            hyperfine_energy = (
                0.0
            )

        else:

            hyperfine_energy = (

                -GROUND_HFS_MHZ

                /

                GAMMA_MHZ
            )


        for mg in range(
            -Fg,
            Fg + 1
        ):

            energies[
                INDEX[
                    (
                        "g",
                        Fg,
                        mg
                    )
                ]
            ] = (

                hyperfine_energy

                +

                GF_GROUND[Fg]

                *
                ZEEMAN_UNIT

                *
                mg
            )


    # --------------------------------------------------------
    # excited manifolds
    # --------------------------------------------------------

    for Fe in FE_LIST:

        for me in range(
            -Fe,
            Fe + 1
        ):


            energies[
                INDEX[
                    (
                        "e",
                        Fe,
                        me
                    )
                ]
            ] = (

                -delta

                +

                HF_OFFSET[Fe]

                +

                GF_EXCITED[Fe]

                *
                ZEEMAN_UNIT

                *
                me
            )


    H0 = csc_matrix(

        np.diag(
            energies
        )
    )


    return (

        commutator_superoperator(
            H0
        )

        +

        L_DISS
    )


# ============================================================
# 18. Optical field modes
# ============================================================

@dataclass(frozen=True)
class FieldMode:

    name: str

    n: int

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
# 19. Interaction branches
# ============================================================

def make_branches(
    beta=BETA,
    omega_pump=OMEGA_PUMP,
    omega_probe=OMEGA_PROBE,
    field_scale=1.0
):

    branches = []


    for mode in make_field_modes(

        beta=beta,

        omega_pump=omega_pump,

        omega_probe=omega_probe,

        field_scale=field_scale
    ):


        H_plus = (

            -0.5

            *
            mode.amp

            *
            D_DRIVE
        )


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

            (

                mode.n,

                mode.k,

                commutator_superoperator(
                    H_plus
                )
            )
        )


        branches.append(

            (

                -mode.n,

                -mode.k,

                commutator_superoperator(
                    H_minus
                )
            )
        )


    return branches


# ============================================================
# 20. Explicit perturbative recursion
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
    key = (q, k)

    Moving-atom slow frequency:

        Omega_slow
        =
        q * FM
        -
        k * kv
    """

    L0 = build_L0(
        delta
    )


    branches = make_branches(

        beta=beta,

        omega_pump=omega_pump,

        omega_probe=omega_probe,

        field_scale=field_scale
    )


    previous = {

        (
            0,
            0
        ):

        RHO0_VEC.copy()
    }


    # --------------------------------------------------------
    # LU factorization cache
    #
    # At kv=0 many different spatial pathways share the same
    # q*FM frequency, so one factorization is reused.
    # --------------------------------------------------------

    factor_cache = {}


    def solve_component(
        slow_frequency,
        source
    ):

        key = round(
            float(
                slow_frequency
            ),
            12
        )


        if (
            key
            not in
            factor_cache
        ):


            A = (

                (
                    -1j
                    *
                    slow_frequency
                )

                *
                I_SUPER

                -

                L0
            )


            if (
                abs(
                    slow_frequency
                )
                <
                1e-13
            ):

                A = (
                    A.tolil()
                )


                A[
                    0,
                    :
                ] = 0.0


                for idx_trace in TRACE_INDICES:

                    A[
                        0,
                        idx_trace
                    ] = 1.0


                A = (
                    A.tocsc()
                )


            factor_cache[
                key
            ] = splu(
                A
            )


        rhs = np.asarray(
            source,
            dtype=complex
        ).copy()


        if (
            abs(
                slow_frequency
            )
            <
            1e-13
        ):

            rhs[0] = 0.0


        return (

            factor_cache[
                key
            ]
            .solve(
                rhs
            )
        )


    # --------------------------------------------------------
    # rho1, rho2, rho3
    # --------------------------------------------------------

    all_orders = {

        0:
        previous
    }


    for order in (
        1,
        2,
        3
    ):


        sources = {}


        for (
            q0,
            k0

        ), rho_previous in previous.items():


            for (
                dn,
                dk,
                interaction

            ) in branches:


                source = (

                    interaction

                    @

                    rho_previous
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
                    dn,

                    k0
                    +
                    dk
                )


                if (
                    key
                    not in
                    sources
                ):

                    sources[
                        key
                    ] = np.zeros(
                        DIM,
                        dtype=complex
                    )


                sources[
                    key
                ] += source


        current = {}


        for (
            q,
            k

        ), source in sources.items():


            slow_frequency = (

                q
                *
                FM

                -

                k
                *
                kv
            )


            current[
                (
                    q,
                    k
                )
            ] = solve_component(

                slow_frequency,

                source
            )


        all_orders[
            order
        ] = current


        previous = current


    return all_orders


# ============================================================
# 21. Third-order polarization
# ============================================================

def positive_polarization(
    rho_vec
):
    """
    Same phase convention as Stage 12.
    """

    rho = np.asarray(
        rho_vec
    ).reshape(
        (
            N,
            N
        ),
        order="F"
    )


    return np.trace(

        D_DRIVE.toarray()

        @

        rho
    )


# ============================================================
# 22. Complete MTS response
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


    rho3 = (
        orders[3]
    )


    zero = np.zeros(
        DIM,
        dtype=complex
    )


    rho_plus = rho3.get(
        (
            +1,
            -1
        ),
        zero
    )


    rho_minus = rho3.get(
        (
            -1,
            -1
        ),
        zero
    )


    P_plus = positive_polarization(
        rho_plus
    )


    P_minus = positive_polarization(
        rho_minus
    )


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
        Z
    }


# ============================================================
# 23. Analysis utilities
# ============================================================

def complex_slope_near(
    x,
    Z,
    center,
    window=0.9
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


    xx = (

        x[mask]

        -
        center
    )


    ZZ = Z[
        mask
    ]


    if (
        len(xx)
        <
        4
    ):

        return (
            np.nan
            +
            1j
            *
            np.nan
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

        +

        1j

        *

        ci[-2]
    )


def refined_zero_near(
    x,
    y,
    center,
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


    xx = x[
        mask
    ]


    yy = y[
        mask
    ]


    if (
        len(xx)
        <
        4
    ):

        return np.nan


    order = np.argsort(
        xx
    )


    xx = xx[
        order
    ]

    yy = yy[
        order
    ]


    interp = PchipInterpolator(
        xx,
        yy
    )


    dense = np.linspace(

        center
        -
        window,

        center
        +
        window,

        4001
    )


    values = interp(
        dense
    )


    roots = []


    for i in range(
        len(dense)
        -
        1
    ):


        if (
            values[i]
            ==
            0
        ):

            roots.append(
                dense[i]
            )


        elif (

            values[i]

            *

            values[
                i + 1
            ]

            <
            0
        ):


            roots.append(

                brentq(

                    interp,

                    dense[i],

                    dense[
                        i + 1
                    ]
                )
            )


    if not roots:

        return np.nan


    return min(

        roots,

        key=lambda r:

        abs(
            r
            -
            center
        )
    )


# ============================================================
# 24. Main frequency grid
# ============================================================

if FAST_MODE:

    coarse_mhz = np.linspace(
        -470,
        +30,
        41
    )


    local_points = [

        HF_OFFSET_MHZ[F]

        +

        np.linspace(
            -25,
            +25,
            25
        )

        for F in FE_LIST
    ]

else:

    coarse_mhz = np.linspace(
        -480,
        +40,
        81
    )


    local_points = [

        HF_OFFSET_MHZ[F]

        +

        np.linspace(
            -35,
            +35,
            51
        )

        for F in FE_LIST
    ]


DETUNING_MHZ = np.unique(

    np.round(

        np.concatenate(

            [
                coarse_mhz
            ]

            +

            local_points
        ),

        10
    )
)


DELTA_SCAN = (

    DETUNING_MHZ

    /

    GAMMA_MHZ
)


# ============================================================
# 25. Hyperfine line strengths
# ============================================================

def hyperfine_line_strengths():

    totals = {}


    for Fe in FE_LIST:


        total = 0.0


        for me in range(
            -Fe,
            Fe + 1
        ):

            for mg in range(
                -2,
                2 + 1
            ):

                for q in (
                    -1,
                    0,
                    +1
                ):


                    total += (

                        abs(

                            emission_element(

                                2,
                                mg,

                                Fe,
                                me,

                                q
                            )
                        )

                        ** 2
                    )


        totals[
            Fe
        ] = total


    normalization = sum(
        totals.values()
    )


    return {

        Fe:

        totals[Fe]

        /

        normalization

        for Fe in FE_LIST
    }


LINE_STRENGTH = (
    hyperfine_line_strengths()
)


# ============================================================
# 26. Spontaneous branching ratios
# ============================================================

def branching_ratio(
    Fe,
    Fg
):
    """
    For one representative m_e=0.
    Rotational symmetry makes the total independent of m_e.
    """

    me = 0


    total = 0.0


    for mg in range(
        -Fg,
        Fg + 1
    ):

        for q in (
            -1,
            0,
            +1
        ):


            total += (

                abs(

                    emission_element(

                        Fg,
                        mg,

                        Fe,
                        me,

                        q
                    )
                )

                ** 2
            )


    return total


# ============================================================
# 27. Main
# ============================================================

def main():

    print(
        "=" * 74
    )

    print(
        "Stage F2: 87Rb D2 hyperfine + Zeeman rho^(3) MTS"
    )

    print(
        "=" * 74
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
        "Excited hyperfine offsets relative to F'=3:"
    )


    for Fe in FE_LIST:

        print(

            f"F'={Fe}: "

            f"{HF_OFFSET_MHZ[Fe]:+.6f} MHz"

            f"   = "

            f"{HF_OFFSET[Fe]:+.6f} Gamma"
        )


    print()
    print(
        "Computed Fg=2 relative hyperfine strengths:"
    )


    for Fe in FE_LIST:

        print(

            f"F=2 -> F'={Fe}: "

            f"{LINE_STRENGTH[Fe]:.10f}"
        )


    print()
    print(
        "Expected pattern:"
    )

    print(
        "F'=1 : 0.05"
    )

    print(
        "F'=2 : 0.25"
    )

    print(
        "F'=3 : 0.70"
    )


    print()
    print(
        "Spontaneous branching:"
    )


    for Fe in FE_LIST:

        b1 = branching_ratio(
            Fe,
            1
        )

        b2 = branching_ratio(
            Fe,
            2
        )


        print(

            f"F'={Fe}: "

            f"to F=1 = {b1:.8f}, "

            f"to F=2 = {b2:.8f}, "

            f"sum = {b1+b2:.8f}"
        )


    print()
    print(
        "spontaneous-decay consistency norm =",
        DECAY_ERROR
    )


    L0_test = build_L0(
        0.0
    )


    rho0_residual = (

        np.linalg.norm(

            L0_test

            @

            RHO0_VEC
        )
    )


    print(
        "rho0 steady-state residual =",
        rho0_residual
    )


    # ========================================================
    # Main scan
    # ========================================================

    print()
    print(
        "Calculating hyperfine-resolved rho^(3) spectrum..."
    )


    P_PLUS = np.zeros(
        len(
            DELTA_SCAN
        ),
        dtype=complex
    )


    P_MINUS = np.zeros_like(
        P_PLUS
    )


    Z = np.zeros_like(
        P_PLUS
    )


    for i, delta in enumerate(
        DELTA_SCAN
    ):


        print(

            f"{i+1:3d}/"
            f"{len(DELTA_SCAN)}"

            f"   detuning = "

            f"{DETUNING_MHZ[i]:+9.3f} MHz"
        )


        result = response_at_detuning(
            delta
        )


        P_PLUS[i] = (
            result[
                "P_plus"
            ]
        )


        P_MINUS[i] = (
            result[
                "P_minus"
            ]
        )


        Z[i] = (
            result[
                "Z"
            ]
        )


    # ========================================================
    # Individual line metrics
    # ========================================================

    LINE_RESULTS = {}


    print()
    print(
        "=" * 74
    )

    print(
        "Hyperfine component metrics"
    )

    print(
        "=" * 74
    )


    for Fe in FE_LIST:


        center = (
            HF_OFFSET[
                Fe
            ]
        )


        slope = complex_slope_near(

            DELTA_SCAN,

            Z,

            center
        )


        phi = np.angle(
            slope
        )


        S_line = np.real(

            Z

            *

            np.exp(
                -1j
                *
                phi
            )
        )


        zero = refined_zero_near(

            DELTA_SCAN,

            S_line,

            center
        )


        if np.isfinite(
            zero
        ):

            zero_shift_mhz = (

                (
                    zero
                    -
                    center
                )

                *
                GAMMA_MHZ
            )

        else:

            zero_shift_mhz = np.nan


        LINE_RESULTS[
            Fe
        ] = {

            "slope":
            slope,

            "phase":
            phi,

            "signal":
            S_line,

            "zero":
            zero,

            "zero_shift_mhz":
            zero_shift_mhz
        }


        print()
        print(
            f"F=2 -> F'={Fe}"
        )

        print(
            "  resonance =",
            HF_OFFSET_MHZ[Fe],
            "MHz"
        )

        print(
            "  |complex slope| =",
            abs(
                slope
            )
        )

        print(
            "  optimum phase =",
            np.degrees(
                phi
            ),
            "deg"
        )

        print(
            "  zero crossing =",
            (
                zero
                *
                GAMMA_MHZ
                if np.isfinite(zero)
                else np.nan
            ),
            "MHz"
        )

        print(
            "  lock shift from nominal resonance =",
            zero_shift_mhz,
            "MHz"
        )


    # ========================================================
    # F'=3 phase for global discriminator
    # ========================================================

    PHI_LOCK = (
        LINE_RESULTS[
            3
        ][
            "phase"
        ]
    )


    S_GLOBAL = np.real(

        Z

        *

        np.exp(
            -1j
            *
            PHI_LOCK
        )
    )


    # ========================================================
    # beta = 0
    # ========================================================

    TEST_DELTA = (

        HF_OFFSET[3]

        +

        FM
    )


    reference = response_at_detuning(
        TEST_DELTA,
        beta=BETA
    )


    null = response_at_detuning(
        TEST_DELTA,
        beta=0.0
    )


    NULL_RATIO = (

        abs(
            null[
                "Z"
            ]
        )

        /

        max(
            abs(
                reference[
                    "Z"
                ]
            ),
            1e-300
        )
    )


    print()
    print(
        "Structural validation:"
    )

    print(
        "beta=0 null-test ratio =",
        NULL_RATIO
    )


    # ========================================================
    # Third-order scaling
    # ========================================================

    scales = np.array(
        [
            0.50,
            0.75,
            1.00
        ]
    )


    P_SCALE = []

    Z_SCALE = []


    if RUN_SCALING_CHECK:


        for scale in scales:


            result = response_at_detuning(

                TEST_DELTA,

                field_scale=scale
            )


            P_SCALE.append(

                max(

                    abs(
                        result[
                            "P_plus"
                        ]
                    ),

                    abs(
                        result[
                            "P_minus"
                        ]
                    )
                )
            )


            Z_SCALE.append(

                abs(
                    result[
                        "Z"
                    ]
                )
            )


        P_SCALE = np.asarray(
            P_SCALE
        )


        Z_SCALE = np.asarray(
            Z_SCALE
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
            scale,
            p,
            z

        ) in zip(

            scales,

            P_SCALE,

            Z_SCALE
        ):


            print(

                f"{scale:8.3f}"

                f"{p/scale**3:20.10e}"

                f"{z/scale**4:20.10e}"
            )


    print()
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
    # Figure 1:
    # real hyperfine positions + line strengths
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5
        )
    )


    for Fe in FE_LIST:


        x = (
            HF_OFFSET_MHZ[
                Fe
            ]
        )


        y = (
            LINE_STRENGTH[
                Fe
            ]
        )


        plt.vlines(

            x,

            0,

            y
        )


        plt.plot(

            x,

            y,

            "o"
        )


        plt.text(

            x,

            y
            +
            0.025,

            f"F'={Fe}",

            ha="center"
        )


    plt.xlabel(
        "Detuning from F=2 -> F'=3 (MHz)"
    )

    plt.ylabel(
        "Relative hyperfine line strength"
    )

    plt.title(
        r"$^{87}$Rb D2 "
        r"$F=2\rightarrow F'=1,2,3$"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    # ========================================================
    # Figure 2:
    # generated sidebands across hyperfine spectrum
    # ========================================================

    SCALE_P = max(

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
        figsize=(
            9,
            5
        )
    )


    plt.plot(

        DETUNING_MHZ,

        np.abs(
            P_PLUS
        )
        /
        SCALE_P,

        label=
        r"$\omega+\Omega_m$"
    )


    plt.plot(

        DETUNING_MHZ,

        np.abs(
            P_MINUS
        )
        /
        SCALE_P,

        label=
        r"$\omega-\Omega_m$"
    )


    for Fe in FE_LIST:

        plt.axvline(

            HF_OFFSET_MHZ[
                Fe
            ],

            linestyle="--",

            linewidth=0.8
        )


    plt.xlabel(
        "Laser detuning from F'=3 (MHz)"
    )

    plt.ylabel(
        "Generated third-order polarization"
    )

    plt.title(
        r"$^{87}$Rb hyperfine-resolved "
        r"$P_\pm^{(3)}$"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    # ========================================================
    # Figure 3:
    # complex heterodyne response
    # ========================================================

    SCALE_Z = max(

        np.max(
            np.abs(
                Z
            )
        ),

        1e-300
    )


    plt.figure(
        figsize=(
            9,
            5
        )
    )


    plt.plot(

        DETUNING_MHZ,

        np.real(
            Z
        )
        /
        SCALE_Z,

        label="Re(Z)"
    )


    plt.plot(

        DETUNING_MHZ,

        np.imag(
            Z
        )
        /
        SCALE_Z,

        label="Im(Z)"
    )


    for Fe in FE_LIST:

        plt.axvline(

            HF_OFFSET_MHZ[
                Fe
            ],

            linestyle="--",

            linewidth=0.8
        )


    plt.axhline(
        0,
        linewidth=0.8
    )


    plt.xlabel(
        "Laser detuning from F'=3 (MHz)"
    )

    plt.ylabel(
        "Complex RF response"
    )

    plt.title(
        r"$^{87}$Rb hyperfine + Zeeman MTS response"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    # ========================================================
    # Figure 4:
    # one fixed lock-in phase optimized at F'=3
    # ========================================================

    SCALE_S = max(

        np.max(
            np.abs(
                S_GLOBAL
            )
        ),

        1e-300
    )


    plt.figure(
        figsize=(
            9,
            5
        )
    )


    plt.plot(

        DETUNING_MHZ,

        S_GLOBAL
        /
        SCALE_S,

        label=(
            "fixed phase = "
            f"{np.degrees(PHI_LOCK):.1f} deg"
        )
    )


    for Fe in FE_LIST:

        plt.axvline(

            HF_OFFSET_MHZ[
                Fe
            ],

            linestyle="--",

            linewidth=0.8,

            label=(
                f"F'={Fe}"
            )
        )


    plt.axhline(
        0,
        linewidth=0.8
    )


    plt.xlabel(
        "Laser detuning from F'=3 (MHz)"
    )

    plt.ylabel(
        "Normalized MTS error signal"
    )

    plt.title(
        r"$^{87}$Rb D2 hyperfine MTS spectrum"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    # ========================================================
    # Figure 5:
    # local line shapes around each hyperfine component
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5
        )
    )


    for Fe in FE_LIST:


        center = (
            HF_OFFSET[
                Fe
            ]
        )


        mask = (

            np.abs(
                DELTA_SCAN
                -
                center
            )

            <=

            5.0
        )


        local_x = (

            DELTA_SCAN[
                mask
            ]

            -

            center
        )


        local_y = (

            LINE_RESULTS[
                Fe
            ][
                "signal"
            ][
                mask
            ]
        )


        local_norm = max(

            np.max(
                np.abs(
                    local_y
                )
            ),

            1e-300
        )


        order = np.argsort(
            local_x
        )


        plt.plot(

            local_x[
                order
            ],

            local_y[
                order
            ]

            /

            local_norm,

            label=(
                f"F'={Fe}, "
                f"phi="
                f"{np.degrees(LINE_RESULTS[Fe]['phase']):.1f} deg"
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
        r"Local detuning "
        r"$(\Delta-\Delta_{F'})/\Gamma$"
    )

    plt.ylabel(
        "Self-normalized MTS signal"
    )

    plt.title(
        "Local MTS structure of each excited hyperfine line"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    # ========================================================
    # Figure 6:
    # strict third-order scaling
    # ========================================================

    if RUN_SCALING_CHECK:


        plt.figure(
            figsize=(
                7,
                5
            )
        )


        plt.loglog(

            scales,

            P_SCALE
            /
            P_SCALE[-1],

            "o-",

            label=
            r"$|P^{(3)}|$"
        )


        plt.loglog(

            scales,

            scales**3,

            "--",

            label=
            r"$s^3$ reference"
        )


        plt.xlabel(
            "Common field-amplitude scale s"
        )

        plt.ylabel(
            "Relative third-order polarization"
        )

        plt.title(
            "Hyperfine-resolved third-order scaling"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()


    plt.show()


if __name__ == "__main__":

    main()