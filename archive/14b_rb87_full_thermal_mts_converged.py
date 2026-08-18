"""
14b_rb87_full_thermal_mts_converged.py

FINAL CONVERGED THERMAL INTEGRATION

87Rb D2 real hyperfine + Zeeman rho^(3) MTS
with a FIXED GLOBAL Maxwell-Boltzmann velocity grid.

This replaces the resonance-aware velocity grid used in
14_rb87_full_thermal_mts.py.

Why this version exists
-----------------------
The previous adaptive/composite velocity grid failed convergence:

    large base -> refined differences
    large +/-4u -> +/-5u differences
    thermal lock point -> NaN

The reason is that the thermal MTS response is a complex coherent
sum over velocity classes:

    <P^(3)> = integral f(q) P^(3)(q) dq

Large complex contributions can cancel strongly. Therefore changing
quadrature nodes with laser detuning is unsafe.

This version uses:

    one fixed q-grid for ALL laser detunings
    q step = 0.50 Gamma       main result
    q step = 0.25 Gamma       refinement test

and a grid anchored at q=0.

The program performs a convergence precheck BEFORE producing the
final spectrum. If the velocity integration fails, it aborts.

No FAST mode exists in this file.
"""


import sys
import importlib.util

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import find_peaks

from scipy.sparse import csc_matrix


# ============================================================
# 0. CALCULATION MODE
# ============================================================

MODE = "FINAL"

if MODE != "FINAL":
    raise RuntimeError(
        "This script is intended only for FINAL quantitative runs."
    )


# ============================================================
# 1. Physical parameters
# ============================================================

TEMPERATURE_K = 300.0

# ------------------------------------------------------------
# Transit relaxation.
#
# Stage 13b showed that this is an IMPORTANT model parameter.
# Do not quote final lock shift without quoting this value.
# ------------------------------------------------------------

GAMMA_TRANSIT = 0.020

LOCK_FE = 3


# ============================================================
# 2. Velocity integration settings
# ============================================================

# ------------------------------------------------------------
# Main fixed velocity grid
# ------------------------------------------------------------

Q_MAIN_STEP = 0.50

# ------------------------------------------------------------
# Independent refinement
# ------------------------------------------------------------

Q_REFINE_STEP = 0.25

# ------------------------------------------------------------
# Main Maxwell boundary
# ------------------------------------------------------------

VELOCITY_SPAN_U = 4.0

# ------------------------------------------------------------
# Boundary convergence comparison
# ------------------------------------------------------------

VELOCITY_SPAN_U_TEST = 5.0


# ============================================================
# 3. Convergence precheck
# ============================================================

RUN_PRECHECK = True

ABORT_IF_PRECHECK_FAILS = True

# Test the complex response at several lock-region points.

PRECHECK_OFFSETS_MHZ = np.array(
    [
        -2.0,
         0.0,
        +2.0
    ]
)

# Acceptance targets

GRID_ERROR_LIMIT = 1.0e-2       # 1 %
BOUNDARY_ERROR_LIMIT = 1.0e-3   # 0.1 %


# ============================================================
# 4. Laser-frequency grid
# ============================================================

# ------------------------------------------------------------
# Dense central region:
#
#     +/- 3 MHz
#     0.25 MHz spacing
# ------------------------------------------------------------

CENTER_SPAN_MHZ = 3.0
CENTER_STEP_MHZ = 0.25

# ------------------------------------------------------------
# Wider MTS region:
#
#     +/- 12 MHz
#     1 MHz spacing
# ------------------------------------------------------------

FULL_SPAN_MHZ = 12.0
WING_STEP_MHZ = 1.0


# ============================================================
# 5. Optional refined lock extraction
#
# Use the finer q=0.25 velocity grid only around the final
# lock point, instead of recomputing the entire spectrum.
# ============================================================

RUN_REFINED_LOCK = True

REFINED_PHASE_OFFSETS_MHZ = np.array(
    [
        -1.0,
        -0.5,
         0.0,
        +0.5,
        +1.0
    ]
)

REFINED_LOCK_STEP_MHZ = 0.25
REFINED_LOCK_HALF_WIDTH_MHZ = 0.75


# ============================================================
# 6. Physical constants
# ============================================================

KB = 1.380649e-23

AMU = 1.66053906660e-27

MASS_RB87 = (
    86.9091805
    *
    AMU
)

LAMBDA_D2 = 780.24e-9


# ============================================================
# 7. Load the validated Stage-13 atomic engine
# ============================================================

THIS_DIR = Path(
    __file__
).resolve().parent

ENGINE_PATH = (
    THIS_DIR
    /
    "13_rb87_hyperfine_zeeman_mts.py"
)

if not ENGINE_PATH.exists():

    raise FileNotFoundError(
        "\nCannot find:\n"
        f"{ENGINE_PATH}\n\n"
        "Place this script and "
        "13_rb87_hyperfine_zeeman_mts.py "
        "in the same folder."
    )


MODULE_NAME = "rb87_hyperfine_engine_14b"

spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    ENGINE_PATH
)

engine = importlib.util.module_from_spec(
    spec
)

sys.modules[
    MODULE_NAME
] = engine

spec.loader.exec_module(
    engine
)


# ============================================================
# 8. Rebuild the selected transit dissipator
# ============================================================

L_DISS_SPONT = csc_matrix(
    (
        engine.DIM,
        engine.DIM
    ),
    dtype=complex
)


for C in engine.C_SPONT:

    L_DISS_SPONT += (
        engine.lindblad_superoperator(
            C
        )
    )


def make_transit_dissipator(
    gamma_transit
):

    L = L_DISS_SPONT.copy()

    if gamma_transit <= 0:
        return L


    branch_rate = (
        gamma_transit
        /
        (
            len(
                engine.GROUND_STATES
            )
            - 1
        )
    )


    for state_from in engine.GROUND_STATES:

        for state_to in engine.GROUND_STATES:

            if state_from == state_to:
                continue


            C = engine.sparse_operator(
                [
                    (
                        engine.INDEX[
                            state_to
                        ],

                        engine.INDEX[
                            state_from
                        ],

                        np.sqrt(
                            branch_rate
                        )
                    )
                ]
            )


            L += (
                engine.lindblad_superoperator(
                    C
                )
            )


    return L


L_DISS_SELECTED = (
    make_transit_dissipator(
        GAMMA_TRANSIT
    )
)


def build_L0_selected(
    delta
):

    energies = np.zeros(
        engine.N,
        dtype=float
    )


    # --------------------------------------------------------
    # Ground manifolds
    # --------------------------------------------------------

    for Fg in engine.FG_LIST:

        if Fg == 2:

            hyperfine_energy = 0.0

        else:

            hyperfine_energy = (
                -engine.GROUND_HFS_MHZ
                /
                engine.GAMMA_MHZ
            )


        for mg in range(
            -Fg,
            Fg + 1
        ):

            energies[
                engine.INDEX[
                    (
                        "g",
                        Fg,
                        mg
                    )
                ]
            ] = (
                hyperfine_energy
                +
                engine.GF_GROUND[
                    Fg
                ]
                *
                engine.ZEEMAN_UNIT
                *
                mg
            )


    # --------------------------------------------------------
    # Excited manifolds
    # --------------------------------------------------------

    for Fe in engine.FE_LIST:

        for me in range(
            -Fe,
            Fe + 1
        ):

            energies[
                engine.INDEX[
                    (
                        "e",
                        Fe,
                        me
                    )
                ]
            ] = (
                -delta
                +
                engine.HF_OFFSET[
                    Fe
                ]
                +
                engine.GF_EXCITED[
                    Fe
                ]
                *
                engine.ZEEMAN_UNIT
                *
                me
            )


    H0 = csc_matrix(
        np.diag(
            energies
        )
    )


    return (
        engine.commutator_superoperator(
            H0
        )
        +
        L_DISS_SELECTED
    )


engine.GAMMA_TRANSIT = GAMMA_TRANSIT

engine.build_L0 = build_L0_selected


# ============================================================
# 9. Zeroth-order validation
# ============================================================

RHO0_RESIDUAL = np.linalg.norm(
    engine.build_L0(
        0.0
    )
    @
    engine.RHO0_VEC
)


# ============================================================
# 10. Thermal speed
# ============================================================

def thermal_speed_u(
    temperature
):

    return np.sqrt(
        2.0
        *
        KB
        *
        temperature
        /
        MASS_RB87
    )


U_THERMAL = thermal_speed_u(
    TEMPERATURE_K
)


DOPPLER_D = (
    U_THERMAL
    /
    LAMBDA_D2
    /
    (
        engine.GAMMA_MHZ
        *
        1e6
    )
)


# ============================================================
# 11. FIXED GLOBAL velocity grid
# ============================================================

def build_fixed_q_grid(
    dq,
    span_u
):
    """
    Build a symmetric q grid anchored exactly at q=0.

    IMPORTANT:
    The same q values are used at every laser detuning.

        q = kv / Gamma
    """

    qmax = (
        span_u
        *
        DOPPLER_D
    )


    N = int(
        np.floor(
            qmax
            /
            dq
        )
    )


    q = (
        np.arange(
            -N,
            N + 1
        )
        *
        dq
    )


    return q


def maxwell_weight_q(
    q
):

    return (
        np.exp(
            -(q / DOPPLER_D)**2
        )
        /
        (
            np.sqrt(
                np.pi
            )
            *
            DOPPLER_D
        )
    )


Q_MAIN = build_fixed_q_grid(
    Q_MAIN_STEP,
    VELOCITY_SPAN_U
)

Q_REFINE = build_fixed_q_grid(
    Q_REFINE_STEP,
    VELOCITY_SPAN_U
)

Q_BOUNDARY = build_fixed_q_grid(
    Q_MAIN_STEP,
    VELOCITY_SPAN_U_TEST
)


W_MAIN = maxwell_weight_q(
    Q_MAIN
)

W_REFINE = maxwell_weight_q(
    Q_REFINE
)

W_BOUNDARY = maxwell_weight_q(
    Q_BOUNDARY
)


MASS_MAIN = np.trapezoid(
    W_MAIN,
    Q_MAIN
)

MASS_REFINE = np.trapezoid(
    W_REFINE,
    Q_REFINE
)

MASS_BOUNDARY = np.trapezoid(
    W_BOUNDARY,
    Q_BOUNDARY
)


# ============================================================
# 12. Atomic response cache
#
# Refined q=0.25 contains all q=0.50 nodes.
# This cache means previously evaluated nodes are reused.
# ============================================================

ATOMIC_CACHE = {}


def cached_atomic_response(
    delta,
    kv,
    beta=engine.BETA,
    field_scale=1.0
):

    key = (
        round(
            float(delta),
            12
        ),

        round(
            float(kv),
            12
        ),

        round(
            float(beta),
            12
        ),

        round(
            float(field_scale),
            10
        )
    )


    if key not in ATOMIC_CACHE:

        ATOMIC_CACHE[
            key
        ] = (
            engine.response_at_detuning(
                delta=delta,
                kv=kv,
                beta=beta,
                field_scale=field_scale
            )
        )


    return ATOMIC_CACHE[
        key
    ]


# ============================================================
# 13. Heterodyne helper
# ============================================================

def heterodyne_from_polarization(
    P_plus,
    P_minus,
    field_scale=1.0
):

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
        engine.OMEGA_PROBE
    )


    return (
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


# ============================================================
# 14. Thermal average on a GIVEN fixed grid
# ============================================================

def thermal_average_on_grid(
    delta,
    q,
    weight,
    beta=engine.BETA,
    field_scale=1.0,
    return_profile=False
):

    Nq = len(
        q
    )


    P_plus_v = np.empty(
        Nq,
        dtype=complex
    )

    P_minus_v = np.empty(
        Nq,
        dtype=complex
    )

    Z_v = np.empty(
        Nq,
        dtype=complex
    )


    for i, kv in enumerate(
        q
    ):

        result = cached_atomic_response(
            delta=delta,
            kv=kv,
            beta=beta,
            field_scale=field_scale
        )


        P_plus_v[
            i
        ] = result[
            "P_plus"
        ]


        P_minus_v[
            i
        ] = result[
            "P_minus"
        ]


        Z_v[
            i
        ] = (
            heterodyne_from_polarization(
                P_plus_v[
                    i
                ],
                P_minus_v[
                    i
                ],
                field_scale=field_scale
            )
        )


    # --------------------------------------------------------
    # Correct macroscopic operation:
    #
    # average COMPLEX polarization before detection.
    # --------------------------------------------------------

    P_plus_avg = np.trapezoid(
        weight
        *
        P_plus_v,
        q
    )


    P_minus_avg = np.trapezoid(
        weight
        *
        P_minus_v,
        q
    )


    Z_avg = (
        heterodyne_from_polarization(
            P_plus_avg,
            P_minus_avg,
            field_scale=field_scale
        )
    )


    # --------------------------------------------------------
    # Independent linearity check:
    #
    # because heterodyne is linear in the generated field,
    #
    #   integral f(q) Z(q) dq
    #
    # must equal Z from averaged polarization.
    # --------------------------------------------------------

    Z_direct = np.trapezoid(
        weight
        *
        Z_v,
        q
    )


    linearity_error = (
        abs(
            Z_direct
            -
            Z_avg
        )
        /
        max(
            abs(
                Z_avg
            ),
            abs(
                Z_direct
            ),
            1e-300
        )
    )


    output = {
        "P_plus":
        P_plus_avg,

        "P_minus":
        P_minus_avg,

        "Z":
        Z_avg,

        "Z_direct":
        Z_direct,

        "linearity_error":
        linearity_error,

        "Nq":
        Nq
    }


    if return_profile:

        output.update(
            {
                "q":
                q,

                "weight":
                weight,

                "P_plus_v":
                P_plus_v,

                "P_minus_v":
                P_minus_v,

                "Z_v":
                Z_v,

                "Z_integrand":
                weight
                *
                Z_v
            }
        )


    return output


# ============================================================
# 15. Convenient wrappers
# ============================================================

def thermal_main(
    delta,
    **kwargs
):

    return thermal_average_on_grid(
        delta,
        Q_MAIN,
        W_MAIN,
        **kwargs
    )


def thermal_refined(
    delta,
    **kwargs
):

    return thermal_average_on_grid(
        delta,
        Q_REFINE,
        W_REFINE,
        **kwargs
    )


def thermal_boundary(
    delta,
    **kwargs
):

    return thermal_average_on_grid(
        delta,
        Q_BOUNDARY,
        W_BOUNDARY,
        **kwargs
    )


# ============================================================
# 16. Header
# ============================================================

print()
print(
    "=" * 80
)

print(
    "Stage F3b FINAL: fixed-grid converged thermal Rb87 MTS"
)

print(
    "=" * 80
)


print()
print(
    "CALCULATION MODE = FINAL"
)


print()
print(
    "Atomic model:"
)

print(
    "Hilbert dimension =",
    engine.N
)

print(
    "Liouville dimension =",
    engine.DIM
)

print(
    "Gamma/(2pi) =",
    engine.GAMMA_MHZ,
    "MHz"
)

print(
    "fm =",
    engine.FM_MHZ,
    "MHz"
)

print(
    "beta =",
    engine.BETA
)

print(
    "polarization q =",
    engine.POL_Q
)


print()
print(
    "Thermal model:"
)

print(
    "Temperature =",
    TEMPERATURE_K,
    "K"
)

print(
    "u =",
    U_THERMAL,
    "m/s"
)

print(
    "D = ku/Gamma =",
    DOPPLER_D
)


print()
print(
    "Transit model:"
)

print(
    "Gamma_transit/Gamma =",
    GAMMA_TRANSIT
)

print(
    "rho0 steady-state residual =",
    RHO0_RESIDUAL
)


print()
print(
    "FIXED velocity grids:"
)

print(
    "main dq =",
    Q_MAIN_STEP
)

print(
    "main q points =",
    len(
        Q_MAIN
    )
)

print(
    "main q range =",
    Q_MAIN[
        0
    ],
    "to",
    Q_MAIN[
        -1
    ]
)

print(
    "main captured Maxwell mass =",
    MASS_MAIN
)


print()
print(
    "refined dq =",
    Q_REFINE_STEP
)

print(
    "refined q points =",
    len(
        Q_REFINE
    )
)

print(
    "refined captured Maxwell mass =",
    MASS_REFINE
)


print()
print(
    "boundary q points =",
    len(
        Q_BOUNDARY
    )
)

print(
    "boundary captured Maxwell mass =",
    MASS_BOUNDARY
)


# ============================================================
# 17. Convergence precheck
# ============================================================

PRECHECK_MAIN = []

PRECHECK_REF = []

PRECHECK_BOUNDARY = []


if RUN_PRECHECK:

    print()
    print(
        "=" * 80
    )

    print(
        "PRECHECK: velocity integration convergence"
    )

    print(
        "=" * 80
    )


    for offset_mhz in PRECHECK_OFFSETS_MHZ:

        delta = (
            (
                engine.HF_OFFSET_MHZ[
                    LOCK_FE
                ]
                +
                offset_mhz
            )
            /
            engine.GAMMA_MHZ
        )


        print()
        print(
            "offset =",
            offset_mhz,
            "MHz"
        )


        print(
            "  calculating dq =",
            Q_MAIN_STEP
        )

        r_main = thermal_main(
            delta
        )


        print(
            "  calculating dq =",
            Q_REFINE_STEP
        )

        r_ref = thermal_refined(
            delta
        )


        print(
            "  calculating +/-5u boundary"
        )

        r_bound = thermal_boundary(
            delta
        )


        PRECHECK_MAIN.append(
            r_main[
                "Z"
            ]
        )

        PRECHECK_REF.append(
            r_ref[
                "Z"
            ]
        )

        PRECHECK_BOUNDARY.append(
            r_bound[
                "Z"
            ]
        )


        print(
            "  |Z main| =",
            abs(
                r_main[
                    "Z"
                ]
            )
        )

        print(
            "  |Z refined| =",
            abs(
                r_ref[
                    "Z"
                ]
            )
        )

        print(
            "  linearity error(main) =",
            r_main[
                "linearity_error"
            ]
        )


    PRECHECK_MAIN = np.asarray(
        PRECHECK_MAIN
    )

    PRECHECK_REF = np.asarray(
        PRECHECK_REF
    )

    PRECHECK_BOUNDARY = np.asarray(
        PRECHECK_BOUNDARY
    )


    # --------------------------------------------------------
    # Vector-norm convergence.
    #
    # This avoids dividing by Z at a point where the true
    # complex response happens to be close to zero.
    # --------------------------------------------------------

    GRID_VECTOR_ERROR = (
        np.linalg.norm(
            PRECHECK_MAIN
            -
            PRECHECK_REF
        )
        /
        max(
            np.linalg.norm(
                PRECHECK_REF
            ),
            1e-300
        )
    )


    BOUNDARY_VECTOR_ERROR = (
        np.linalg.norm(
            PRECHECK_BOUNDARY
            -
            PRECHECK_MAIN
        )
        /
        max(
            np.linalg.norm(
                PRECHECK_BOUNDARY
            ),
            1e-300
        )
    )


    # --------------------------------------------------------
    # Also print scaled pointwise errors.
    #
    # All points use ONE common response scale.
    # --------------------------------------------------------

    COMMON_Z_SCALE = max(
        np.max(
            np.abs(
                PRECHECK_REF
            )
        ),
        1e-300
    )


    POINTWISE_GRID_ERRORS = (
        np.abs(
            PRECHECK_MAIN
            -
            PRECHECK_REF
        )
        /
        COMMON_Z_SCALE
    )


    POINTWISE_BOUNDARY_ERRORS = (
        np.abs(
            PRECHECK_BOUNDARY
            -
            PRECHECK_MAIN
        )
        /
        COMMON_Z_SCALE
    )


    print()
    print(
        "=" * 80
    )

    print(
        "PRECHECK SUMMARY"
    )

    print(
        "=" * 80
    )


    for (
        offset,
        e_grid,
        e_bound

    ) in zip(
        PRECHECK_OFFSETS_MHZ,
        POINTWISE_GRID_ERRORS,
        POINTWISE_BOUNDARY_ERRORS
    ):

        print(
            f"offset={offset:+5.1f} MHz"
            f"   scaled grid error={e_grid:.6e}"
            f"   scaled boundary error={e_bound:.6e}"
        )


    print()
    print(
        "vector dq=0.50 -> 0.25 error =",
        GRID_VECTOR_ERROR
    )

    print(
        "vector +/-4u -> +/-5u error =",
        BOUNDARY_VECTOR_ERROR
    )


    GRID_PASS = (
        GRID_VECTOR_ERROR
        <
        GRID_ERROR_LIMIT
    )


    BOUNDARY_PASS = (
        BOUNDARY_VECTOR_ERROR
        <
        BOUNDARY_ERROR_LIMIT
    )


    if GRID_PASS:

        print(
            "PASS: fixed velocity grid refinement."
        )

    else:

        print(
            "FAIL: fixed velocity grid refinement."
        )


    if BOUNDARY_PASS:

        print(
            "PASS: Maxwell boundary convergence."
        )

    else:

        print(
            "FAIL: Maxwell boundary convergence."
        )


    if (
        ABORT_IF_PRECHECK_FAILS
        and
        (
            not GRID_PASS
            or
            not BOUNDARY_PASS
        )
    ):

        raise RuntimeError(
            "\nFINAL thermal calculation aborted.\n"
            "Velocity integration is not converged.\n\n"
            "Do NOT interpret thermal lock-point results.\n"
        )


else:

    GRID_VECTOR_ERROR = np.nan
    BOUNDARY_VECTOR_ERROR = np.nan


# ============================================================
# 18. Laser-frequency grid
# ============================================================

def regular_grid(
    start,
    stop,
    step
):

    return np.arange(
        start,
        stop
        +
        0.5
        *
        step,
        step
    )


CENTER_GRID = regular_grid(
    -CENTER_SPAN_MHZ,
    +CENTER_SPAN_MHZ,
    CENTER_STEP_MHZ
)

LEFT_GRID = regular_grid(
    -FULL_SPAN_MHZ,
    -CENTER_SPAN_MHZ,
    WING_STEP_MHZ
)

RIGHT_GRID = regular_grid(
    +CENTER_SPAN_MHZ,
    +FULL_SPAN_MHZ,
    WING_STEP_MHZ
)


LOCAL_OFFSET_MHZ = np.unique(
    np.round(
        np.concatenate(
            [
                LEFT_GRID,
                CENTER_GRID,
                RIGHT_GRID
            ]
        ),
        10
    )
)


LOCK_CENTER_MHZ = (
    engine.HF_OFFSET_MHZ[
        LOCK_FE
    ]
)


DETUNING_MHZ = (
    LOCK_CENTER_MHZ
    +
    LOCAL_OFFSET_MHZ
)


DELTA_SCAN = (
    DETUNING_MHZ
    /
    engine.GAMMA_MHZ
)


# ============================================================
# 19. No-Doppler reference
# ============================================================

print()
print(
    "=" * 80
)

print(
    "Calculating no-Doppler reference"
)

print(
    "=" * 80
)


P_PLUS_0 = np.zeros(
    len(
        DELTA_SCAN
    ),
    dtype=complex
)

P_MINUS_0 = np.zeros_like(
    P_PLUS_0
)

Z_0 = np.zeros_like(
    P_PLUS_0
)


for i, delta in enumerate(
    DELTA_SCAN
):

    result = (
        engine.response_at_detuning(
            delta=delta,
            kv=0.0
        )
    )


    P_PLUS_0[
        i
    ] = result[
        "P_plus"
    ]


    P_MINUS_0[
        i
    ] = result[
        "P_minus"
    ]


    Z_0[
        i
    ] = result[
        "Z"
    ]


# ============================================================
# 20. Main FINAL thermal scan
# ============================================================

print()
print(
    "=" * 80
)

print(
    "Calculating FINAL fixed-grid thermal spectrum"
)

print(
    "=" * 80
)


P_PLUS_T = np.zeros_like(
    P_PLUS_0
)

P_MINUS_T = np.zeros_like(
    P_PLUS_0
)

Z_T = np.zeros_like(
    P_PLUS_0
)

LINEARITY_ERRORS = np.zeros(
    len(
        DELTA_SCAN
    )
)


for i, delta in enumerate(
    DELTA_SCAN
):

    print()
    print(
        f"{i+1:3d}/{len(DELTA_SCAN)}"
        f"   offset="
        f"{LOCAL_OFFSET_MHZ[i]:+7.3f} MHz"
    )


    result = thermal_main(
        delta
    )


    P_PLUS_T[
        i
    ] = result[
        "P_plus"
    ]


    P_MINUS_T[
        i
    ] = result[
        "P_minus"
    ]


    Z_T[
        i
    ] = result[
        "Z"
    ]


    LINEARITY_ERRORS[
        i
    ] = result[
        "linearity_error"
    ]


# ============================================================
# 21. Analysis utilities
# ============================================================

def complex_center_slope(
    offset_mhz,
    Z,
    fit_window_mhz=2.0
):

    mask = (
        np.abs(
            offset_mhz
        )
        <=
        fit_window_mhz
    )


    x = (
        offset_mhz[
            mask
        ]
        /
        engine.GAMMA_MHZ
    )


    zz = Z[
        mask
    ]


    cr = np.polyfit(
        x,
        np.real(
            zz
        ),
        3
    )


    ci = np.polyfit(
        x,
        np.imag(
            zz
        ),
        3
    )


    return (
        cr[-2]
        +
        1j
        *
        ci[-2]
    )


def demodulate(
    Z,
    phase
):

    return np.real(
        Z
        *
        np.exp(
            -1j
            *
            phase
        )
    )


def find_zero_bracket(
    x,
    y,
    search_window=3.0
):

    mask = (
        np.abs(
            x
        )
        <=
        search_window
    )


    xx = x[
        mask
    ]

    yy = y[
        mask
    ]


    candidates = []


    for i in range(
        len(xx)
        - 1
    ):

        if (
            yy[i]
            ==
            0
        ):

            return (
                xx[i],
                xx[i]
            )


        if (
            yy[i]
            *
            yy[
                i + 1
            ]
            <
            0
        ):

            center = (
                xx[i]
                +
                xx[
                    i + 1
                ]
            ) / 2.0


            candidates.append(
                (
                    abs(
                        center
                    ),
                    xx[i],
                    xx[
                        i + 1
                    ]
                )
            )


    if not candidates:

        return None


    candidates.sort(
        key=lambda z:
        z[0]
    )


    return (
        candidates[
            0
        ][
            1
        ],
        candidates[
            0
        ][
            2
        ]
    )


def refined_zero_from_curve(
    x,
    y,
    window=3.0
):

    mask = (
        np.abs(
            x
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


    roots = []


    for i in range(
        len(xx)
        - 1
    ):

        if (
            yy[i]
            ==
            0
        ):

            roots.append(
                xx[i]
            )


        elif (
            yy[i]
            *
            yy[
                i + 1
            ]
            <
            0
        ):

            roots.append(
                brentq(
                    interp,
                    xx[i],
                    xx[
                        i + 1
                    ]
                )
            )


    if not roots:

        return np.nan


    return min(
        roots,
        key=abs
    )


def extrema_metrics(
    x,
    y,
    zero
):

    order = np.argsort(
        x
    )


    xx = x[
        order
    ]

    yy = y[
        order
    ]


    interp = PchipInterpolator(
        xx,
        yy
    )


    dense = np.linspace(
        xx.min(),
        xx.max(),
        30001
    )


    y_dense = interp(
        dense
    )


    prominence = (
        0.01
        *
        (
            y_dense.max()
            -
            y_dense.min()
        )
    )


    maxima, _ = find_peaks(
        y_dense,
        prominence=prominence
    )


    minima, _ = find_peaks(
        -y_dense,
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
        dense[
            ext
        ]
        <
        zero
    ]


    right = ext[
        dense[
            ext
        ]
        >
        zero
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


    il = left[
        -1
    ]

    ir = right[
        0
    ]


    return (
        abs(
            y_dense[
                ir
            ]
            -
            y_dense[
                il
            ]
        ),

        dense[
            ir
        ]
        -
        dense[
            il
        ],

        dense[
            il
        ],

        dense[
            ir
        ]
    )


# ============================================================
# 22. Main-grid thermal discriminator
# ============================================================

SLOPE_T_MAIN = complex_center_slope(
    LOCAL_OFFSET_MHZ,
    Z_T
)


PHI_T_MAIN = np.angle(
    SLOPE_T_MAIN
)


S_T_MAIN = demodulate(
    Z_T,
    PHI_T_MAIN
)


ZERO_T_MAIN = refined_zero_from_curve(
    LOCAL_OFFSET_MHZ,
    S_T_MAIN
)


# ============================================================
# 23. Refined-q lock extraction
# ============================================================

if RUN_REFINED_LOCK:

    print()
    print(
        "=" * 80
    )

    print(
        "Refining final lock with dq =",
        Q_REFINE_STEP
    )

    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # 23a. Refined complex center slope
    # --------------------------------------------------------

    Z_PHASE_FINE = []


    for offset_mhz in (
        REFINED_PHASE_OFFSETS_MHZ
    ):

        delta = (
            (
                LOCK_CENTER_MHZ
                +
                offset_mhz
            )
            /
            engine.GAMMA_MHZ
        )


        result = thermal_refined(
            delta
        )


        Z_PHASE_FINE.append(
            result[
                "Z"
            ]
        )


    Z_PHASE_FINE = np.asarray(
        Z_PHASE_FINE
    )


    x_phase_norm = (
        REFINED_PHASE_OFFSETS_MHZ
        /
        engine.GAMMA_MHZ
    )


    real_fit = np.polyfit(
        x_phase_norm,
        np.real(
            Z_PHASE_FINE
        ),
        3
    )


    imag_fit = np.polyfit(
        x_phase_norm,
        np.imag(
            Z_PHASE_FINE
        ),
        3
    )


    SLOPE_T_FINE = (
        real_fit[
            -2
        ]
        +
        1j
        *
        imag_fit[
            -2
        ]
    )


    PHI_T_FINE = np.angle(
        SLOPE_T_FINE
    )


    # --------------------------------------------------------
    # 23b. Find approximate root from main scan
    # --------------------------------------------------------

    S_MAIN_AT_FINE_PHASE = demodulate(
        Z_T,
        PHI_T_FINE
    )


    ZERO_APPROX = refined_zero_from_curve(
        LOCAL_OFFSET_MHZ,
        S_MAIN_AT_FINE_PHASE
    )


    if not np.isfinite(
        ZERO_APPROX
    ):

        raise RuntimeError(
            "\nNo thermal zero crossing exists in the "
            "main fixed-grid spectrum.\n"
        )


    # --------------------------------------------------------
    # 23c. Local refined-q frequency scan around root
    # --------------------------------------------------------

    lock_offsets = np.arange(
        ZERO_APPROX
        -
        REFINED_LOCK_HALF_WIDTH_MHZ,

        ZERO_APPROX
        +
        REFINED_LOCK_HALF_WIDTH_MHZ
        +
        0.5
        *
        REFINED_LOCK_STEP_MHZ,

        REFINED_LOCK_STEP_MHZ
    )


    lock_offsets = np.unique(
        np.round(
            np.append(
                lock_offsets,
                ZERO_APPROX
            ),
            10
        )
    )


    Z_LOCK_FINE = np.zeros(
        len(
            lock_offsets
        ),
        dtype=complex
    )


    for i, offset_mhz in enumerate(
        lock_offsets
    ):

        print(
            f"  refined lock point "
            f"{i+1:2d}/{len(lock_offsets)}"
            f"   offset={offset_mhz:+.4f} MHz"
        )


        delta = (
            (
                LOCK_CENTER_MHZ
                +
                offset_mhz
            )
            /
            engine.GAMMA_MHZ
        )


        Z_LOCK_FINE[
            i
        ] = thermal_refined(
            delta
        )[
            "Z"
        ]


    S_LOCK_FINE = demodulate(
        Z_LOCK_FINE,
        PHI_T_FINE
    )


    ZERO_T_FINAL = refined_zero_from_curve(
        lock_offsets,
        S_LOCK_FINE,
        window=max(
            abs(
                lock_offsets
            )
        )
        +
        0.1
    )


    SLOPE_T_FINAL = SLOPE_T_FINE

    PHI_T_FINAL = PHI_T_FINE


else:

    ZERO_T_FINAL = ZERO_T_MAIN

    SLOPE_T_FINAL = SLOPE_T_MAIN

    PHI_T_FINAL = PHI_T_MAIN


# ============================================================
# 24. Final thermal signal on main spectrum
# ============================================================

S_T_FINAL_PHASE = demodulate(
    Z_T,
    PHI_T_FINAL
)


(
    VPP_T,
    SEP_T,
    EXT_LEFT_T,
    EXT_RIGHT_T

) = extrema_metrics(
    LOCAL_OFFSET_MHZ,
    S_T_FINAL_PHASE,
    ZERO_T_FINAL
)


# ============================================================
# 25. No-Doppler reference
#
# Report BOTH:
# 1. its own optimum
# 2. the thermal optimum phase
# ============================================================

SLOPE_0 = complex_center_slope(
    LOCAL_OFFSET_MHZ,
    Z_0
)


PHI_0 = np.angle(
    SLOPE_0
)


S_0_OPT = demodulate(
    Z_0,
    PHI_0
)


ZERO_0_OPT = refined_zero_from_curve(
    LOCAL_OFFSET_MHZ,
    S_0_OPT
)


S_0_THERMAL_PHASE = demodulate(
    Z_0,
    PHI_T_FINAL
)


ZERO_0_THERMAL_PHASE = refined_zero_from_curve(
    LOCAL_OFFSET_MHZ,
    S_0_THERMAL_PHASE
)


SLOPE_0_AT_THERMAL_PHASE = abs(
    np.real(
        SLOPE_0
        *
        np.exp(
            -1j
            *
            PHI_T_FINAL
        )
    )
)


SLOPE_T_MAG = abs(
    SLOPE_T_FINAL
)


# ============================================================
# 26. beta = 0 thermal null test
# ============================================================

if np.isfinite(
    ZERO_T_FINAL
):

    TEST_OFFSET_MHZ = (
        ZERO_T_FINAL
        +
        2.0
    )

else:

    TEST_OFFSET_MHZ = 2.0


TEST_DELTA = (
    (
        LOCK_CENTER_MHZ
        +
        TEST_OFFSET_MHZ
    )
    /
    engine.GAMMA_MHZ
)


NORMAL_TEST = thermal_main(
    TEST_DELTA,
    beta=engine.BETA
)


NULL_TEST = thermal_main(
    TEST_DELTA,
    beta=0.0
)


NULL_RATIO = (
    abs(
        NULL_TEST[
            "Z"
        ]
    )
    /
    max(
        abs(
            NORMAL_TEST[
                "Z"
            ]
        ),
        1e-300
    )
)


# ============================================================
# 27. Velocity profile at final lock
# ============================================================

if np.isfinite(
    ZERO_T_FINAL
):

    PROFILE_OFFSET = ZERO_T_FINAL

else:

    PROFILE_OFFSET = 0.0


PROFILE_DELTA = (
    (
        LOCK_CENTER_MHZ
        +
        PROFILE_OFFSET
    )
    /
    engine.GAMMA_MHZ
)


PROFILE = thermal_refined(
    PROFILE_DELTA,
    return_profile=True
)


# ============================================================
# 28. Final diagnostics
# ============================================================

print()
print(
    "=" * 80
)

print(
    "FINAL CONVERGED THERMAL MTS RESULT"
)

print(
    "=" * 80
)


print()
print(
    "Model condition:"
)

print(
    "T =",
    TEMPERATURE_K,
    "K"
)

print(
    "Gamma_transit/Gamma =",
    GAMMA_TRANSIT
)

print(
    "main dq =",
    Q_MAIN_STEP
)

print(
    "refined dq =",
    Q_REFINE_STEP
)


print()
print(
    "Thermal discriminator:"
)

print(
    "complex center slope =",
    SLOPE_T_FINAL
)

print(
    "|center slope| =",
    SLOPE_T_MAG
)

print(
    "optimum demodulation phase =",
    np.degrees(
        PHI_T_FINAL
    ),
    "deg"
)

print(
    "main-grid lock shift =",
    ZERO_T_MAIN,
    "MHz"
)

print(
    "FINAL refined lock shift =",
    ZERO_T_FINAL,
    "MHz"
)

print(
    "FINAL refined lock shift =",
    (
        ZERO_T_FINAL
        *
        1000.0
        if np.isfinite(
            ZERO_T_FINAL
        )
        else np.nan
    ),
    "kHz"
)

print(
    "nearest-extrema Vpp =",
    VPP_T
)

print(
    "nearest-extrema separation =",
    SEP_T,
    "MHz"
)


print()
print(
    "No-Doppler reference:"
)

print(
    "own optimum phase =",
    np.degrees(
        PHI_0
    ),
    "deg"
)

print(
    "own optimum lock shift =",
    ZERO_0_OPT,
    "MHz"
)

print(
    "lock shift at THERMAL phase =",
    ZERO_0_THERMAL_PHASE,
    "MHz"
)

print(
    "|complex slope| =",
    abs(
        SLOPE_0
    )
)


print()
print(
    "Thermal / no-Doppler comparison:"
)

print(
    "thermal slope magnitude =",
    SLOPE_T_MAG
)

print(
    "no-Doppler slope projected on thermal phase =",
    SLOPE_0_AT_THERMAL_PHASE
)

print(
    "thermal/no-Doppler slope ratio =",
    (
        SLOPE_T_MAG
        /
        max(
            SLOPE_0_AT_THERMAL_PHASE,
            1e-300
        )
    )
)


print()
print(
    "Numerical validation:"
)

print(
    "dq=0.50 -> 0.25 vector error =",
    GRID_VECTOR_ERROR
)

print(
    "+/-4u -> +/-5u vector error =",
    BOUNDARY_VECTOR_ERROR
)

print(
    "max heterodyne-linearity error =",
    np.max(
        LINEARITY_ERRORS
    )
)


print()
print(
    "beta=0 thermal null ratio =",
    NULL_RATIO
)


ALL_FINITE = (
    np.all(
        np.isfinite(
            P_PLUS_T
        )
    )
    and
    np.all(
        np.isfinite(
            P_MINUS_T
        )
    )
    and
    np.all(
        np.isfinite(
            Z_T
        )
    )
)


print(
    "All finite =",
    ALL_FINITE
)


print()
print(
    "=" * 80
)

print(
    "FINAL VALIDATION"
)

print(
    "=" * 80
)


if GRID_VECTOR_ERROR < GRID_ERROR_LIMIT:

    print(
        "PASS: fixed velocity-grid convergence."
    )

else:

    print(
        "FAIL: fixed velocity-grid convergence."
    )


if BOUNDARY_VECTOR_ERROR < BOUNDARY_ERROR_LIMIT:

    print(
        "PASS: Maxwell boundary convergence."
    )

else:

    print(
        "FAIL: Maxwell boundary convergence."
    )


if NULL_RATIO < 1e-3:

    print(
        "PASS: beta=0 thermal null test."
    )

else:

    print(
        "FAIL: beta=0 thermal null test."
    )


if np.isfinite(
    ZERO_T_FINAL
):

    print(
        "PASS: finite thermal lock zero found."
    )

else:

    print(
        "FAIL: no thermal lock zero found."
    )


if ALL_FINITE:

    print(
        "PASS: all calculated thermal arrays finite."
    )


# ============================================================
# 29. Figure 1
#
# Maxwell distribution
# ============================================================

q_plot = np.linspace(
    -4.0
    *
    DOPPLER_D,

    +4.0
    *
    DOPPLER_D,

    2001
)


w_plot = maxwell_weight_q(
    q_plot
)


plt.figure(
    figsize=(
        9,
        5
    )
)


plt.plot(
    q_plot,
    w_plot,
    label="Maxwell distribution"
)


for Fe in engine.FE_LIST:

    q_res = (
        PROFILE_DELTA
        -
        engine.HF_OFFSET[
            Fe
        ]
    )


    plt.axvline(
        +q_res,
        linestyle="--",
        linewidth=0.8,
        label=(
            f"F'={Fe}"
        )
    )


    if abs(
        q_res
    ) > 1e-10:

        plt.axvline(
            -q_res,
            linestyle=":",
            linewidth=0.8
        )


plt.xlabel(
    r"Normalized Doppler shift "
    r"$q=kv/\Gamma$"
)

plt.ylabel(
    "Maxwell probability density"
)

plt.title(
    "Rb87 thermal velocity distribution"
)

plt.legend(
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 2
#
# Generated third-order sidebands
# ============================================================

SIDE_SCALE = max(
    np.max(
        np.abs(
            P_PLUS_0
        )
    ),

    np.max(
        np.abs(
            P_MINUS_0
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
    LOCAL_OFFSET_MHZ,
    np.abs(
        P_PLUS_0
    )
    /
    SIDE_SCALE,
    label=
    r"No Doppler: $\omega+\Omega_m$"
)


plt.plot(
    LOCAL_OFFSET_MHZ,
    np.abs(
        P_MINUS_0
    )
    /
    SIDE_SCALE,
    label=
    r"No Doppler: $\omega-\Omega_m$"
)


plt.plot(
    LOCAL_OFFSET_MHZ,
    np.abs(
        P_PLUS_T
    )
    /
    SIDE_SCALE,
    "--",
    label=
    r"Thermal: $\omega+\Omega_m$"
)


plt.plot(
    LOCAL_OFFSET_MHZ,
    np.abs(
        P_MINUS_T
    )
    /
    SIDE_SCALE,
    "--",
    label=
    r"Thermal: $\omega-\Omega_m$"
)


plt.axvline(
    0,
    linestyle="--",
    linewidth=0.8
)


plt.xlabel(
    r"Detuning from "
    r"$F=2\rightarrow F'=3$ (MHz)"
)

plt.ylabel(
    "Third-order polarization "
    "(common normalization)"
)

plt.title(
    "Fixed-grid Maxwell average of MTS sidebands"
)

plt.legend(
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 3
#
# Complex heterodyne response
# ============================================================

Z_SCALE = max(
    np.max(
        np.abs(
            Z_0
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
    LOCAL_OFFSET_MHZ,
    np.real(
        Z_0
    )
    /
    Z_SCALE,
    label="No Doppler Re(Z)"
)


plt.plot(
    LOCAL_OFFSET_MHZ,
    np.imag(
        Z_0
    )
    /
    Z_SCALE,
    label="No Doppler Im(Z)"
)


plt.plot(
    LOCAL_OFFSET_MHZ,
    np.real(
        Z_T
    )
    /
    Z_SCALE,
    "--",
    label="Thermal Re(Z)"
)


plt.plot(
    LOCAL_OFFSET_MHZ,
    np.imag(
        Z_T
    )
    /
    Z_SCALE,
    "--",
    label="Thermal Im(Z)"
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
    r"Detuning from "
    r"$F=2\rightarrow F'=3$ (MHz)"
)

plt.ylabel(
    "Complex RF response "
    "(common normalization)"
)

plt.title(
    "Converged thermal MTS heterodyne response"
)

plt.legend(
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 4
#
# Final MTS discriminator
# ============================================================

S_0_FINAL_PHASE = demodulate(
    Z_0,
    PHI_T_FINAL
)


SIGNAL_SCALE = max(
    np.max(
        np.abs(
            S_0_FINAL_PHASE
        )
    ),

    np.max(
        np.abs(
            S_T_FINAL_PHASE
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
    LOCAL_OFFSET_MHZ,
    S_0_FINAL_PHASE
    /
    SIGNAL_SCALE,
    label="No Doppler"
)


plt.plot(
    LOCAL_OFFSET_MHZ,
    S_T_FINAL_PHASE
    /
    SIGNAL_SCALE,
    "--",
    label="Full thermal Rb87"
)


plt.axhline(
    0,
    linewidth=0.8
)


plt.axvline(
    0,
    linestyle="--",
    linewidth=0.8,
    label="Nominal F'=3 resonance"
)


if np.isfinite(
    ZERO_T_FINAL
):

    plt.axvline(
        ZERO_T_FINAL,
        linestyle=":",
        linewidth=1.0,
        label=(
            "thermal lock = "
            f"{ZERO_T_FINAL*1e3:.1f} kHz"
        )
    )


plt.xlabel(
    r"Detuning from "
    r"$F=2\rightarrow F'=3$ (MHz)"
)

plt.ylabel(
    "MTS error signal "
    "(common normalization)"
)

plt.title(
    "FINAL converged thermal MTS discriminator\n"
    f"T={TEMPERATURE_K:.0f} K, "
    f"Gamma_t/Gamma={GAMMA_TRANSIT:.3f}, "
    f"phi={np.degrees(PHI_T_FINAL):.1f} deg"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 5
#
# TRUE complex velocity-class contribution
# ============================================================

q_profile = PROFILE[
    "q"
]

Z_integrand = PROFILE[
    "Z_integrand"
]


INTEGRAND_SCALE = max(
    np.max(
        np.abs(
            Z_integrand
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
    q_profile,
    np.real(
        Z_integrand
    )
    /
    INTEGRAND_SCALE,
    label=
    r"$\mathrm{Re}[f(q)Z(q)]$"
)


plt.plot(
    q_profile,
    np.imag(
        Z_integrand
    )
    /
    INTEGRAND_SCALE,
    label=
    r"$\mathrm{Im}[f(q)Z(q)]$"
)


plt.axhline(
    0,
    linewidth=0.8
)


for Fe in engine.FE_LIST:

    q_res = (
        PROFILE_DELTA
        -
        engine.HF_OFFSET[
            Fe
        ]
    )


    plt.axvline(
        +q_res,
        linestyle="--",
        linewidth=0.7
    )


    if abs(
        q_res
    ) > 1e-10:

        plt.axvline(
            -q_res,
            linestyle=":",
            linewidth=0.7
        )


plt.xlabel(
    r"$q=kv/\Gamma$"
)

plt.ylabel(
    "Complex velocity-class contribution "
    "(common normalization)"
)

plt.title(
    "Coherent velocity-class contributions at final lock"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 6
#
# Cumulative complex velocity integral
#
# This directly shows cancellation between velocity classes.
# ============================================================

dq_profile = np.diff(
    q_profile
)


cum_real = np.zeros(
    len(
        q_profile
    )
)

cum_imag = np.zeros(
    len(
        q_profile
    )
)


for i in range(
    1,
    len(
        q_profile
    )
):

    cum_real[
        i
    ] = (
        cum_real[
            i - 1
        ]
        +
        0.5
        *
        (
            np.real(
                Z_integrand[
                    i - 1
                ]
            )
            +
            np.real(
                Z_integrand[
                    i
                ]
            )
        )
        *
        dq_profile[
            i - 1
        ]
    )


    cum_imag[
        i
    ] = (
        cum_imag[
            i - 1
        ]
        +
        0.5
        *
        (
            np.imag(
                Z_integrand[
                    i - 1
                ]
            )
            +
            np.imag(
                Z_integrand[
                    i
                ]
            )
        )
        *
        dq_profile[
            i - 1
        ]
    )


CUM_SCALE = max(
    np.max(
        np.abs(
            cum_real
        )
    ),

    np.max(
        np.abs(
            cum_imag
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
    q_profile,
    cum_real
    /
    CUM_SCALE,
    label="Cumulative Re(Z)"
)


plt.plot(
    q_profile,
    cum_imag
    /
    CUM_SCALE,
    label="Cumulative Im(Z)"
)


plt.xlabel(
    r"Upper integration limit "
    r"$q=kv/\Gamma$"
)

plt.ylabel(
    "Cumulative thermal heterodyne response"
)

plt.title(
    "Coherent cancellation in the Maxwell velocity integral"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.show()