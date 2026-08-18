"""
14c_rb87_full_thermal_mts_composite.py

Stage F3c

FINAL thermal 87Rb hyperfine + Zeeman rho^(3) MTS
using a FIXED COMPOSITE velocity grid.

Why 14c?
--------
14b showed:

    +/-4u -> +/-5u boundary convergence:
        excellent (~1e-11)

    heterodyne linearity:
        machine precision (~1e-15)

but:

    global dq = 0.50 -> 0.25:
        ~28% error

Therefore the problem is NOT the Maxwell boundary and NOT the
order of thermal averaging / heterodyne detection.

The remaining issue is narrow velocity-class structure.

Strategy
--------
All laser detunings use exactly the SAME q grid.

Background:
    dq = 0.50

Fine windows around the physical velocity families:
    q ~ 0
    q ~ +/- |Delta_F'=2|
    q ~ +/- |Delta_F'=1|

The windows are broad enough to contain:

    full laser scan
    +/-3 modulation-frequency shifts
    additional resonance margin

Local refinement hierarchy:

    dq_local = 0.10
            -> 0.05
            -> 0.025

FINAL spectrum uses dq_local = 0.05.

dq_local = 0.025 is used as independent convergence reference
and again for final local lock extraction.

There is NO FAST MODE in this file.

If convergence fails, the script aborts and does not report a
"final" thermal lock point.
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
# 0. Calculation mode
# ============================================================

MODE = "FINAL"

if MODE != "FINAL":
    raise RuntimeError(
        "14c is a FINAL quantitative script. "
        "MODE must remain FINAL."
    )


# ============================================================
# 1. Physical parameters
# ============================================================

TEMPERATURE_K = 300.0

# IMPORTANT:
# This remains a model parameter, as established by Stage 13b.
GAMMA_TRANSIT = 0.020

LOCK_FE = 3


# ============================================================
# 2. Velocity-grid parameters
# ============================================================

VELOCITY_SPAN_U = 4.0
VELOCITY_SPAN_U_TEST = 5.0

# Coarse background
DQ_BACKGROUND = 0.50

# Local convergence hierarchy
DQ_LOCAL_1 = 0.10
DQ_LOCAL_2 = 0.05       # final production grid
DQ_LOCAL_3 = 0.025      # refinement reference


# ============================================================
# 3. Frequency scan
# ============================================================

CENTER_SPAN_MHZ = 3.0
CENTER_STEP_MHZ = 0.25

FULL_SPAN_MHZ = 12.0
WING_STEP_MHZ = 1.0


# ============================================================
# 4. Fine-window construction
# ============================================================

# We include modulation-related structures up to rho^(3),
# therefore +/-3 fm is a conservative range.

MAX_MODULATION_ORDER = 3

# Additional width around the expected resonances.
RESONANCE_MARGIN_Q = 2.0


# ============================================================
# 5. Convergence settings
# ============================================================

PRECHECK_OFFSETS_MHZ = np.array(
    [
        -2.0,
         0.0,
        +2.0
    ]
)

GRID_ERROR_LIMIT = 1.0e-2
BOUNDARY_ERROR_LIMIT = 1.0e-3

ABORT_IF_PRECHECK_FAILS = True


# ============================================================
# 6. Refined final lock
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

REFINED_LOCK_HALF_WIDTH_MHZ = 0.75
REFINED_LOCK_STEP_MHZ = 0.25


# ============================================================
# 7. Physical constants
# ============================================================

KB = 1.380649e-23
AMU = 1.66053906660e-27

MASS_RB87 = 86.9091805 * AMU

LAMBDA_D2 = 780.24e-9


# ============================================================
# 8. Load Stage-13 validated atomic engine
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

ENGINE_PATH = (
    THIS_DIR
    /
    "13_rb87_hyperfine_zeeman_mts.py"
)

if not ENGINE_PATH.exists():
    raise FileNotFoundError(
        f"\nCannot find:\n{ENGINE_PATH}\n\n"
        "Place 14c and Stage-13 in the same directory."
    )


MODULE_NAME = "rb87_engine_14c"

spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    ENGINE_PATH
)

engine = importlib.util.module_from_spec(spec)

sys.modules[MODULE_NAME] = engine

spec.loader.exec_module(engine)


# ============================================================
# 9. Rebuild transit dissipator
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
        engine.lindblad_superoperator(C)
    )


def make_transit_dissipator(gamma_transit):

    L = L_DISS_SPONT.copy()

    if gamma_transit <= 0:
        return L

    branch_rate = (
        gamma_transit
        /
        (
            len(engine.GROUND_STATES)
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
                        engine.INDEX[state_to],
                        engine.INDEX[state_from],
                        np.sqrt(branch_rate)
                    )
                ]
            )

            L += (
                engine.lindblad_superoperator(C)
            )

    return L


L_DISS_SELECTED = make_transit_dissipator(
    GAMMA_TRANSIT
)


def build_L0_selected(delta):

    energies = np.zeros(
        engine.N,
        dtype=float
    )

    # Ground manifolds
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
                engine.GF_GROUND[Fg]
                *
                engine.ZEEMAN_UNIT
                *
                mg
            )

    # Excited manifolds
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
                engine.HF_OFFSET[Fe]
                +
                engine.GF_EXCITED[Fe]
                *
                engine.ZEEMAN_UNIT
                *
                me
            )

    H0 = csc_matrix(
        np.diag(energies)
    )

    return (
        engine.commutator_superoperator(H0)
        +
        L_DISS_SELECTED
    )


engine.GAMMA_TRANSIT = GAMMA_TRANSIT
engine.build_L0 = build_L0_selected


RHO0_RESIDUAL = np.linalg.norm(
    engine.build_L0(0.0)
    @
    engine.RHO0_VEC
)


# ============================================================
# 10. Thermal distribution
# ============================================================

def thermal_speed_u(T):

    return np.sqrt(
        2.0
        *
        KB
        *
        T
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


def maxwell_weight_q(q):

    return (
        np.exp(
            -(q / DOPPLER_D)**2
        )
        /
        (
            np.sqrt(np.pi)
            *
            DOPPLER_D
        )
    )


# ============================================================
# 11. Fixed physical fine windows
# ============================================================

SCAN_HALF_WIDTH_Q = (
    FULL_SPAN_MHZ
    /
    engine.GAMMA_MHZ
)


FINE_HALF_WIDTH_Q = (
    SCAN_HALF_WIDTH_Q
    +
    MAX_MODULATION_ORDER
    *
    engine.FM
    +
    RESONANCE_MARGIN_Q
)


# Physical velocity families at Delta ~ F'=3:
#
# q ~ 0
# q ~ +/-43.97
# q ~ +/-69.84

FAMILY_CENTERS = []


for Fe in engine.FE_LIST:

    q0 = abs(
        engine.HF_OFFSET[Fe]
    )

    FAMILY_CENTERS.append(
        +q0
    )

    FAMILY_CENTERS.append(
        -q0
    )


FAMILY_CENTERS = np.unique(
    np.round(
        FAMILY_CENTERS,
        10
    )
)


def merge_intervals(intervals):

    intervals = sorted(
        intervals,
        key=lambda z: z[0]
    )

    merged = []

    for left, right in intervals:

        if not merged:
            merged.append(
                [
                    left,
                    right
                ]
            )

        elif left <= merged[-1][1]:

            merged[-1][1] = max(
                merged[-1][1],
                right
            )

        else:

            merged.append(
                [
                    left,
                    right
                ]
            )

    return [
        tuple(x)
        for x in merged
    ]


FINE_INTERVALS = merge_intervals(
    [
        (
            center
            -
            FINE_HALF_WIDTH_Q,

            center
            +
            FINE_HALF_WIDTH_Q
        )

        for center in FAMILY_CENTERS
    ]
)


# ============================================================
# 12. Symmetric lattice
# ============================================================

def symmetric_grid(
    dq,
    span_u
):

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

    return (
        np.arange(
            -N,
            N + 1
        )
        *
        dq
    )


def inside_fine_intervals(q):

    mask = np.zeros(
        q.shape,
        dtype=bool
    )

    for left, right in FINE_INTERVALS:

        mask |= (
            (q >= left)
            &
            (q <= right)
        )

    return mask


# ============================================================
# 13. Fixed composite grid
# ============================================================

def build_composite_grid(
    local_dq,
    span_u
):
    """
    Same physical grid for every laser detuning.

    coarse background:
        dq = 0.5

    fine windows:
        chosen local_dq
    """

    background = symmetric_grid(
        DQ_BACKGROUND,
        span_u
    )

    fine_full = symmetric_grid(
        local_dq,
        span_u
    )

    fine = fine_full[
        inside_fine_intervals(
            fine_full
        )
    ]

    q = np.unique(
        np.round(
            np.concatenate(
                [
                    background,
                    fine
                ]
            ),
            12
        )
    )

    return q


Q_LOCAL_1 = build_composite_grid(
    DQ_LOCAL_1,
    VELOCITY_SPAN_U
)

Q_LOCAL_2 = build_composite_grid(
    DQ_LOCAL_2,
    VELOCITY_SPAN_U
)

Q_LOCAL_3 = build_composite_grid(
    DQ_LOCAL_3,
    VELOCITY_SPAN_U
)


Q_BOUNDARY = build_composite_grid(
    DQ_LOCAL_2,
    VELOCITY_SPAN_U_TEST
)


W_LOCAL_1 = maxwell_weight_q(
    Q_LOCAL_1
)

W_LOCAL_2 = maxwell_weight_q(
    Q_LOCAL_2
)

W_LOCAL_3 = maxwell_weight_q(
    Q_LOCAL_3
)

W_BOUNDARY = maxwell_weight_q(
    Q_BOUNDARY
)


# ============================================================
# 14. Captured Maxwell mass
# ============================================================

def captured_mass(q, w):

    return np.trapezoid(
        w,
        q
    )


MASS_1 = captured_mass(
    Q_LOCAL_1,
    W_LOCAL_1
)

MASS_2 = captured_mass(
    Q_LOCAL_2,
    W_LOCAL_2
)

MASS_3 = captured_mass(
    Q_LOCAL_3,
    W_LOCAL_3
)

MASS_BOUNDARY = captured_mass(
    Q_BOUNDARY,
    W_BOUNDARY
)


# ============================================================
# 15. Atomic-response cache
# ============================================================

ATOMIC_CACHE = {}


def cached_atomic_response(
    delta,
    kv,
    beta=engine.BETA,
    field_scale=1.0
):

    key = (
        round(float(delta), 12),
        round(float(kv), 12),
        round(float(beta), 12),
        round(float(field_scale), 10)
    )

    if key not in ATOMIC_CACHE:

        ATOMIC_CACHE[key] = (
            engine.response_at_detuning(
                delta=delta,
                kv=kv,
                beta=beta,
                field_scale=field_scale
            )
        )

    return ATOMIC_CACHE[key]


# ============================================================
# 16. Heterodyne
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
        np.conj(E_c)
        +
        np.conj(E_minus)
        *
        E_c
    )


# ============================================================
# 17. Thermal average
# ============================================================

def thermal_average_on_grid(
    delta,
    q,
    weight,
    beta=engine.BETA,
    field_scale=1.0,
    return_profile=False,
    verbose=False
):

    Nq = len(q)

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


    for i, kv in enumerate(q):

        if (
            verbose
            and
            (
                i % 500 == 0
                or
                i == Nq - 1
            )
        ):

            print(
                f"        velocity "
                f"{i+1}/{Nq}"
            )

        r = cached_atomic_response(
            delta=delta,
            kv=kv,
            beta=beta,
            field_scale=field_scale
        )

        P_plus_v[i] = r["P_plus"]
        P_minus_v[i] = r["P_minus"]

        Z_v[i] = (
            heterodyne_from_polarization(
                P_plus_v[i],
                P_minus_v[i],
                field_scale=field_scale
            )
        )


    # Coherent average of complex polarization
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


    # Independent linearity validation
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
            abs(Z_direct),
            abs(Z_avg),
            1e-300
        )
    )


    output = {
        "P_plus": P_plus_avg,
        "P_minus": P_minus_avg,
        "Z": Z_avg,
        "Z_direct": Z_direct,
        "linearity_error": linearity_error,
        "Nq": Nq
    }


    if return_profile:

        output.update(
            {
                "q": q,
                "weight": weight,

                "P_plus_v": P_plus_v,
                "P_minus_v": P_minus_v,

                "Z_v": Z_v,

                "Z_integrand":
                weight
                *
                Z_v
            }
        )

    return output


def thermal_level_1(
    delta,
    **kwargs
):

    return thermal_average_on_grid(
        delta,
        Q_LOCAL_1,
        W_LOCAL_1,
        **kwargs
    )


def thermal_main(
    delta,
    **kwargs
):

    return thermal_average_on_grid(
        delta,
        Q_LOCAL_2,
        W_LOCAL_2,
        **kwargs
    )


def thermal_refined(
    delta,
    **kwargs
):

    return thermal_average_on_grid(
        delta,
        Q_LOCAL_3,
        W_LOCAL_3,
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
# 18. Header
# ============================================================

print()
print("=" * 80)

print(
    "Stage F3c FINAL: fixed-composite-grid thermal Rb87 MTS"
)

print("=" * 80)


print()
print(
    "CALCULATION MODE = FINAL"
)


print()
print("Atomic model:")

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
    "fm/Gamma =",
    engine.FM
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
print("Thermal model:")

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
print("Transit model:")

print(
    "Gamma_transit/Gamma =",
    GAMMA_TRANSIT
)

print(
    "rho0 residual =",
    RHO0_RESIDUAL
)


print()
print("Fine velocity families:")

print(
    "centers =",
    FAMILY_CENTERS
)

print(
    "fine half width =",
    FINE_HALF_WIDTH_Q
)

print(
    "merged intervals:"
)

for interval in FINE_INTERVALS:
    print(
        "   ",
        interval
    )


print()
print("Composite grids:")

print(
    "dq_local=0.10: N =",
    len(Q_LOCAL_1),
    " mass =",
    MASS_1
)

print(
    "dq_local=0.05: N =",
    len(Q_LOCAL_2),
    " mass =",
    MASS_2
)

print(
    "dq_local=0.025: N =",
    len(Q_LOCAL_3),
    " mass =",
    MASS_3
)

print(
    "+/-5u boundary: N =",
    len(Q_BOUNDARY),
    " mass =",
    MASS_BOUNDARY
)


# ============================================================
# 19. PRECHECK
# ============================================================

print()
print("=" * 80)

print(
    "PRECHECK: composite velocity-grid convergence"
)

print("=" * 80)


Z_1 = []
Z_2 = []
Z_3 = []
Z_BOUND = []

LINEARITY_PRECHECK = []


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
        "    dq_local =",
        DQ_LOCAL_1
    )

    r1 = thermal_level_1(
        delta
    )


    print(
        "    dq_local =",
        DQ_LOCAL_2
    )

    r2 = thermal_main(
        delta
    )


    print(
        "    dq_local =",
        DQ_LOCAL_3
    )

    r3 = thermal_refined(
        delta,
        verbose=True
    )


    print(
        "    boundary +/-5u"
    )

    rb = thermal_boundary(
        delta
    )


    Z_1.append(
        r1["Z"]
    )

    Z_2.append(
        r2["Z"]
    )

    Z_3.append(
        r3["Z"]
    )

    Z_BOUND.append(
        rb["Z"]
    )


    LINEARITY_PRECHECK.extend(
        [
            r1["linearity_error"],
            r2["linearity_error"],
            r3["linearity_error"],
            rb["linearity_error"]
        ]
    )


    print(
        "    |Z(0.10)| =",
        abs(r1["Z"])
    )

    print(
        "    |Z(0.05)| =",
        abs(r2["Z"])
    )

    print(
        "    |Z(0.025)| =",
        abs(r3["Z"])
    )


Z_1 = np.asarray(Z_1)
Z_2 = np.asarray(Z_2)
Z_3 = np.asarray(Z_3)
Z_BOUND = np.asarray(Z_BOUND)


# ============================================================
# 20. Convergence metrics
# ============================================================

ERROR_1_TO_2 = (
    np.linalg.norm(
        Z_1
        -
        Z_2
    )
    /
    max(
        np.linalg.norm(Z_2),
        1e-300
    )
)


ERROR_2_TO_3 = (
    np.linalg.norm(
        Z_2
        -
        Z_3
    )
    /
    max(
        np.linalg.norm(Z_3),
        1e-300
    )
)


BOUNDARY_ERROR = (
    np.linalg.norm(
        Z_BOUND
        -
        Z_2
    )
    /
    max(
        np.linalg.norm(Z_BOUND),
        1e-300
    )
)


COMMON_SCALE = max(
    np.max(
        np.abs(Z_3)
    ),
    1e-300
)


POINT_ERROR_12 = (
    np.abs(
        Z_1
        -
        Z_2
    )
    /
    COMMON_SCALE
)


POINT_ERROR_23 = (
    np.abs(
        Z_2
        -
        Z_3
    )
    /
    COMMON_SCALE
)


print()
print("=" * 80)
print("PRECHECK SUMMARY")
print("=" * 80)


print(
    f"{'offset':>10}"
    f"{'0.10->0.05':>20}"
    f"{'0.05->0.025':>20}"
)


for offset, e12, e23 in zip(
    PRECHECK_OFFSETS_MHZ,
    POINT_ERROR_12,
    POINT_ERROR_23
):

    print(
        f"{offset:10.3f}"
        f"{e12:20.8e}"
        f"{e23:20.8e}"
    )


print()

print(
    "vector dq_local 0.10 -> 0.05 error =",
    ERROR_1_TO_2
)

print(
    "vector dq_local 0.05 -> 0.025 error =",
    ERROR_2_TO_3
)

print(
    "vector +/-4u -> +/-5u error =",
    BOUNDARY_ERROR
)

print(
    "max precheck heterodyne-linearity error =",
    np.max(
        LINEARITY_PRECHECK
    )
)


GRID_PASS = (
    ERROR_2_TO_3
    <
    GRID_ERROR_LIMIT
)


BOUNDARY_PASS = (
    BOUNDARY_ERROR
    <
    BOUNDARY_ERROR_LIMIT
)


if GRID_PASS:

    print(
        "PASS: local velocity-grid convergence."
    )

else:

    print(
        "FAIL: local velocity-grid convergence."
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
        "\n14c thermal calculation aborted.\n"
        "Velocity integration is not yet converged.\n\n"
        "Do NOT interpret thermal lock-point values.\n"
    )


# ============================================================
# 21. Frequency grid
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


DELTA_SCAN = (
    (
        LOCK_CENTER_MHZ
        +
        LOCAL_OFFSET_MHZ
    )
    /
    engine.GAMMA_MHZ
)


# ============================================================
# 22. No-Doppler spectrum
# ============================================================

print()
print("=" * 80)

print(
    "Calculating no-Doppler spectrum"
)

print("=" * 80)


P_PLUS_0 = np.zeros(
    len(DELTA_SCAN),
    dtype=complex
)

P_MINUS_0 = np.zeros_like(
    P_PLUS_0
)

Z_0 = np.zeros_like(
    P_PLUS_0
)


for i, delta in enumerate(DELTA_SCAN):

    r = engine.response_at_detuning(
        delta=delta,
        kv=0.0
    )

    P_PLUS_0[i] = r["P_plus"]
    P_MINUS_0[i] = r["P_minus"]
    Z_0[i] = r["Z"]


# ============================================================
# 23. Thermal production spectrum
#
# dq_local = 0.05
# ============================================================

print()
print("=" * 80)

print(
    "Calculating FINAL composite-grid thermal spectrum"
)

print("=" * 80)


P_PLUS_T = np.zeros_like(
    P_PLUS_0
)

P_MINUS_T = np.zeros_like(
    P_PLUS_0
)

Z_T = np.zeros_like(
    P_PLUS_0
)

LINEARITY_ERRORS = []


for i, delta in enumerate(
    DELTA_SCAN
):

    print()
    print(
        f"{i+1:3d}/{len(DELTA_SCAN)}"
        f"   offset="
        f"{LOCAL_OFFSET_MHZ[i]:+7.3f} MHz"
    )

    r = thermal_main(
        delta,
        verbose=True
    )

    P_PLUS_T[i] = r["P_plus"]
    P_MINUS_T[i] = r["P_minus"]
    Z_T[i] = r["Z"]

    LINEARITY_ERRORS.append(
        r["linearity_error"]
    )


LINEARITY_ERRORS = np.asarray(
    LINEARITY_ERRORS
)


# ============================================================
# 24. Analysis functions
# ============================================================

def complex_center_slope(
    x_mhz,
    Z,
    window_mhz=2.0
):

    mask = (
        np.abs(x_mhz)
        <=
        window_mhz
    )

    x = (
        x_mhz[mask]
        /
        engine.GAMMA_MHZ
    )

    z = Z[mask]


    cr = np.polyfit(
        x,
        np.real(z),
        3
    )

    ci = np.polyfit(
        x,
        np.imag(z),
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


def zero_from_curve(
    x,
    y,
    window=3.0
):

    mask = (
        np.abs(x)
        <=
        window
    )

    xx = x[mask]
    yy = y[mask]

    order = np.argsort(xx)

    xx = xx[order]
    yy = yy[order]


    interp = PchipInterpolator(
        xx,
        yy
    )


    roots = []


    for i in range(
        len(xx) - 1
    ):

        if yy[i] == 0:

            roots.append(
                xx[i]
            )

        elif (
            yy[i]
            *
            yy[i + 1]
            <
            0
        ):

            roots.append(
                brentq(
                    interp,
                    xx[i],
                    xx[i + 1]
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

    order = np.argsort(x)

    xx = x[order]
    yy = y[order]


    interp = PchipInterpolator(
        xx,
        yy
    )


    dense = np.linspace(
        xx.min(),
        xx.max(),
        30001
    )


    yd = interp(dense)


    prominence = (
        0.01
        *
        (
            yd.max()
            -
            yd.min()
        )
    )


    maxima, _ = find_peaks(
        yd,
        prominence=prominence
    )

    minima, _ = find_peaks(
        -yd,
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
        dense[ext]
        <
        zero
    ]

    right = ext[
        dense[ext]
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
            np.nan
        )


    il = left[-1]
    ir = right[0]


    Vpp = abs(
        yd[ir]
        -
        yd[il]
    )

    separation = (
        dense[ir]
        -
        dense[il]
    )


    return (
        float(Vpp),
        float(separation)
    )


# ============================================================
# 25. Main thermal discriminator
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


ZERO_T_MAIN = zero_from_curve(
    LOCAL_OFFSET_MHZ,
    S_T_MAIN
)


# ============================================================
# 26. Refined final phase and lock
#
# dq_local = 0.025
# ============================================================

if RUN_REFINED_LOCK:

    print()
    print("=" * 80)

    print(
        "Refining final thermal phase/lock "
        "with dq_local = 0.025"
    )

    print("=" * 80)


    Z_PHASE = []


    for offset in REFINED_PHASE_OFFSETS_MHZ:

        delta = (
            (
                LOCK_CENTER_MHZ
                +
                offset
            )
            /
            engine.GAMMA_MHZ
        )

        print(
            "phase point:",
            offset,
            "MHz"
        )

        r = thermal_refined(
            delta,
            verbose=True
        )

        Z_PHASE.append(
            r["Z"]
        )


    Z_PHASE = np.asarray(
        Z_PHASE
    )


    x_phase = (
        REFINED_PHASE_OFFSETS_MHZ
        /
        engine.GAMMA_MHZ
    )


    cr = np.polyfit(
        x_phase,
        np.real(Z_PHASE),
        3
    )

    ci = np.polyfit(
        x_phase,
        np.imag(Z_PHASE),
        3
    )


    SLOPE_T_FINAL = (
        cr[-2]
        +
        1j
        *
        ci[-2]
    )


    PHI_T_FINAL = np.angle(
        SLOPE_T_FINAL
    )


    # Use main curve at final phase to locate root approximately.
    S_MAIN_FINAL_PHASE = demodulate(
        Z_T,
        PHI_T_FINAL
    )


    ZERO_APPROX = zero_from_curve(
        LOCAL_OFFSET_MHZ,
        S_MAIN_FINAL_PHASE
    )


    if not np.isfinite(
        ZERO_APPROX
    ):

        raise RuntimeError(
            "No thermal zero crossing was found "
            "after converged thermal averaging."
        )


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


    Z_LOCK = np.zeros(
        len(lock_offsets),
        dtype=complex
    )


    for i, offset in enumerate(
        lock_offsets
    ):

        print(
            f"refined lock "
            f"{i+1}/{len(lock_offsets)}: "
            f"{offset:+.4f} MHz"
        )

        delta = (
            (
                LOCK_CENTER_MHZ
                +
                offset
            )
            /
            engine.GAMMA_MHZ
        )


        Z_LOCK[i] = thermal_refined(
            delta,
            verbose=True
        )["Z"]


    S_LOCK = demodulate(
        Z_LOCK,
        PHI_T_FINAL
    )


    ZERO_T_FINAL = zero_from_curve(
        lock_offsets,
        S_LOCK,
        window=max(
            np.abs(lock_offsets)
        )
        +
        0.1
    )


else:

    SLOPE_T_FINAL = SLOPE_T_MAIN
    PHI_T_FINAL = PHI_T_MAIN
    ZERO_T_FINAL = ZERO_T_MAIN


# ============================================================
# 27. Final thermal signal
# ============================================================

S_T_FINAL = demodulate(
    Z_T,
    PHI_T_FINAL
)


VPP_T, SEP_T = extrema_metrics(
    LOCAL_OFFSET_MHZ,
    S_T_FINAL,
    ZERO_T_FINAL
)


# ============================================================
# 28. No-Doppler comparison
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

ZERO_0_OPT = zero_from_curve(
    LOCAL_OFFSET_MHZ,
    S_0_OPT
)


S_0_THERMAL_PHASE = demodulate(
    Z_0,
    PHI_T_FINAL
)


ZERO_0_THERMAL_PHASE = zero_from_curve(
    LOCAL_OFFSET_MHZ,
    S_0_THERMAL_PHASE
)


NO_DOPPLER_PROJECTED_SLOPE = abs(
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


# ============================================================
# 29. beta=0 regression
# ============================================================

TEST_OFFSET = (
    ZERO_T_FINAL
    +
    2.0
    if np.isfinite(ZERO_T_FINAL)
    else 2.0
)


TEST_DELTA = (
    (
        LOCK_CENTER_MHZ
        +
        TEST_OFFSET
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
        NULL_TEST["Z"]
    )
    /
    max(
        abs(
            NORMAL_TEST["Z"]
        ),
        1e-300
    )
)


# ============================================================
# 30. Velocity profile at final lock
# ============================================================

PROFILE_OFFSET = (
    ZERO_T_FINAL
    if np.isfinite(
        ZERO_T_FINAL
    )
    else 0.0
)


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
    return_profile=True,
    verbose=True
)


# ============================================================
# 31. Final result
# ============================================================

print()
print("=" * 80)

print(
    "FINAL CONVERGED THERMAL MTS RESULT"
)

print("=" * 80)


print()
print("Model conditions:")

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
    "background dq =",
    DQ_BACKGROUND
)

print(
    "production local dq =",
    DQ_LOCAL_2
)

print(
    "refined local dq =",
    DQ_LOCAL_3
)


print()
print("Thermal discriminator:")

print(
    "complex center slope =",
    SLOPE_T_FINAL
)

print(
    "|center slope| =",
    abs(
        SLOPE_T_FINAL
    )
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
    ZERO_T_FINAL
    *
    1000.0
    if np.isfinite(
        ZERO_T_FINAL
    )
    else np.nan,
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
print("No-Doppler reference:")

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
    "lock shift at thermal phase =",
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
print("Thermal / no-Doppler:")

print(
    "thermal slope =",
    abs(
        SLOPE_T_FINAL
    )
)

print(
    "no-Doppler projected slope =",
    NO_DOPPLER_PROJECTED_SLOPE
)

print(
    "slope ratio =",
    abs(
        SLOPE_T_FINAL
    )
    /
    max(
        NO_DOPPLER_PROJECTED_SLOPE,
        1e-300
    )
)


print()
print("Numerical validation:")

print(
    "0.10 -> 0.05 error =",
    ERROR_1_TO_2
)

print(
    "0.05 -> 0.025 error =",
    ERROR_2_TO_3
)

print(
    "+/-4u -> +/-5u error =",
    BOUNDARY_ERROR
)

print(
    "max heterodyne-linearity error =",
    np.max(
        LINEARITY_ERRORS
    )
)

print(
    "beta=0 thermal null ratio =",
    NULL_RATIO
)


ALL_FINITE = (
    np.all(
        np.isfinite(
            Z_T
        )
    )
    and
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
)


print(
    "All finite =",
    ALL_FINITE
)


print()
print("=" * 80)
print("FINAL VALIDATION")
print("=" * 80)


print(
    "PASS: local velocity-grid convergence."
    if GRID_PASS
    else
    "FAIL: local velocity-grid convergence."
)

print(
    "PASS: Maxwell boundary convergence."
    if BOUNDARY_PASS
    else
    "FAIL: Maxwell boundary convergence."
)

print(
    "PASS: beta=0 thermal null test."
    if NULL_RATIO < 1e-3
    else
    "FAIL: beta=0 thermal null test."
)

print(
    "PASS: finite thermal lock zero."
    if np.isfinite(
        ZERO_T_FINAL
    )
    else
    "FAIL: thermal lock zero."
)

print(
    "PASS: all thermal arrays finite."
    if ALL_FINITE
    else
    "FAIL: non-finite thermal array."
)


# ============================================================
# 32. Figure 1
# Fixed composite velocity grid
# ============================================================

plt.figure(
    figsize=(9, 5)
)


q_plot = np.linspace(
    -4 * DOPPLER_D,
    +4 * DOPPLER_D,
    3001
)


plt.plot(
    q_plot,
    maxwell_weight_q(q_plot),
    label="Maxwell distribution"
)


for center in FAMILY_CENTERS:

    plt.axvline(
        center,
        linestyle="--",
        linewidth=0.7
    )


for left, right in FINE_INTERVALS:

    plt.axvspan(
        left,
        right,
        alpha=0.08
    )


plt.xlabel(
    r"$q=kv/\Gamma$"
)

plt.ylabel(
    "Maxwell probability density"
)

plt.title(
    "Fixed fine-velocity windows used for all laser detunings"
)

plt.grid(alpha=0.3)

plt.tight_layout()


# ============================================================
# Figure 2
# Sidebands
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
    figsize=(9, 5)
)


plt.plot(
    LOCAL_OFFSET_MHZ,
    np.abs(P_PLUS_0) / SIDE_SCALE,
    label=r"No Doppler $\omega+\Omega_m$"
)

plt.plot(
    LOCAL_OFFSET_MHZ,
    np.abs(P_MINUS_0) / SIDE_SCALE,
    label=r"No Doppler $\omega-\Omega_m$"
)

plt.plot(
    LOCAL_OFFSET_MHZ,
    np.abs(P_PLUS_T) / SIDE_SCALE,
    "--",
    label=r"Thermal $\omega+\Omega_m$"
)

plt.plot(
    LOCAL_OFFSET_MHZ,
    np.abs(P_MINUS_T) / SIDE_SCALE,
    "--",
    label=r"Thermal $\omega-\Omega_m$"
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
    "Third-order polarization "
    "(common normalization)"
)

plt.title(
    "Converged thermal averaging of generated MTS sidebands"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()


# ============================================================
# Figure 3
# Complex heterodyne
# ============================================================

Z_SCALE = max(
    np.max(
        np.abs(
            Z_0
        )
    ),
    np.max(
        np.abs(
            Z_T
        )
    ),
    1e-300
)


plt.figure(
    figsize=(9, 5)
)


plt.plot(
    LOCAL_OFFSET_MHZ,
    np.real(Z_0) / Z_SCALE,
    label="No Doppler Re(Z)"
)

plt.plot(
    LOCAL_OFFSET_MHZ,
    np.imag(Z_0) / Z_SCALE,
    label="No Doppler Im(Z)"
)

plt.plot(
    LOCAL_OFFSET_MHZ,
    np.real(Z_T) / Z_SCALE,
    "--",
    label="Thermal Re(Z)"
)

plt.plot(
    LOCAL_OFFSET_MHZ,
    np.imag(Z_T) / Z_SCALE,
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
    r"Detuning from $F=2\rightarrow F'=3$ (MHz)"
)

plt.ylabel(
    "Complex RF response"
)

plt.title(
    "Converged thermal Rb87 MTS heterodyne response"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()


# ============================================================
# Figure 4
# Final discriminator
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
            S_T_FINAL
        )
    ),
    1e-300
)


plt.figure(
    figsize=(9, 5)
)


plt.plot(
    LOCAL_OFFSET_MHZ,
    S_0_FINAL_PHASE / SIGNAL_SCALE,
    label="No Doppler"
)

plt.plot(
    LOCAL_OFFSET_MHZ,
    S_T_FINAL / SIGNAL_SCALE,
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
    label="Nominal F'=3"
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
            f"{ZERO_T_FINAL*1000:.1f} kHz"
        )
    )


plt.xlabel(
    r"Detuning from $F=2\rightarrow F'=3$ (MHz)"
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

plt.grid(alpha=0.3)

plt.tight_layout()


# ============================================================
# Figure 5
# Complex velocity integrand
# ============================================================

q_profile = PROFILE[
    "q"
]

Z_integrand = PROFILE[
    "Z_integrand"
]


INT_SCALE = max(
    np.max(
        np.abs(
            Z_integrand
        )
    ),
    1e-300
)


plt.figure(
    figsize=(9, 5)
)


plt.plot(
    q_profile,
    np.real(Z_integrand) / INT_SCALE,
    label=r"$\Re[f(q)Z(q)]$"
)

plt.plot(
    q_profile,
    np.imag(Z_integrand) / INT_SCALE,
    label=r"$\Im[f(q)Z(q)]$"
)


plt.axhline(
    0,
    linewidth=0.8
)


plt.xlabel(
    r"$q=kv/\Gamma$"
)

plt.ylabel(
    "Velocity-class contribution"
)

plt.title(
    "Coherent velocity-class contributions at final lock"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()


# ============================================================
# Figure 6
# Cumulative complex integral
# ============================================================

cum = np.zeros(
    len(q_profile),
    dtype=complex
)


for i in range(
    1,
    len(q_profile)
):

    dq = (
        q_profile[i]
        -
        q_profile[i - 1]
    )

    cum[i] = (
        cum[i - 1]
        +
        0.5
        *
        (
            Z_integrand[i - 1]
            +
            Z_integrand[i]
        )
        *
        dq
    )


CUM_SCALE = max(
    np.max(
        np.abs(cum)
    ),
    1e-300
)


plt.figure(
    figsize=(9, 5)
)


plt.plot(
    q_profile,
    np.real(cum) / CUM_SCALE,
    label="Cumulative Re"
)

plt.plot(
    q_profile,
    np.imag(cum) / CUM_SCALE,
    label="Cumulative Im"
)


plt.xlabel(
    r"Upper integration limit $q$"
)

plt.ylabel(
    "Cumulative thermal response"
)

plt.title(
    "Coherent cancellation in the thermal velocity integral"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()


plt.show()