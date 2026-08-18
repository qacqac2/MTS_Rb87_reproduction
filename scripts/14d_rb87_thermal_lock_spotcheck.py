"""
14d_rb87_thermal_lock_spotcheck.py

FINAL LOCAL LOCK-POINT SPOT CHECK

Purpose
-------
Stage 14c already passed:

    velocity-grid convergence
    Maxwell-boundary convergence
    heterodyne-linearity validation
    beta=0 null test

This script does NOT repeat the full thermal spectrum.

It keeps the converged velocity grid:

    background dq = 0.5
    local dq      = 0.025

and checks only the final thermal lock point using laser-frequency
steps:

    0.25 MHz
    0.125 MHz
    0.0625 MHz

around the Stage-14c result:

    approximately -1.618 MHz

The demodulation phase is fixed to the FINAL Stage-14c value:

    phi = 39.1846916 deg

because this test is intended to isolate FREQUENCY-GRID convergence.

There is NO FAST MODE.
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.sparse import csc_matrix


# ============================================================
# 1. Physical settings: MUST match Stage 14c
# ============================================================

TEMPERATURE_K = 300.0
GAMMA_TRANSIT = 0.020

LOCK_FE = 3

# Fixed Stage-14c final mixer phase.
PHI_THERMAL_DEG = 39.1846916161284
PHI_THERMAL = np.deg2rad(PHI_THERMAL_DEG)


# ============================================================
# 2. Converged velocity grid from Stage 14c
# ============================================================

VELOCITY_SPAN_U = 4.0

DQ_BACKGROUND = 0.50
DQ_LOCAL = 0.025

FULL_SPAN_MHZ = 12.0
MAX_MODULATION_ORDER = 3
RESONANCE_MARGIN_Q = 2.0


# ============================================================
# 3. LOCAL frequency spot check
# ============================================================

# Covers the Stage-14c lock at -1.6177 MHz.
FREQ_START_MHZ = -2.000
FREQ_STOP_MHZ = -1.250

# Finest grid calculated explicitly.
FREQ_FINE_STEP_MHZ = 0.0625

# Acceptance targets.
#
# 0.25 -> 0.125:
#     should preferably change by < 25 kHz
#
# 0.125 -> 0.0625:
#     final target < 10 kHz
#
LOCK_CHANGE_025_TO_0125_LIMIT_KHZ = 25.0
LOCK_CHANGE_0125_TO_00625_LIMIT_KHZ = 10.0


# ============================================================
# 4. Constants
# ============================================================

KB = 1.380649e-23
AMU = 1.66053906660e-27

MASS_RB87 = 86.9091805 * AMU

LAMBDA_D2 = 780.24e-9


# ============================================================
# 5. Load Stage-13 engine
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
        "Place 14d and 13_rb87_hyperfine_zeeman_mts.py "
        "in the same directory."
    )


MODULE_NAME = "rb87_engine_14d"

spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    ENGINE_PATH
)

engine = importlib.util.module_from_spec(spec)

sys.modules[MODULE_NAME] = engine

spec.loader.exec_module(engine)


# ============================================================
# 6. Rebuild transit dissipator exactly as 13b / 14c
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

            L += engine.lindblad_superoperator(C)

    return L


L_DISS_SELECTED = make_transit_dissipator(
    GAMMA_TRANSIT
)


def build_L0_selected(delta):

    energies = np.zeros(
        engine.N,
        dtype=float
    )

    # Ground states
    for Fg in engine.FG_LIST:

        if Fg == 2:

            hfs_energy = 0.0

        else:

            hfs_energy = (
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
                hfs_energy
                +
                engine.GF_GROUND[Fg]
                *
                engine.ZEEMAN_UNIT
                *
                mg
            )

    # Excited states
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
# 7. Thermal distribution
# ============================================================

U_THERMAL = np.sqrt(
    2.0
    *
    KB
    *
    TEMPERATURE_K
    /
    MASS_RB87
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
# 8. Same fixed composite grid as Stage 14c
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


FAMILY_CENTERS = []

for Fe in engine.FE_LIST:

    q0 = abs(
        engine.HF_OFFSET[Fe]
    )

    FAMILY_CENTERS.extend(
        [
            -q0,
            +q0
        ]
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
            center - FINE_HALF_WIDTH_Q,
            center + FINE_HALF_WIDTH_Q
        )
        for center in FAMILY_CENTERS
    ]
)


def symmetric_grid(dq, span_u):

    qmax = (
        span_u
        *
        DOPPLER_D
    )

    N = int(
        np.floor(
            qmax / dq
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
        len(q),
        dtype=bool
    )

    for left, right in FINE_INTERVALS:

        mask |= (
            (q >= left)
            &
            (q <= right)
        )

    return mask


def build_composite_grid():

    background = symmetric_grid(
        DQ_BACKGROUND,
        VELOCITY_SPAN_U
    )

    fine_full = symmetric_grid(
        DQ_LOCAL,
        VELOCITY_SPAN_U
    )

    fine = fine_full[
        inside_fine_intervals(
            fine_full
        )
    ]

    return np.unique(
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


Q = build_composite_grid()

W = maxwell_weight_q(Q)

MAXWELL_MASS = np.trapezoid(
    W,
    Q
)


# ============================================================
# 9. Atomic-response cache
# ============================================================

ATOMIC_CACHE = {}


def cached_atomic_response(delta, kv):

    key = (
        round(float(delta), 12),
        round(float(kv), 12)
    )

    if key not in ATOMIC_CACHE:

        ATOMIC_CACHE[key] = (
            engine.response_at_detuning(
                delta=delta,
                kv=kv,
                beta=engine.BETA,
                field_scale=1.0
            )
        )

    return ATOMIC_CACHE[key]


# ============================================================
# 10. Heterodyne
# ============================================================

def heterodyne(
    P_plus,
    P_minus
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

    E_c = engine.OMEGA_PROBE

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
# 11. Thermal response
# ============================================================

def thermal_response(delta, verbose=True):

    P_plus_v = np.empty(
        len(Q),
        dtype=complex
    )

    P_minus_v = np.empty_like(
        P_plus_v
    )

    for i, kv in enumerate(Q):

        if (
            verbose
            and
            (
                i == 0
                or
                (i + 1) % 500 == 0
                or
                i == len(Q) - 1
            )
        ):

            print(
                f"        velocity "
                f"{i+1}/{len(Q)}"
            )

        r = cached_atomic_response(
            delta,
            kv
        )

        P_plus_v[i] = r["P_plus"]
        P_minus_v[i] = r["P_minus"]


    P_plus = np.trapezoid(
        W * P_plus_v,
        Q
    )

    P_minus = np.trapezoid(
        W * P_minus_v,
        Q
    )


    return heterodyne(
        P_plus,
        P_minus
    )


# ============================================================
# 12. Demodulation
# ============================================================

def demodulate(Z):

    return np.real(
        Z
        *
        np.exp(
            -1j
            *
            PHI_THERMAL
        )
    )


# ============================================================
# 13. Root extraction
# ============================================================

def root_from_samples(x, y):

    order = np.argsort(x)

    x = np.asarray(x)[order]
    y = np.asarray(y)[order]

    interp = PchipInterpolator(
        x,
        y
    )

    roots = []

    for i in range(
        len(x) - 1
    ):

        if y[i] == 0:

            roots.append(
                x[i]
            )

        elif (
            y[i]
            *
            y[i + 1]
            <
            0
        ):

            roots.append(
                brentq(
                    interp,
                    x[i],
                    x[i + 1]
                )
            )

    if not roots:

        return np.nan

    # We know the physical target is close to the Stage-14c root.
    return min(
        roots,
        key=lambda r:
        abs(
            r + 1.6177
        )
    )


# ============================================================
# 14. Frequency grids
#
# Calculate ONLY finest grid.
# Coarser grids are strict subsets.
# ============================================================

FREQ_FINE = np.arange(
    FREQ_START_MHZ,
    FREQ_STOP_MHZ
    +
    0.5
    *
    FREQ_FINE_STEP_MHZ,
    FREQ_FINE_STEP_MHZ
)


# Every second fine point:
FREQ_0125 = FREQ_FINE[::2]

# Every fourth fine point:
FREQ_025 = FREQ_FINE[::4]


# ============================================================
# 15. Header
# ============================================================

print()
print("=" * 80)

print(
    "Stage F3d FINAL: thermal lock-point frequency-grid spot check"
)

print("=" * 80)


print()
print(
    "Physical conditions:"
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
    "fm =",
    engine.FM_MHZ,
    "MHz"
)

print(
    "beta =",
    engine.BETA
)

print(
    "fixed thermal demodulation phase =",
    PHI_THERMAL_DEG,
    "deg"
)


print()
print(
    "Velocity integration:"
)

print(
    "background dq =",
    DQ_BACKGROUND
)

print(
    "local dq =",
    DQ_LOCAL
)

print(
    "velocity points =",
    len(Q)
)

print(
    "captured Maxwell mass =",
    MAXWELL_MASS
)

print(
    "rho0 residual =",
    RHO0_RESIDUAL
)


print()
print(
    "Laser-frequency spot-check interval:"
)

print(
    FREQ_START_MHZ,
    "to",
    FREQ_STOP_MHZ,
    "MHz"
)

print(
    "finest step =",
    FREQ_FINE_STEP_MHZ,
    "MHz"
)


# ============================================================
# 16. Calculate finest frequency scan
# ============================================================

Z_FINE = np.zeros(
    len(FREQ_FINE),
    dtype=complex
)


print()
print("=" * 80)

print(
    "Calculating finest local frequency grid"
)

print("=" * 80)


for i, offset in enumerate(FREQ_FINE):

    print()
    print(
        f"frequency "
        f"{i+1}/{len(FREQ_FINE)}"
        f"   offset={offset:+.6f} MHz"
    )

    delta = (
        offset
        /
        engine.GAMMA_MHZ
    )

    Z_FINE[i] = thermal_response(
        delta
    )


S_FINE = demodulate(
    Z_FINE
)


# ============================================================
# 17. Extract coarser subsets
# ============================================================

Z_0125 = Z_FINE[::2]
S_0125 = S_FINE[::2]

Z_025 = Z_FINE[::4]
S_025 = S_FINE[::4]


ZERO_025 = root_from_samples(
    FREQ_025,
    S_025
)

ZERO_0125 = root_from_samples(
    FREQ_0125,
    S_0125
)

ZERO_00625 = root_from_samples(
    FREQ_FINE,
    S_FINE
)


# ============================================================
# 18. Convergence
# ============================================================

CHANGE_025_TO_0125_KHZ = (
    abs(
        ZERO_0125
        -
        ZERO_025
    )
    *
    1000.0
)


CHANGE_0125_TO_00625_KHZ = (
    abs(
        ZERO_00625
        -
        ZERO_0125
    )
    *
    1000.0
)


PASS_025_TO_0125 = (
    CHANGE_025_TO_0125_KHZ
    <
    LOCK_CHANGE_025_TO_0125_LIMIT_KHZ
)


PASS_FINAL = (
    CHANGE_0125_TO_00625_KHZ
    <
    LOCK_CHANGE_0125_TO_00625_LIMIT_KHZ
)


# ============================================================
# 19. Local numerical slope at final root
#
# This is NOT used to redefine mixer phase.
# It is just an additional local diagnostic.
# ============================================================

interp_final = PchipInterpolator(
    FREQ_FINE,
    S_FINE
)


LOCAL_SLOPE_PER_MHZ = (
    interp_final.derivative()(
        ZERO_00625
    )
)


# ============================================================
# 20. Results
# ============================================================

print()
print("=" * 80)

print(
    "FINAL LOCK-POINT SPOT-CHECK RESULT"
)

print("=" * 80)


print()
print(
    "0.250 MHz grid lock =",
    ZERO_025,
    "MHz"
)

print(
    "0.125 MHz grid lock =",
    ZERO_0125,
    "MHz"
)

print(
    "0.0625 MHz grid lock =",
    ZERO_00625,
    "MHz"
)


print()
print(
    "0.250 -> 0.125 MHz change =",
    CHANGE_025_TO_0125_KHZ,
    "kHz"
)

print(
    "0.125 -> 0.0625 MHz change =",
    CHANGE_0125_TO_00625_KHZ,
    "kHz"
)


print()
print(
    "FINAL recommended thermal lock shift =",
    ZERO_00625,
    "MHz"
)

print(
    "FINAL recommended thermal lock shift =",
    ZERO_00625
    *
    1000.0,
    "kHz"
)


print()
print(
    "local demodulated slope =",
    LOCAL_SLOPE_PER_MHZ,
    "signal units / MHz"
)


print()
print("=" * 80)

print(
    "FINAL SPOT-CHECK VALIDATION"
)

print("=" * 80)


if PASS_025_TO_0125:

    print(
        "PASS: 0.250 -> 0.125 MHz lock convergence."
    )

else:

    print(
        "CHECK: 0.250 -> 0.125 MHz change exceeds 25 kHz."
    )


if PASS_FINAL:

    print(
        "PASS: 0.125 -> 0.0625 MHz lock convergence < 10 kHz."
    )

else:

    print(
        "CHECK: final frequency-grid convergence exceeds 10 kHz."
    )


if (
    PASS_025_TO_0125
    and
    PASS_FINAL
):

    print()
    print(
        "FINAL RESULT ACCEPTED."
    )

    print(
        "The thermal lock point is numerically converged "
        "with respect to both velocity and laser-frequency grids."
    )

else:

    print()
    print(
        "Do not freeze the final lock value yet."
    )


# ============================================================
# 21. Figure 1
#
# Three frequency-grid resolutions
# ============================================================

SIGNAL_SCALE = max(
    np.max(
        np.abs(
            S_FINE
        )
    ),
    1e-300
)


plt.figure(
    figsize=(9, 5)
)


plt.plot(
    FREQ_FINE,
    S_FINE / SIGNAL_SCALE,
    "-",
    label="0.0625 MHz"
)


plt.plot(
    FREQ_0125,
    S_0125 / SIGNAL_SCALE,
    "o",
    label="0.125 MHz"
)


plt.plot(
    FREQ_025,
    S_025 / SIGNAL_SCALE,
    "s",
    label="0.250 MHz"
)


plt.axhline(
    0,
    linewidth=0.8
)


plt.axvline(
    ZERO_00625,
    linestyle=":",
    linewidth=1.0,
    label=(
        "final root = "
        f"{ZERO_00625*1000:.1f} kHz"
    )
)


plt.xlabel(
    r"Detuning from $F=2\rightarrow F'=3$ (MHz)"
)

plt.ylabel(
    "Thermal MTS error signal "
    "(self-normalized)"
)

plt.title(
    "Final thermal MTS lock-point frequency-grid convergence"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()


# ============================================================
# Figure 2
#
# Lock point versus frequency step
# ============================================================

steps = np.array(
    [
        0.250,
        0.125,
        0.0625
    ]
)


roots = np.array(
    [
        ZERO_025,
        ZERO_0125,
        ZERO_00625
    ]
)


plt.figure(
    figsize=(7, 5)
)


plt.plot(
    steps,
    roots * 1000.0,
    "o-"
)


plt.gca().invert_xaxis()


plt.xlabel(
    "Laser-frequency grid step (MHz)"
)

plt.ylabel(
    "Extracted thermal lock shift (kHz)"
)

plt.title(
    "Convergence of final thermal MTS lock point"
)

plt.grid(alpha=0.3)

plt.tight_layout()


plt.show()