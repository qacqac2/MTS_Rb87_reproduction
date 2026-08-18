"""
13b_rb87_hyperfine_refine.py

Stage F2+:
Fine-grid convergence + ground-transit sensitivity
for the validated Stage-13 87Rb hyperfine/Zeeman rho^(3) engine.

This script DOES NOT rebuild the atomic model.

It imports:

    13_rb87_hyperfine_zeeman_mts.py

and tests:

1. local frequency-grid convergence
       0.50 MHz -> 0.25 MHz

2. sensitivity to ground-state transit relaxation
       Gamma_transit/Gamma =
       0.005, 0.010, 0.020, 0.050

Main observables:

    lock-point shift
    complex discriminator slope
    optimum demodulation phase
    central peak-to-peak amplitude

The goal is to determine whether the sub-MHz lock shifts
reported by Stage 13 are numerically resolved and whether
they are sensitive to the ground-state reservoir model.
"""


import sys
import importlib.util

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import find_peaks

from scipy.sparse import (
    csc_matrix
)


# ============================================================
# 0. User settings
# ============================================================

FAST_MODE = False


# ------------------------------------------------------------
# Reference transit rate used in Stage 13
# ------------------------------------------------------------

REFERENCE_TRANSIT = 0.020


# ------------------------------------------------------------
# Grid-convergence test
#
# This is the most important part of this script.
# ------------------------------------------------------------

GRID_SPAN_MHZ = 6.0

COARSE_STEP_MHZ = 0.50

FINE_STEP_MHZ = 0.25


# ------------------------------------------------------------
# Transit sensitivity
#
# A hybrid grid is used:
#
# central region:
#     fine enough for zero crossing
#
# wings:
#     enough to capture nearby extrema
# ------------------------------------------------------------

TRANSIT_VALUES = np.array(
    [
        0.005,
        0.010,
        0.020,
        0.050
    ]
)


if FAST_MODE:

    TRANSIT_CENTER_STEP_MHZ = 0.50
    TRANSIT_WING_STEP_MHZ = 1.00

else:

    TRANSIT_CENTER_STEP_MHZ = 0.25
    TRANSIT_WING_STEP_MHZ = 0.50


TRANSIT_CENTER_SPAN_MHZ = 6.0

TRANSIT_FULL_SPAN_MHZ = 15.0


# ------------------------------------------------------------
# Local fitting / zero search
# ------------------------------------------------------------

SLOPE_FIT_WINDOW_MHZ = 4.0

ZERO_SEARCH_WINDOW_MHZ = 5.0


# ------------------------------------------------------------
# Numerical acceptance suggestions
# ------------------------------------------------------------

ZERO_GRID_TOL_MHZ = 0.10

SLOPE_GRID_TOL = 0.01


# ============================================================
# 1. Load Stage-13 engine
#
# Python cannot normally write
#
#     import 13_rb87_...
#
# so importlib is used.
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

        "Put this script and "
        "13_rb87_hyperfine_zeeman_mts.py "
        "in the same directory."
    )


MODULE_NAME = (
    "rb87_hyperfine_engine"
)


spec = (
    importlib.util.spec_from_file_location(

        MODULE_NAME,

        ENGINE_PATH
    )
)


engine = (
    importlib.util.module_from_spec(
        spec
    )
)


# Needed for dataclass/module metadata safety

sys.modules[
    MODULE_NAME
] = engine


spec.loader.exec_module(
    engine
)


# ============================================================
# 2. Basic engine checks
# ============================================================

print()
print(
    "=" * 76
)

print(
    "Stage F2+: hyperfine lock-point refinement "
    "+ transit sensitivity"
)

print(
    "=" * 76
)


print()
print(
    "Stage-13 engine:"
)

print(
    ENGINE_PATH
)


print()
print(
    "Hilbert dimension =",
    engine.N
)

print(
    "Liouville dimension =",
    engine.DIM
)


print()
print(
    "Hyperfine resonances:"
)


for Fe in engine.FE_LIST:

    print(

        f"F'= {Fe}: "

        f"{engine.HF_OFFSET_MHZ[Fe]:+.6f} MHz"
    )


# ============================================================
# 3. Build spontaneous-only dissipator
#
# Stage 13 stored spontaneous + transit together.
#
# For the transit scan we reconstruct the dissipator so that
# only Gamma_transit changes.
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


DISSIPATOR_CACHE = {}


def make_transit_dissipator(
    gamma_transit
):
    """
    Reproduce exactly the transit-mixing model
    used in Stage 13, but with adjustable rate.
    """

    if gamma_transit in DISSIPATOR_CACHE:

        return (
            DISSIPATOR_CACHE[
                gamma_transit
            ]
        )


    L = L_DISS_SPONT.copy()


    if (
        gamma_transit
        >
        0
    ):

        branch_rate = (

            gamma_transit

            /

            (
                len(
                    engine.GROUND_STATES
                )
                -
                1
            )
        )


        for state_from in (
            engine.GROUND_STATES
        ):

            for state_to in (
                engine.GROUND_STATES
            ):

                if (
                    state_to
                    ==
                    state_from
                ):

                    continue


                C = (
                    engine.sparse_operator(

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
                )


                L += (
                    engine.lindblad_superoperator(
                        C
                    )
                )


    DISSIPATOR_CACHE[
        gamma_transit
    ] = L


    return L


# ============================================================
# 4. Adjustable field-free Liouvillian
# ============================================================

def make_build_L0(
    gamma_transit
):
    """
    Return a build_L0(delta) function using the selected
    transit relaxation rate.
    """

    L_DISS = (
        make_transit_dissipator(
            gamma_transit
        )
    )


    def build_L0_custom(
        delta
    ):

        energies = np.zeros(
            engine.N,
            dtype=float
        )


        # ----------------------------------------------------
        # ground states
        # ----------------------------------------------------

        for Fg in engine.FG_LIST:

            if (
                Fg
                ==
                2
            ):

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


        # ----------------------------------------------------
        # excited states
        # ----------------------------------------------------

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

            L_DISS
        )


    return build_L0_custom


def install_transit_model(
    gamma_transit
):
    """
    Monkey-patch only Stage-13 build_L0.

    The validated rho1/rho2/rho3 recursion,
    optical operators and heterodyne code remain untouched.
    """

    engine.GAMMA_TRANSIT = (
        gamma_transit
    )


    engine.build_L0 = (
        make_build_L0(
            gamma_transit
        )
    )


# ============================================================
# 5. Verify rho0 remains stationary
# ============================================================

def rho0_residual(
    gamma_transit
):

    install_transit_model(
        gamma_transit
    )


    L0 = engine.build_L0(
        0.0
    )


    return np.linalg.norm(

        L0

        @

        engine.RHO0_VEC
    )


print()
print(
    "rho0 residual versus transit:"
)


for gamma_t in TRANSIT_VALUES:

    print(

        f"Gamma_t/Gamma = "
        f"{gamma_t:7.4f}"

        "   residual = "

        f"{rho0_residual(gamma_t):.6e}"
    )


# Return to the reference model

install_transit_model(
    REFERENCE_TRANSIT
)


# ============================================================
# 6. Local detuning grids
# ============================================================

def regular_offsets(
    span,
    step
):

    N = int(
        round(
            2.0
            *
            span
            /
            step
        )
    )


    return np.linspace(

        -span,

        +span,

        N + 1
    )


def transit_offsets():
    """
    Dense central region + coarser wings.
    """

    center = regular_offsets(

        TRANSIT_CENTER_SPAN_MHZ,

        TRANSIT_CENTER_STEP_MHZ
    )


    wings = regular_offsets(

        TRANSIT_FULL_SPAN_MHZ,

        TRANSIT_WING_STEP_MHZ
    )


    return np.unique(

        np.round(

            np.concatenate(
                [
                    center,
                    wings
                ]
            ),

            10
        )
    )


# ============================================================
# 7. Run one local hyperfine scan
# ============================================================

def local_scan(
    Fe,
    offsets_mhz,
    gamma_transit
):

    install_transit_model(
        gamma_transit
    )


    center_mhz = (
        engine.HF_OFFSET_MHZ[
            Fe
        ]
    )


    absolute_mhz = (

        center_mhz

        +

        offsets_mhz
    )


    normalized_detuning = (

        absolute_mhz

        /

        engine.GAMMA_MHZ
    )


    Z = np.zeros(

        len(
            offsets_mhz
        ),

        dtype=complex
    )


    for i, delta in enumerate(
        normalized_detuning
    ):

        result = (
            engine.response_at_detuning(
                delta
            )
        )


        Z[i] = (
            result[
                "Z"
            ]
        )


    return {

        "Fe":
        Fe,

        "gamma_transit":
        gamma_transit,

        "offset_mhz":
        offsets_mhz.copy(),

        "absolute_mhz":
        absolute_mhz,

        "delta_norm":
        normalized_detuning,

        "Z":
        Z
    }


# ============================================================
# 8. Local metrics
# ============================================================

def complex_local_slope(
    offsets_mhz,
    Z
):
    """
    Fit against normalized detuning Delta/Gamma,
    so slope units remain comparable to Stage 13.
    """

    mask = (

        np.abs(
            offsets_mhz
        )

        <=

        SLOPE_FIT_WINDOW_MHZ
    )


    x = (

        offsets_mhz[
            mask
        ]

        /

        engine.GAMMA_MHZ
    )


    zz = (
        Z[
            mask
        ]
    )


    if (
        len(x)
        <
        5
    ):

        return (
            np.nan
            +
            1j
            *
            np.nan
        )


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


def zero_crossing(
    offsets_mhz,
    signal
):
    """
    PCHIP interpolation on the actual local scan.

    Returns lock shift from nominal hyperfine resonance,
    in MHz.
    """

    mask = (

        np.abs(
            offsets_mhz
        )

        <=

        ZERO_SEARCH_WINDOW_MHZ
    )


    x = (
        offsets_mhz[
            mask
        ]
    )


    y = (
        signal[
            mask
        ]
    )


    order = np.argsort(
        x
    )


    x = x[
        order
    ]

    y = y[
        order
    ]


    interp = PchipInterpolator(
        x,
        y
    )


    dense = np.linspace(

        -ZERO_SEARCH_WINDOW_MHZ,

        +ZERO_SEARCH_WINDOW_MHZ,

        8001
    )


    yy = interp(
        dense
    )


    roots = []


    for i in range(
        len(dense) - 1
    ):

        if (
            yy[i]
            ==
            0
        ):

            roots.append(
                dense[i]
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

                    dense[i],

                    dense[
                        i + 1
                    ]
                )
            )


    if not roots:

        return np.nan


    return float(

        min(
            roots,
            key=abs
        )
    )


def nearest_extrema_metrics(
    offsets_mhz,
    signal,
    zero_shift
):

    order = np.argsort(
        offsets_mhz
    )


    x = offsets_mhz[
        order
    ]

    y = signal[
        order
    ]


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


    dynamic_range = (

        yy.max()

        -

        yy.min()
    )


    prominence = (

        0.01

        *
        dynamic_range
    )


    maxima, _ = find_peaks(

        yy,

        prominence=prominence
    )


    minima, _ = find_peaks(

        -yy,

        prominence=prominence
    )


    extrema = np.sort(

        np.concatenate(
            [
                maxima,
                minima
            ]
        )
    )


    left = extrema[

        dense[
            extrema
        ]

        <
        zero_shift
    ]


    right = extrema[

        dense[
            extrema
        ]

        >
        zero_shift
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

        yy[ir]

        -

        yy[il]
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


def analyze_scan(
    scan
):

    offsets = (
        scan[
            "offset_mhz"
        ]
    )


    Z = (
        scan[
            "Z"
        ]
    )


    slope = (
        complex_local_slope(
            offsets,
            Z
        )
    )


    phi = np.angle(
        slope
    )


    signal = np.real(

        Z

        *

        np.exp(
            -1j
            *
            phi
        )
    )


    zero_shift = (
        zero_crossing(
            offsets,
            signal
        )
    )


    Vpp, separation = (
        nearest_extrema_metrics(

            offsets,

            signal,

            (
                zero_shift

                if np.isfinite(
                    zero_shift
                )

                else 0.0
            )
        )
    )


    peak_abs = np.max(
        np.abs(
            signal
        )
    )


    return {

        **scan,

        "complex_slope":
        slope,

        "slope_mag":
        abs(
            slope
        ),

        "phi":
        phi,

        "signal":
        signal,

        "zero_shift_mhz":
        zero_shift,

        "Vpp":
        Vpp,

        "separation_mhz":
        separation,

        "peak_abs":
        peak_abs
    }


# ============================================================
# 9. Grid convergence
# ============================================================

print()
print(
    "=" * 76
)

print(
    "Part A: local-grid convergence"
)

print(
    "=" * 76
)


COARSE_OFFSETS = regular_offsets(

    GRID_SPAN_MHZ,

    COARSE_STEP_MHZ
)


FINE_OFFSETS = regular_offsets(

    GRID_SPAN_MHZ,

    FINE_STEP_MHZ
)


GRID_RESULTS = {}


for Fe in engine.FE_LIST:


    print()
    print(
        f"F'= {Fe}"
    )


    print(
        "  calculating 0.50 MHz grid..."
    )


    coarse = analyze_scan(

        local_scan(

            Fe,

            COARSE_OFFSETS,

            REFERENCE_TRANSIT
        )
    )


    print(
        "  calculating 0.25 MHz grid..."
    )


    fine = analyze_scan(

        local_scan(

            Fe,

            FINE_OFFSETS,

            REFERENCE_TRANSIT
        )
    )


    zero_difference = (

        fine[
            "zero_shift_mhz"
        ]

        -

        coarse[
            "zero_shift_mhz"
        ]
    )


    slope_relative_error = (

        abs(

            fine[
                "slope_mag"
            ]

            -

            coarse[
                "slope_mag"
            ]
        )

        /

        max(

            fine[
                "slope_mag"
            ],

            1e-300
        )
    )


    GRID_RESULTS[
        Fe
    ] = {

        "coarse":
        coarse,

        "fine":
        fine,

        "zero_difference_mhz":
        zero_difference,

        "slope_relative_error":
        slope_relative_error
    }


    print(
        "  coarse lock shift =",
        coarse[
            "zero_shift_mhz"
        ],
        "MHz"
    )


    print(
        "  fine   lock shift =",
        fine[
            "zero_shift_mhz"
        ],
        "MHz"
    )


    print(
        "  grid change =",
        zero_difference
        *
        1e3,
        "kHz"
    )


    print(
        "  slope relative change =",
        slope_relative_error
    )


# ============================================================
# 10. Grid-convergence summary
# ============================================================

print()
print(
    "=" * 76
)

print(
    "Grid-convergence summary"
)

print(
    "=" * 76
)


print(
    f"{'F':>4}"
    f"{'zero@0.5(MHz)':>18}"
    f"{'zero@0.25(MHz)':>19}"
    f"{'change(kHz)':>15}"
    f"{'slope change':>16}"
)


GRID_ALL_PASS = True


for Fe in engine.FE_LIST:


    r = (
        GRID_RESULTS[
            Fe
        ]
    )


    coarse_zero = (
        r[
            "coarse"
        ][
            "zero_shift_mhz"
        ]
    )


    fine_zero = (
        r[
            "fine"
        ][
            "zero_shift_mhz"
        ]
    )


    dz_khz = (

        r[
            "zero_difference_mhz"
        ]

        *
        1000.0
    )


    ds = (
        r[
            "slope_relative_error"
        ]
    )


    print(

        f"{Fe:4d}"

        f"{coarse_zero:18.6f}"

        f"{fine_zero:19.6f}"

        f"{dz_khz:15.3f}"

        f"{ds:16.6e}"
    )


    passed = (

        abs(
            r[
                "zero_difference_mhz"
            ]
        )

        <
        ZERO_GRID_TOL_MHZ

        and

        ds
        <
        SLOPE_GRID_TOL
    )


    GRID_ALL_PASS &= (
        passed
    )


print()


if GRID_ALL_PASS:

    print(
        "PASS: 0.50 -> 0.25 MHz local grid is converged "
        "at the current acceptance level."
    )

else:

    print(
        "CHECK: at least one hyperfine component is not yet "
        "fully converged at 0.25 MHz."
    )


# ============================================================
# 11. Transit-rate scan
# ============================================================

print()
print(
    "=" * 76
)

print(
    "Part B: transit-relaxation sensitivity"
)

print(
    "=" * 76
)


TRANSIT_OFFSETS = (
    transit_offsets()
)


TRANSIT_RESULTS = {

    Fe: {}

    for Fe in engine.FE_LIST
}


for gamma_t in TRANSIT_VALUES:


    print()
    print(
        "Gamma_transit/Gamma =",
        gamma_t
    )


    for Fe in engine.FE_LIST:


        print(
            f"  F'={Fe}"
        )


        result = analyze_scan(

            local_scan(

                Fe,

                TRANSIT_OFFSETS,

                gamma_t
            )
        )


        TRANSIT_RESULTS[
            Fe
        ][
            gamma_t
        ] = result


# ============================================================
# 12. Transit tables
# ============================================================

for Fe in engine.FE_LIST:


    print()
    print(
        "-" * 76
    )

    print(
        f"F=2 -> F'={Fe}"
    )

    print(
        "-" * 76
    )


    print(

        f"{'Gamma_t/Gamma':>14}"

        f"{'lock shift MHz':>18}"

        f"{'|slope|':>18}"

        f"{'Vpp':>18}"

        f"{'phi(deg)':>14}"
    )


    for gamma_t in TRANSIT_VALUES:


        r = (
            TRANSIT_RESULTS[
                Fe
            ][
                gamma_t
            ]
        )


        print(

            f"{gamma_t:14.5f}"

            f"{r['zero_shift_mhz']:18.6f}"

            f"{r['slope_mag']:18.8e}"

            f"{r['Vpp']:18.8e}"

            f"{np.degrees(r['phi']):14.4f}"
        )


# ============================================================
# 13. Quantitative transit sensitivity
# ============================================================

print()
print(
    "=" * 76
)

print(
    "Transit sensitivity summary"
)

print(
    "=" * 76
)


for Fe in engine.FE_LIST:


    zeros = np.array(

        [

            TRANSIT_RESULTS[
                Fe
            ][
                gamma_t
            ][
                "zero_shift_mhz"
            ]

            for gamma_t in TRANSIT_VALUES
        ]
    )


    slopes = np.array(

        [

            TRANSIT_RESULTS[
                Fe
            ][
                gamma_t
            ][
                "slope_mag"
            ]

            for gamma_t in TRANSIT_VALUES
        ]
    )


    vpps = np.array(

        [

            TRANSIT_RESULTS[
                Fe
            ][
                gamma_t
            ][
                "Vpp"
            ]

            for gamma_t in TRANSIT_VALUES
        ]
    )


    zero_span = (

        np.nanmax(
            zeros
        )

        -

        np.nanmin(
            zeros
        )
    )


    slope_span = (

        np.nanmax(
            slopes
        )

        -

        np.nanmin(
            slopes
        )

    ) / max(

        slopes[
            np.argmin(

                np.abs(

                    TRANSIT_VALUES

                    -

                    REFERENCE_TRANSIT
                )
            )
        ],

        1e-300
    )


    vpp_span = (

        np.nanmax(
            vpps
        )

        -

        np.nanmin(
            vpps
        )

    ) / max(

        vpps[
            np.argmin(

                np.abs(

                    TRANSIT_VALUES

                    -

                    REFERENCE_TRANSIT
                )
            )
        ],

        1e-300
    )


    print()
    print(
        f"F'={Fe}:"
    )

    print(
        "  lock-point total span =",
        zero_span,
        "MHz"
    )

    print(
        "  slope variation / reference =",
        slope_span
    )

    print(
        "  Vpp variation / reference =",
        vpp_span
    )


# ============================================================
# 14. beta=0 regression test
# ============================================================

install_transit_model(
    REFERENCE_TRANSIT
)


TEST_DELTA = (

    engine.HF_OFFSET[
        3
    ]

    +

    engine.FM
)


reference = (
    engine.response_at_detuning(

        TEST_DELTA,

        beta=engine.BETA
    )
)


null = (
    engine.response_at_detuning(

        TEST_DELTA,

        beta=0.0
    )
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
    "Regression validation:"
)

print(
    "beta=0 null-test ratio =",
    NULL_RATIO
)


# ============================================================
# 15. Figure 1
#
# 0.50 MHz vs 0.25 MHz convergence
# ============================================================

plt.figure(
    figsize=(8, 5)
)


for Fe in engine.FE_LIST:


    fine = (
        GRID_RESULTS[
            Fe
        ][
            "fine"
        ]
    )


    coarse = (
        GRID_RESULTS[
            Fe
        ][
            "coarse"
        ]
    )


    # Use the FINE-grid phase for both,
    # so this is a true grid comparison.

    phi = (
        fine[
            "phi"
        ]
    )


    S_fine = np.real(

        fine[
            "Z"
        ]

        *

        np.exp(
            -1j
            *
            phi
        )
    )


    S_coarse = np.real(

        coarse[
            "Z"
        ]

        *

        np.exp(
            -1j
            *
            phi
        )
    )


    scale = max(

        np.max(
            np.abs(
                S_fine
            )
        ),

        1e-300
    )


    plt.plot(

        fine[
            "offset_mhz"
        ],

        S_fine
        /
        scale,

        label=(
            f"F'={Fe}, 0.25 MHz"
        )
    )


    plt.plot(

        coarse[
            "offset_mhz"
        ],

        S_coarse
        /
        scale,

        "o",

        markersize=3,

        label=(
            f"F'={Fe}, 0.50 MHz"
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
    "Local detuning from nominal resonance (MHz)"
)


plt.ylabel(
    "Common-normalized MTS signal"
)


plt.title(
    "Local-grid convergence of hyperfine MTS"
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
# Lock shift vs transit rate
# ============================================================

plt.figure(
    figsize=(8, 5)
)


for Fe in engine.FE_LIST:


    y = [

        TRANSIT_RESULTS[
            Fe
        ][
            gamma_t
        ][
            "zero_shift_mhz"
        ]

        for gamma_t in TRANSIT_VALUES
    ]


    plt.plot(

        TRANSIT_VALUES,

        y,

        "o-",

        label=(
            f"F'={Fe}"
        )
    )


plt.axhline(
    0,
    linewidth=0.8
)


plt.xlabel(
    r"$\Gamma_{\rm transit}/\Gamma$"
)


plt.ylabel(
    "Lock shift from nominal resonance (MHz)"
)


plt.title(
    "Lock-point sensitivity to ground-state transit relaxation"
)


plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 3
#
# Slope sensitivity
# ============================================================

plt.figure(
    figsize=(8, 5)
)


for Fe in engine.FE_LIST:


    slopes = np.array(

        [

            TRANSIT_RESULTS[
                Fe
            ][
                gamma_t
            ][
                "slope_mag"
            ]

            for gamma_t in TRANSIT_VALUES
        ]
    )


    iref = np.argmin(

        np.abs(

            TRANSIT_VALUES

            -

            REFERENCE_TRANSIT
        )
    )


    plt.plot(

        TRANSIT_VALUES,

        slopes
        /
        slopes[
            iref
        ],

        "o-",

        label=(
            f"F'={Fe}"
        )
    )


plt.xlabel(
    r"$\Gamma_{\rm transit}/\Gamma$"
)


plt.ylabel(
    "Center-slope ratio"
)


plt.title(
    "Discriminator-slope sensitivity to transit relaxation"
)


plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 4
#
# Vpp sensitivity
# ============================================================

plt.figure(
    figsize=(8, 5)
)


for Fe in engine.FE_LIST:


    values = np.array(

        [

            TRANSIT_RESULTS[
                Fe
            ][
                gamma_t
            ][
                "Vpp"
            ]

            for gamma_t in TRANSIT_VALUES
        ]
    )


    iref = np.argmin(

        np.abs(

            TRANSIT_VALUES

            -

            REFERENCE_TRANSIT
        )
    )


    plt.plot(

        TRANSIT_VALUES,

        values
        /
        values[
            iref
        ],

        "o-",

        label=(
            f"F'={Fe}"
        )
    )


plt.xlabel(
    r"$\Gamma_{\rm transit}/\Gamma$"
)


plt.ylabel(
    "Vpp ratio"
)


plt.title(
    "MTS amplitude sensitivity to transit relaxation"
)


plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 5
#
# F'=3 error signal for different transit rates
#
# Common normalization reveals whether the CLOSED cycling
# transition is actually stable.
# ============================================================

plt.figure(
    figsize=(8, 5)
)


REFERENCE_RESULT = (
    TRANSIT_RESULTS[
        3
    ][
        REFERENCE_TRANSIT
    ]
)


COMMON_SCALE = max(

    np.max(

        np.abs(

            REFERENCE_RESULT[
                "signal"
            ]
        )
    ),

    1e-300
)


for gamma_t in TRANSIT_VALUES:


    r = (
        TRANSIT_RESULTS[
            3
        ][
            gamma_t
        ]
    )


    plt.plot(

        r[
            "offset_mhz"
        ],

        r[
            "signal"
        ]
        /
        COMMON_SCALE,

        label=(

            r"$\Gamma_t/\Gamma=$"

            f"{gamma_t:.3f}"
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
    r"Local detuning from "
    r"$F=2\rightarrow F'=3$ (MHz)"
)


plt.ylabel(
    "MTS signal "
    "(common normalization)"
)


plt.title(
    r"$F'=3$ stability versus transit relaxation"
)


plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.show()