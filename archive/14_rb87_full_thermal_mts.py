"""
14_rb87_full_thermal_mts.py

FINAL INTEGRATION STAGE

87Rb D2:

    Fg = 1,2
        ->
    Fe = 1,2,3

including:

    real hyperfine offsets
    Zeeman sublevels
    Wigner 3j / 6j dipole strengths
    spontaneous branching
    ground-state transit relaxation
    explicit rho^(3)
    FWM generated probe sidebands
    heterodyne detection
    Maxwell-Boltzmann velocity averaging
    optimized lock-in demodulation

Core chain
----------

    real 87Rb atom
          |
          v
    rho^(3)(Delta, kv)
          |
          v
    P_+^(3), P_-^(3)
          |
          v
    Maxwell velocity integral
          |
          v
    <P_+>, <P_->
          |
          v
    generated optical fields
          |
          v
    complex RF heterodyne Z
          |
          v
    MTS discriminator


IMPORTANT
---------

MODE = "FINAL"

is the quantitative mode.

Do NOT use "SMOKE" values in the final report.

The selected GAMMA_TRANSIT is a model parameter.
Stage 13b showed that lock point and slope depend on it.
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
# 0. CALCULATION MODE
# ============================================================

# ------------------------------------------------------------
# FINAL:
#     quantitative calculation
#     use for report / final interpretation
#
# SMOKE:
#     ONLY check that the program runs
#     DO NOT quote its numerical values
# ------------------------------------------------------------

MODE = "FINAL"


if MODE not in (
    "FINAL",
    "SMOKE"
):

    raise ValueError(
        'MODE must be "FINAL" or "SMOKE".'
    )


# ============================================================
# 1. Main physical settings
# ============================================================

TEMPERATURE_K = 300.0


# ------------------------------------------------------------
# Ground-state transit relaxation
#
# Stage 13b demonstrated that this is physically important.
#
# Reference value retained from Stage 13:
#
#     Gamma_t / Gamma = 0.02
#
# Do NOT describe it as a universal Rb87 constant.
# ------------------------------------------------------------

GAMMA_TRANSIT = 0.020


# ------------------------------------------------------------
# We focus the final thermal calculation on the actual
# stabilization target:
#
#     F=2 -> F'=3
#
# IMPORTANT:
# F'=1 and F'=2 are STILL fully included in the Hamiltonian.
# We are only limiting the laser-frequency scan range.
# ------------------------------------------------------------

LOCK_FE = 3


# ============================================================
# 2. FINAL numerical settings
# ============================================================

if MODE == "FINAL":

    # --------------------------------------------------------
    # Laser-frequency grid
    #
    # Fine near the lock point:
    #       0.25 MHz
    #
    # Coarser in the wings:
    #       1 MHz
    # --------------------------------------------------------

    CENTER_SPAN_MHZ = 3.0
    CENTER_STEP_MHZ = 0.25

    FULL_SPAN_MHZ = 12.0
    WING_STEP_MHZ = 1.0


    # --------------------------------------------------------
    # Velocity integration
    #
    # Broad Maxwell background:
    #       4 Gamma spacing
    #
    # Fine resonant velocity windows:
    #       0.5 Gamma spacing
    #
    # Windows are automatically placed around every
    # F'=1,2,3 resonant velocity class.
    # --------------------------------------------------------

    Q_COARSE_STEP = 4.0

    Q_FINE_STEP = 0.50

    VELOCITY_SPAN_U = 4.0


    # --------------------------------------------------------
    # Convergence calculation
    # --------------------------------------------------------

    RUN_CONVERGENCE_CHECK = True

    Q_CONV_COARSE_STEP = 2.0

    Q_CONV_FINE_STEP = 0.25

    CONVERGENCE_OFFSETS_MHZ = np.array(
        [
            -2.0,
            0.0,
            +2.0
        ]
    )


else:

    print()
    print(
        "WARNING: MODE = SMOKE"
    )

    print(
        "These values are NOT quantitative."
    )


    CENTER_SPAN_MHZ = 2.0
    CENTER_STEP_MHZ = 1.0

    FULL_SPAN_MHZ = 6.0
    WING_STEP_MHZ = 2.0

    Q_COARSE_STEP = 10.0
    Q_FINE_STEP = 1.5

    VELOCITY_SPAN_U = 3.0

    RUN_CONVERGENCE_CHECK = False

    CONVERGENCE_OFFSETS_MHZ = np.array(
        [
            0.0
        ]
    )


# ============================================================
# 3. Physical constants
# ============================================================

KB = 1.380649e-23

AMU = 1.66053906660e-27


MASS_RB87 = (
    86.9091805
    *
    AMU
)


LAMBDA_D2 = (
    780.24e-9
)


# ============================================================
# 4. Load validated Stage-13 engine
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

        "Put 14_rb87_full_thermal_mts.py and "
        "13_rb87_hyperfine_zeeman_mts.py "
        "in the same directory."
    )


MODULE_NAME = (
    "rb87_hyperfine_engine_final"
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


sys.modules[
    MODULE_NAME
] = engine


spec.loader.exec_module(
    engine
)


# ============================================================
# 5. Install selected transit model
#
# Reconstruct the dissipator exactly as in Stage 13b.
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

    L = (
        L_DISS_SPONT.copy()
    )


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
                    state_from
                    ==
                    state_to
                ):

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


# Replace only the transit-dependent L0 builder.
#
# All validated rho^(3), FWM and heterodyne code
# remains untouched.

engine.GAMMA_TRANSIT = (
    GAMMA_TRANSIT
)

engine.build_L0 = (
    build_L0_selected
)


# ============================================================
# 6. Check rho0 stationarity
# ============================================================

RHO0_RESIDUAL = np.linalg.norm(

    engine.build_L0(
        0.0
    )

    @

    engine.RHO0_VEC
)


# ============================================================
# 7. Thermal velocity distribution
# ============================================================

def thermal_speed_u(
    temperature
):
    """
    1D Maxwell distribution:

        f(v)
        =
        exp[-(v/u)^2]
        /
        (sqrt(pi) u)

    where

        u = sqrt(2 k_B T / m)
    """

    return np.sqrt(

        2.0

        *
        KB

        *
        temperature

        /

        MASS_RB87
    )


U_THERMAL = (
    thermal_speed_u(
        TEMPERATURE_K
    )
)


# ------------------------------------------------------------
# q = kv / Gamma
#
# In the normalized frequency convention used by the
# validated engine:
#
#     D = (u/lambda) / [Gamma/(2pi)]
#
# numerically using Gamma_MHZ = 6.065 MHz.
# ------------------------------------------------------------

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
# 8. Laser detuning grid
# ============================================================

def regular_grid(
    start,
    stop,
    step
):

    count = int(

        np.floor(

            (
                stop
                -
                start
            )

            /

            step

            +
            0.5
        )
    )


    return (

        start

        +

        np.arange(
            count + 1
        )

        *
        step
    )


CENTER_GRID = regular_grid(

    -CENTER_SPAN_MHZ,

    +CENTER_SPAN_MHZ,

    CENTER_STEP_MHZ
)


LEFT_WING = regular_grid(

    -FULL_SPAN_MHZ,

    -CENTER_SPAN_MHZ,

    WING_STEP_MHZ
)


RIGHT_WING = regular_grid(

    +CENTER_SPAN_MHZ,

    +FULL_SPAN_MHZ,

    WING_STEP_MHZ
)


LOCAL_OFFSET_MHZ = np.unique(

    np.round(

        np.concatenate(
            [
                LEFT_WING,
                CENTER_GRID,
                RIGHT_WING
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
# 9. Maxwell weight
# ============================================================

def maxwell_weight_q(
    q,
    D
):

    return (

        np.exp(
            -(q / D)**2
        )

        /

        (
            np.sqrt(np.pi)

            *
            D
        )
    )


# ============================================================
# 10. Resonance-aware velocity grid
#
# Uniform ultra-fine integration over +/-4D would be
# extremely expensive for the 529-dimensional Liouvillian.
#
# Instead:
#
# 1. broad coarse grid covers the full Maxwell distribution
#
# 2. fine windows are inserted around every resonant
#    velocity class of F'=1,2,3
#
# 3. convergence is checked against a denser grid
#
# Nothing is inferred from the result unless convergence
# passes.
# ============================================================

def build_velocity_grid(
    delta,
    D,
    coarse_step,
    fine_step,
    span_u=VELOCITY_SPAN_U
):

    qmax = (
        span_u
        *
        D
    )


    coarse = regular_grid(

        -qmax,

        +qmax,

        coarse_step
    )


    grids = [
        coarse,
        np.array(
            [
                0.0
            ]
        )
    ]


    # --------------------------------------------------------
    # Fine window width
    #
    # Third-order recursion may contain harmonics out to
    # approximately +/-3 Omega_m.
    #
    # Add several natural linewidths around that range.
    # --------------------------------------------------------

    fine_half_width = (

        3.0
        *
        engine.FM

        +

        4.0
    )


    # --------------------------------------------------------
    # Resonant velocity classes:
    #
    # approximately
    #
    #     kv/Gamma
    #       ~
    #     Delta - Delta_F'
    #
    # with opposite propagation directions producing +/-.
    # --------------------------------------------------------

    centers = []


    for Fe in engine.FE_LIST:

        base = (

            delta

            -

            engine.HF_OFFSET[
                Fe
            ]
        )


        centers.extend(
            [
                +base,
                -base
            ]
        )


    for center in centers:

        if (
            center
            <
            -qmax
            -
            fine_half_width
        ):

            continue


        if (
            center
            >
            +qmax
            +
            fine_half_width
        ):

            continue


        left = max(

            -qmax,

            center
            -
            fine_half_width
        )


        right = min(

            +qmax,

            center
            +
            fine_half_width
        )


        grids.append(

            regular_grid(

                left,

                right,

                fine_step
            )
        )


    q = np.unique(

        np.round(

            np.concatenate(
                grids
            ),

            10
        )
    )


    q = q[
        (
            q >= -qmax
        )

        &

        (
            q <= +qmax
        )
    ]


    weight = (
        maxwell_weight_q(
            q,
            D
        )
    )


    normalization = np.trapezoid(

        weight,

        q
    )


    weight = (

        weight

        /

        normalization
    )


    return (
        q,
        weight
    )


# ============================================================
# 11. Response cache
# ============================================================

RESPONSE_CACHE = {}


def cached_atomic_response(
    delta,
    kv,
    beta=engine.BETA,
    field_scale=1.0
):

    key = (

        round(
            float(delta),
            10
        ),

        round(
            float(kv),
            10
        ),

        round(
            float(beta),
            10
        ),

        round(
            float(field_scale),
            8
        ),

        round(
            float(GAMMA_TRANSIT),
            8
        )
    )


    if key not in RESPONSE_CACHE:

        RESPONSE_CACHE[
            key
        ] = engine.response_at_detuning(

            delta=delta,

            kv=kv,

            beta=beta,

            field_scale=field_scale
        )


    return (
        RESPONSE_CACHE[
            key
        ]
    )


# ============================================================
# 12. Thermal average at one laser detuning
# ============================================================

def thermal_average_point(
    delta,
    coarse_step=Q_COARSE_STEP,
    fine_step=Q_FINE_STEP,
    span_u=VELOCITY_SPAN_U,
    beta=engine.BETA,
    field_scale=1.0,
    return_profile=False
):

    q, weight = (
        build_velocity_grid(

            delta=delta,

            D=DOPPLER_D,

            coarse_step=coarse_step,

            fine_step=fine_step,

            span_u=span_u
        )
    )


    P_plus = np.zeros(
        len(q),
        dtype=complex
    )


    P_minus = np.zeros_like(
        P_plus
    )


    Z_velocity = np.zeros_like(
        P_plus
    )


    for i, kv in enumerate(
        q
    ):

        result = (
            cached_atomic_response(

                delta=delta,

                kv=kv,

                beta=beta,

                field_scale=field_scale
            )
        )


        P_plus[i] = (
            result[
                "P_plus"
            ]
        )


        P_minus[i] = (
            result[
                "P_minus"
            ]
        )


        Z_velocity[i] = (
            result[
                "Z"
            ]
        )


    # --------------------------------------------------------
    # Correct macroscopic order:
    #
    # average atomic polarization first
    # --------------------------------------------------------

    P_plus_avg = np.trapezoid(

        P_plus
        *
        weight,

        q
    )


    P_minus_avg = np.trapezoid(

        P_minus
        *
        weight,

        q
    )


    # --------------------------------------------------------
    # Then Maxwell-generated sidebands and heterodyne.
    #
    # Same convention as Stage 13.
    # --------------------------------------------------------

    E_plus = (

        -1j

        *
        P_plus_avg
    )


    E_minus = (

        -1j

        *
        P_minus_avg
    )


    E_c = (

        field_scale

        *
        engine.OMEGA_PROBE
    )


    Z_avg = (

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


    output = {

        "P_plus":
        P_plus_avg,

        "P_minus":
        P_minus_avg,

        "Z":
        Z_avg,

        "Nq":
        len(q)
    }


    if return_profile:

        output.update(
            {
                "q":
                q,

                "weight":
                weight,

                "P_plus_v":
                P_plus,

                "P_minus_v":
                P_minus,

                "Z_v":
                Z_velocity
            }
        )


    return output


# ============================================================
# 13. No-Doppler reference
# ============================================================

def no_doppler_response(
    delta
):

    return (
        engine.response_at_detuning(

            delta=delta,

            kv=0.0
        )
    )


# ============================================================
# 14. Analysis utilities
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


    zz = (
        Z[
            mask
        ]
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


def refined_zero(
    offset_mhz,
    signal,
    window_mhz=2.5
):

    mask = (

        np.abs(
            offset_mhz
        )

        <=

        window_mhz
    )


    x = offset_mhz[
        mask
    ]


    y = signal[
        mask
    ]


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

        -window_mhz,

        +window_mhz,

        10001
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


def extrema_metrics(
    offset_mhz,
    signal,
    zero
):

    order = np.argsort(
        offset_mhz
    )


    x = offset_mhz[
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

        20001
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


    il = left[-1]
    ir = right[0]


    Vpp = abs(

        yy[
            ir
        ]

        -

        yy[
            il
        ]
    )


    separation = (

        dense[
            ir
        ]

        -

        dense[
            il
        ]
    )


    return (

        float(Vpp),

        float(separation),

        float(
            dense[
                il
            ]
        ),

        float(
            dense[
                ir
            ]
        )
    )


# ============================================================
# 15. Header
# ============================================================

print()
print(
    "=" * 78
)

print(
    "Stage F3 / FINAL: full thermal 87Rb hyperfine + Zeeman MTS"
)

print(
    "=" * 78
)


print()
print(
    "CALCULATION MODE =",
    MODE
)


if MODE != "FINAL":

    print()
    print(
        "WARNING: SMOKE results are not quantitative."
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

print(
    "velocity integration = resonance-aware composite trapezoid"
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
    "IMPORTANT: this is a model parameter, not a universal constant."
)

print(
    "rho0 steady-state residual =",
    RHO0_RESIDUAL
)


print()
print(
    "Hyperfine offsets:"
)


for Fe in engine.FE_LIST:

    print(

        f"F'= {Fe}: "

        f"{engine.HF_OFFSET_MHZ[Fe]:+.6f} MHz"
    )


# ============================================================
# 16. Main no-Doppler scan
# ============================================================

print()
print(
    "=" * 78
)

print(
    "Calculating no-Doppler reference..."
)

print(
    "=" * 78
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
        no_doppler_response(
            delta
        )
    )


    P_PLUS_0[i] = (
        result[
            "P_plus"
        ]
    )


    P_MINUS_0[i] = (
        result[
            "P_minus"
        ]
    )


    Z_0[i] = (
        result[
            "Z"
        ]
    )


# ============================================================
# 17. Full thermal scan
# ============================================================

print()
print(
    "=" * 78
)

print(
    "Calculating FULL thermal spectrum around F'=3..."
)

print(
    "=" * 78
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


NQ_USED = np.zeros(
    len(
        DELTA_SCAN
    ),
    dtype=int
)


for i, delta in enumerate(
    DELTA_SCAN
):

    print()

    print(
        f"{i+1:3d}/{len(DELTA_SCAN)}"
        f"   offset = "
        f"{LOCAL_OFFSET_MHZ[i]:+7.3f} MHz"
    )


    result = (
        thermal_average_point(
            delta
        )
    )


    P_PLUS_T[i] = (
        result[
            "P_plus"
        ]
    )


    P_MINUS_T[i] = (
        result[
            "P_minus"
        ]
    )


    Z_T[i] = (
        result[
            "Z"
        ]
    )


    NQ_USED[i] = (
        result[
            "Nq"
        ]
    )


    print(
        "    velocity points =",
        NQ_USED[i]
    )


# ============================================================
# 18. Thermal lock analysis
# ============================================================

SLOPE_T = (
    complex_center_slope(

        LOCAL_OFFSET_MHZ,

        Z_T
    )
)


PHI_T = np.angle(
    SLOPE_T
)


S_T = demodulate(

    Z_T,

    PHI_T
)


ZERO_T = refined_zero(

    LOCAL_OFFSET_MHZ,

    S_T
)


(
    VPP_T,
    SEP_T,
    EXT_LEFT_T,
    EXT_RIGHT_T

) = extrema_metrics(

    LOCAL_OFFSET_MHZ,

    S_T,

    ZERO_T
)


# ============================================================
# 19. No-Doppler comparison at SAME mixer phase
# ============================================================

S_0_FIXED = demodulate(

    Z_0,

    PHI_T
)


SLOPE_0_COMPLEX = (
    complex_center_slope(

        LOCAL_OFFSET_MHZ,

        Z_0
    )
)


SLOPE_0_FIXED = abs(

    np.real(

        SLOPE_0_COMPLEX

        *

        np.exp(
            -1j
            *
            PHI_T
        )
    )
)


SLOPE_T_FIXED = abs(

    np.real(

        SLOPE_T

        *

        np.exp(
            -1j
            *
            PHI_T
        )
    )
)


ZERO_0_FIXED = refined_zero(

    LOCAL_OFFSET_MHZ,

    S_0_FIXED
)


# ============================================================
# 20. Velocity-integration convergence
# ============================================================

CONVERGENCE_ERROR = np.nan

CONVERGENCE_ERRORS = []


if RUN_CONVERGENCE_CHECK:

    print()
    print(
        "=" * 78
    )

    print(
        "Velocity integration convergence"
    )

    print(
        "=" * 78
    )


    for offset_mhz in (
        CONVERGENCE_OFFSETS_MHZ
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


        print()
        print(
            "offset =",
            offset_mhz,
            "MHz"
        )


        base = thermal_average_point(

            delta,

            coarse_step=Q_COARSE_STEP,

            fine_step=Q_FINE_STEP
        )


        refined = thermal_average_point(

            delta,

            coarse_step=Q_CONV_COARSE_STEP,

            fine_step=Q_CONV_FINE_STEP
        )


        error = (

            abs(

                refined[
                    "Z"
                ]

                -

                base[
                    "Z"
                ]
            )

            /

            max(

                abs(
                    refined[
                        "Z"
                    ]
                ),

                1e-300
            )
        )


        CONVERGENCE_ERRORS.append(
            error
        )


        print(
            "relative Z error =",
            error
        )


    CONVERGENCE_ERRORS = np.asarray(
        CONVERGENCE_ERRORS
    )


    CONVERGENCE_ERROR = (
        np.linalg.norm(
            CONVERGENCE_ERRORS
        )

        /

        np.sqrt(
            len(
                CONVERGENCE_ERRORS
            )
        )
    )


    # --------------------------------------------------------
    # +/-4u versus +/-5u boundary test at center
    # --------------------------------------------------------

    delta_center = (

        LOCK_CENTER_MHZ

        /

        engine.GAMMA_MHZ
    )


    span4 = thermal_average_point(

        delta_center,

        span_u=4.0
    )


    span5 = thermal_average_point(

        delta_center,

        span_u=5.0
    )


    BOUNDARY_ERROR = (

        abs(

            span5[
                "Z"
            ]

            -

            span4[
                "Z"
            ]
        )

        /

        max(

            abs(
                span5[
                    "Z"
                ]
            ),

            1e-300
        )
    )


else:

    BOUNDARY_ERROR = np.nan


# ============================================================
# 21. beta=0 thermal regression
# ============================================================

TEST_INDEX = np.argmax(
    np.abs(
        Z_T
    )
)


TEST_DELTA = (
    DELTA_SCAN[
        TEST_INDEX
    ]
)


TEST_OFFSET = (
    LOCAL_OFFSET_MHZ[
        TEST_INDEX
    ]
)


NORMAL_TEST = thermal_average_point(

    TEST_DELTA,

    beta=engine.BETA
)


NULL_TEST = thermal_average_point(

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
# 22. Thermal velocity contribution at final lock point
# ============================================================

if np.isfinite(
    ZERO_T
):

    PROFILE_DELTA = (

        (
            LOCK_CENTER_MHZ
            +
            ZERO_T
        )

        /

        engine.GAMMA_MHZ
    )

else:

    PROFILE_DELTA = (

        LOCK_CENTER_MHZ

        /

        engine.GAMMA_MHZ
    )


PROFILE = thermal_average_point(

    PROFILE_DELTA,

    return_profile=True
)


# ============================================================
# 23. Final diagnostics
# ============================================================

print()
print(
    "=" * 78
)

print(
    "FINAL THERMAL MTS RESULT"
)

print(
    "=" * 78
)


print()
print(
    "Thermal F=2 -> F'=3 discriminator:"
)

print(
    "complex center slope =",
    SLOPE_T
)

print(
    "|center slope| =",
    abs(
        SLOPE_T
    )
)

print(
    "optimum demodulation phase =",
    np.degrees(
        PHI_T
    ),
    "deg"
)

print(
    "thermal lock shift from nominal F'=3 =",
    ZERO_T,
    "MHz"
)

print(
    "thermal lock shift =",
    (
        ZERO_T
        *
        1e3
        if np.isfinite(
            ZERO_T
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
    "No-Doppler comparison at SAME demodulation phase:"
)

print(
    "no-Doppler lock shift =",
    ZERO_0_FIXED,
    "MHz"
)

print(
    "no-Doppler slope =",
    SLOPE_0_FIXED
)

print(
    "thermal slope =",
    SLOPE_T_FIXED
)

print(
    "thermal / no-Doppler slope ratio =",
    (

        SLOPE_T_FIXED

        /

        max(
            SLOPE_0_FIXED,
            1e-300
        )
    )
)


print()
print(
    "Velocity integration:"
)

print(
    "mean q-grid points =",
    np.mean(
        NQ_USED
    )
)

print(
    "min/max q-grid points =",
    np.min(
        NQ_USED
    ),
    np.max(
        NQ_USED
    )
)


if RUN_CONVERGENCE_CHECK:

    print(
        "RMS refined-grid relative error =",
        CONVERGENCE_ERROR
    )

    print(
        "+/-4u -> +/-5u boundary error =",
        BOUNDARY_ERROR
    )


print()
print(
    "Structural regression:"
)

print(
    "beta=0 test offset =",
    TEST_OFFSET,
    "MHz"
)

print(
    "beta=0 thermal null ratio =",
    NULL_RATIO
)


print()
print(
    "Model condition:"
)

print(
    "Temperature =",
    TEMPERATURE_K,
    "K"
)

print(
    "Gamma_transit/Gamma =",
    GAMMA_TRANSIT
)

print(
    "DO NOT quote the lock shift without quoting "
    "this transit parameter."
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


print()
print(
    "All finite =",
    ALL_FINITE
)


# ------------------------------------------------------------
# Suggested numerical acceptance
# ------------------------------------------------------------

print()
print(
    "=" * 78
)

print(
    "FINAL VALIDATION"
)

print(
    "=" * 78
)


if MODE == "FINAL":

    if (
        RUN_CONVERGENCE_CHECK
        and
        CONVERGENCE_ERROR
        <
        1e-2
    ):

        print(
            "PASS: velocity integration RMS error < 1%."
        )

    else:

        print(
            "CHECK: velocity-grid convergence has not "
            "yet reached the 1% target."
        )


    if (
        RUN_CONVERGENCE_CHECK
        and
        BOUNDARY_ERROR
        <
        1e-3
    ):

        print(
            "PASS: Maxwell integration boundary is converged."
        )

    else:

        print(
            "CHECK: Maxwell boundary convergence."
        )


    if (
        NULL_RATIO
        <
        1e-3
    ):

        print(
            "PASS: beta=0 thermal null test."
        )

    else:

        print(
            "CHECK: beta=0 thermal residual."
        )


if ALL_FINITE:

    print(
        "PASS: all thermal quantities finite."
    )


# ============================================================
# 24. Figure 1
#
# Maxwell distribution and resonant velocity classes
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

    q_plot,

    DOPPLER_D
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


delta_lock_norm = (
    PROFILE_DELTA
)


for Fe in engine.FE_LIST:

    center_q = (

        delta_lock_norm

        -

        engine.HF_OFFSET[
            Fe
        ]
    )


    plt.axvline(

        +center_q,

        linestyle="--",

        linewidth=0.8,

        label=(
            f"F'={Fe} velocity class"
        )
    )


    if abs(
        center_q
    ) > 1e-8:

        plt.axvline(

            -center_q,

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
    "Thermal Rb87 velocity distribution "
    "and resonant hyperfine velocity classes"
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
# Generated sidebands
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
    "Generated third-order polarization "
    "(common normalization)"
)

plt.title(
    "Full thermal averaging of generated MTS sidebands"
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
# Complex thermal heterodyne response
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
    "(common no-Doppler normalization)"
)

plt.title(
    "Full Rb87 thermal MTS heterodyne response"
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
# FINAL MTS discriminator
# ============================================================

ERROR_SCALE = max(

    np.max(
        np.abs(
            S_0_FIXED
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

    S_0_FIXED

    /

    ERROR_SCALE,

    label="No Doppler"
)


plt.plot(

    LOCAL_OFFSET_MHZ,

    S_T

    /

    ERROR_SCALE,

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
    ZERO_T
):

    plt.axvline(

        ZERO_T,

        linestyle=":",

        linewidth=1.0,

        label=(
            "thermal lock = "
            f"{ZERO_T*1e3:.1f} kHz"
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

    "FINAL thermal MTS discriminator\n"

    f"T={TEMPERATURE_K:.0f} K, "
    f"Gamma_t/Gamma={GAMMA_TRANSIT:.3f}, "
    f"phi={np.degrees(PHI_T):.1f} deg"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 5
#
# Velocity-resolved contribution at lock point
# ============================================================

q = (
    PROFILE[
        "q"
    ]
)


weight = (
    PROFILE[
        "weight"
    ]
)


Zv = (
    PROFILE[
        "Z_v"
    ]
)


contribution = (

    np.abs(
        Zv
    )

    *
    weight
)


CONTRIBUTION_SCALE = max(

    np.max(
        contribution
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

    q,

    contribution
    /
    CONTRIBUTION_SCALE
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
    ) > 1e-8:

        plt.axvline(

            -q_res,

            linestyle=":",

            linewidth=0.8
        )


plt.xlabel(
    r"$q=kv/\Gamma$"
)

plt.ylabel(
    "Relative velocity-class contribution"
)

plt.title(
    "Velocity classes contributing to the final MTS lock"
)

plt.legend(
    fontsize=8
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 6
#
# Numerical convergence
# ============================================================

if RUN_CONVERGENCE_CHECK:

    plt.figure(
        figsize=(
            7,
            5
        )
    )


    plt.plot(

        CONVERGENCE_OFFSETS_MHZ,

        CONVERGENCE_ERRORS,

        "o-"
    )


    plt.axhline(

        1e-2,

        linestyle="--",

        linewidth=0.8,

        label="1% acceptance"
    )


    plt.xlabel(
        "Local laser detuning (MHz)"
    )

    plt.ylabel(
        "Base -> refined velocity-grid relative error"
    )

    plt.title(
        "Final thermal velocity-integration convergence"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


plt.show()