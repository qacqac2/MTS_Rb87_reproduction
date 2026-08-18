"""
11_mts_rho3_doppler_average.py

Stage E:
Thermal/Doppler averaging of the VALIDATED Stage-09
explicit third-order MTS engine.

Core chain:

    rho^(3)(x, kv)
        ->
    P_+^(3), P_-^(3)
        ->
    Maxwell-Boltzmann velocity average
        ->
    generated optical sidebands
        ->
    RF heterodyne envelope Z
        ->
    lock-in MTS error signal

IMPORTANT:
Doppler shift is NOT implemented by simply replacing

    x -> x - kv.

Instead, Stage-09 already keeps the propagation label k.
The moving-atom Fourier frequency is

    Omega_slow = q*Omega_m - k*kv,

so counterpropagating pump/probe Doppler signs are retained
automatically.
"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import importlib.util

from scipy.signal import find_peaks


# ============================================================
# 0. Calculation mode
# ============================================================

FAST_MODE = False

RUN_CONVERGENCE_CHECK = True
RUN_SCALING_CHECK = True
RUN_DOPPLER_WIDTH_SCAN = False


# ============================================================
# 1. Dynamically load Stage-09 engine
#
# Python normal import syntax cannot import a module whose
# filename starts with "09_".
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

ENGINE_PATH = (
    THIS_DIR
    / "09_mts_rho3_chain.py"
)


if not ENGINE_PATH.exists():

    raise FileNotFoundError(
        "\nCannot find Stage-09 engine:\n"
        f"{ENGINE_PATH}\n\n"
        "Put 09_mts_rho3_chain.py and this file "
        "in the same folder."
    )


spec = importlib.util.spec_from_file_location(
    "mts_rho3_engine",
    ENGINE_PATH
)

engine = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    engine
)


# ------------------------------------------------------------
# API compatibility
#
# If you added the wrappers above, use rho3_sidebands().
#
# If not, the script can still fall back to response_at_x().
# ------------------------------------------------------------

def engine_sidebands(
    x,
    kv,
    beta=None,
    omega_pump=None,
    omega_probe=None,
    field_scale=1.0
):

    if beta is None:
        beta = engine.BETA

    if omega_pump is None:
        omega_pump = engine.OMEGA_PUMP

    if omega_probe is None:
        omega_probe = engine.OMEGA_PROBE


    if hasattr(
        engine,
        "rho3_sidebands"
    ):

        return engine.rho3_sidebands(
            x=x,
            beta=beta,
            omega_pump=omega_pump,
            omega_probe=omega_probe,
            field_scale=field_scale,
            kv=kv
        )


    # fallback for the original Stage-09 file

    result = engine.response_at_x(
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


def heterodyne_from_sidebands(
    P_plus,
    P_minus,
    omega_probe=None,
    field_scale=1.0
):

    if omega_probe is None:
        omega_probe = engine.OMEGA_PROBE


    # Keep exactly the same Maxwell/heterodyne convention
    # as validated Stage 09.

    E_plus = (
        -1j * P_plus
    )

    E_minus = (
        -1j * P_minus
    )


    E_c = (
        field_scale
        * omega_probe
    )


    return (
        E_plus * np.conj(E_c)
        +
        np.conj(E_minus) * E_c
    )


# ============================================================
# 2. Physical Rb87 mapping
#
# This only maps the Doppler width to the normalized
# gamma_ref units.
#
# Stage-09 level structure itself is STILL the generic
# three-level model with LEVEL_OFFSET = 20 gamma_ref.
#
# Full real-Rb hyperfine structure comes in the later stage.
# ============================================================

KB = 1.380649e-23
AMU = 1.66053906660e-27

MASS_RB87 = (
    86.9091805
    * AMU
)

LAMBDA_RB87_D2 = (
    780.24e-9
)  # m


# ------------------------------------------------------------
# gamma_ref physical mapping
#
# Use Gamma/(2pi) = 6.065 MHz.
#
# Because optical Doppler shift in ordinary-frequency units is
#
#     delta_nu = v/lambda,
#
# the normalized Doppler coordinate is
#
#     q = (v/lambda) / gamma_ref_Hz.
# ------------------------------------------------------------

GAMMA_REF_HZ = (
    6.065e6
)


TEMPERATURE_K = 300.0


def thermal_speed_u(
    temperature
):
    """
    1D Maxwell parameter

        f(v) =
        exp[-(v/u)^2]
        /(sqrt(pi) u)

    with

        u = sqrt(2 k_B T / m).
    """

    if temperature <= 0:
        return 0.0

    return np.sqrt(
        2.0
        * KB
        * temperature
        / MASS_RB87
    )


def doppler_width_normalized(
    temperature
):
    """
    Return

        D = k*u/Gamma_angular

    which numerically equals

        (u/lambda) / [Gamma/(2pi)]

    in the normalized units used by Stage 09.
    """

    if temperature <= 0:
        return 0.0

    u = thermal_speed_u(
        temperature
    )

    doppler_1e_hz = (
        u
        / LAMBDA_RB87_D2
    )

    return (
        doppler_1e_hz
        / GAMMA_REF_HZ
    )


DOPPLER_D = (
    doppler_width_normalized(
        TEMPERATURE_K
    )
)


# ============================================================
# 3. Frequency scan
# ============================================================

if FAST_MODE:

    X_SCAN = np.linspace(
        -4.0,
        +4.0,
        101
    )

    # Target q-grid spacing.
    #
    # D~50 at room temperature, giving roughly 1000 points
    # over +/-4D.
    DQ_TARGET = 0.40

else:

    X_SCAN = np.linspace(
        -4.0,
        +4.0,
        161
    )

    DQ_TARGET = 0.20


VELOCITY_SPAN_U = 4.0


# ============================================================
# 4. Maxwell grid in normalized q = kv/gamma_ref
# ============================================================

def make_q_grid(
    D,
    dq_target=DQ_TARGET,
    span_u=VELOCITY_SPAN_U
):

    if D <= 0:

        return (
            np.array([0.0]),
            np.array([1.0])
        )


    qmax = (
        span_u * D
    )


    Nq = int(
        np.ceil(
            2.0
            * qmax
            / dq_target
        )
    ) + 1


    # odd number gives an exact q=0 point

    if Nq % 2 == 0:
        Nq += 1


    q = np.linspace(
        -qmax,
        +qmax,
        Nq
    )


    weight = (
        np.exp(
            -(q / D)**2
        )

        /
        (
            D
            * np.sqrt(np.pi)
        )
    )


    # compensate tiny finite +/-4u truncation

    normalization = np.trapezoid(
        weight,
        q
    )


    weight = (
        weight
        / normalization
    )


    return (
        q,
        weight
    )


Q_GRID, Q_WEIGHT = (
    make_q_grid(
        DOPPLER_D
    )
)


# ============================================================
# 5. One thermal-average point
# ============================================================

def thermal_sidebands_at_x(
    x,
    D=DOPPLER_D,
    dq_target=DQ_TARGET,
    beta=None,
    omega_pump=None,
    omega_probe=None,
    field_scale=1.0
):
    """
    Calculate

        <P_+>_v
        <P_->_v

    before Maxwell propagation / optical heterodyne.

    This is the physically clean place to perform the
    Maxwell-Boltzmann average.
    """

    # Exact Doppler-free limit

    if D <= 0:

        return engine_sidebands(
            x=x,
            kv=0.0,
            beta=beta,
            omega_pump=omega_pump,
            omega_probe=omega_probe,
            field_scale=field_scale
        )


    q, weight = make_q_grid(
        D=D,
        dq_target=dq_target
    )


    P_plus_v = np.empty(
        len(q),
        dtype=np.complex128
    )

    P_minus_v = np.empty(
        len(q),
        dtype=np.complex128
    )


    for i, kv in enumerate(q):

        (
            P_plus_v[i],
            P_minus_v[i]

        ) = engine_sidebands(
            x=x,
            kv=kv,
            beta=beta,
            omega_pump=omega_pump,
            omega_probe=omega_probe,
            field_scale=field_scale
        )


    P_plus_avg = np.trapezoid(
        P_plus_v * weight,
        q
    )


    P_minus_avg = np.trapezoid(
        P_minus_v * weight,
        q
    )


    return (
        P_plus_avg,
        P_minus_avg
    )


# ============================================================
# 6. Thermal complex MTS response
# ============================================================

def thermal_response_at_x(
    x,
    D=DOPPLER_D,
    dq_target=DQ_TARGET,
    beta=None,
    omega_pump=None,
    omega_probe=None,
    field_scale=1.0
):

    P_plus, P_minus = (
        thermal_sidebands_at_x(
            x=x,
            D=D,
            dq_target=dq_target,
            beta=beta,
            omega_pump=omega_pump,
            omega_probe=omega_probe,
            field_scale=field_scale
        )
    )


    Z = heterodyne_from_sidebands(
        P_plus=P_plus,
        P_minus=P_minus,
        omega_probe=omega_probe,
        field_scale=field_scale
    )


    return {
        "P_plus": P_plus,
        "P_minus": P_minus,
        "Z": Z
    }


# ============================================================
# 7. Analysis utilities
# ============================================================

def complex_center_slope(
    x,
    Z,
    window=0.30
):

    mask = (
        np.abs(x)
        <= window
    )


    xx = x[mask]
    ZZ = Z[mask]


    if len(xx) < 4:

        raise RuntimeError(
            "Center grid is too coarse "
            "for cubic slope fitting."
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


def demodulate(
    Z,
    phi
):

    return np.real(
        Z
        * np.exp(
            -1j * phi
        )
    )


def zero_crossing_near_zero(
    x,
    y,
    window=1.0
):

    mask = (
        np.abs(x)
        <= window
    )


    xx = x[mask]
    yy = y[mask]


    crossing = np.where(
        yy[:-1]
        * yy[1:]
        <= 0
    )[0]


    if len(crossing) == 0:

        return np.nan


    centers = (
        xx[crossing]
        + xx[crossing + 1]
    ) / 2.0


    j = crossing[
        np.argmin(
            np.abs(centers)
        )
    ]


    x1 = xx[j]
    x2 = xx[j + 1]

    y1 = yy[j]
    y2 = yy[j + 1]


    if abs(y2 - y1) < 1e-30:

        return (
            0.5
            * (
                x1 + x2
            )
        )


    return (
        x1

        -

        y1
        * (
            x2 - x1
        )

        / (
            y2 - y1
        )
    )


def nearest_extrema(
    x,
    y,
    zero
):

    maxima, _ = find_peaks(
        y
    )

    minima, _ = find_peaks(
        -y
    )


    extrema = np.sort(
        np.concatenate([
            maxima,
            minima
        ])
    )


    left = extrema[
        x[extrema] < zero
    ]

    right = extrema[
        x[extrema] > zero
    ]


    if (
        len(left) == 0
        or len(right) == 0
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
        y[ir]
        - y[il]
    )


    separation = (
        x[ir]
        - x[il]
    )


    return (
        Vpp,
        separation,
        x[il],
        x[ir]
    )


# ============================================================
# 8. Doppler-free reference scan
# ============================================================

print()
print(
    "============================================================"
)

print(
    "Stage E: validated rho^(3) + explicit thermal Doppler average"
)

print(
    "============================================================"
)


print()
print(
    "Loading Stage-09 engine from:"
)

print(
    ENGINE_PATH
)


print()
print(
    "Stage-09 parameters:"
)

print(
    "FM/gamma_ref =",
    engine.FM
)

print(
    "beta =",
    engine.BETA
)

print(
    "LEVEL_OFFSET/gamma_ref =",
    engine.LEVEL_OFFSET
)

print(
    "Omega_pump/gamma_ref =",
    engine.OMEGA_PUMP
)

print(
    "Omega_probe/gamma_ref =",
    engine.OMEGA_PROBE
)


print()
print(
    "Rb87 thermal mapping:"
)

print(
    "Temperature =",
    TEMPERATURE_K,
    "K"
)

print(
    "u =",
    thermal_speed_u(
        TEMPERATURE_K
    ),
    "m/s"
)

print(
    "1/e Doppler scale u/lambda =",
    thermal_speed_u(
        TEMPERATURE_K
    )
    / LAMBDA_RB87_D2
    / 1e6,
    "MHz"
)

print(
    "D = ku/gamma_ref =",
    DOPPLER_D
)

print(
    "Velocity q-grid points =",
    len(Q_GRID)
)

print(
    "q range =",
    Q_GRID[0],
    "to",
    Q_GRID[-1]
)

print(
    "q spacing =",
    (
        Q_GRID[1] - Q_GRID[0]
        if len(Q_GRID) > 1
        else 0.0
    )
)


print()
print(
    "IMPORTANT:"
)

print(
    "The velocity width is mapped to room-temperature Rb87,"
)

print(
    "but Stage-09 is still the generic three-level model."
)

print(
    "This is a Doppler bridge stage, not yet the final full-Rb model."
)


print()
print(
    "Calculating Doppler-free reference..."
)


P_PLUS_0 = np.zeros(
    len(X_SCAN),
    dtype=np.complex128
)

P_MINUS_0 = np.zeros_like(
    P_PLUS_0
)

Z_0 = np.zeros_like(
    P_PLUS_0
)


for i, x in enumerate(
    X_SCAN
):

    if (
        i % max(
            len(X_SCAN) // 10,
            1
        )
        == 0
    ):

        print(
            f"  no Doppler "
            f"{i+1:3d}/{len(X_SCAN)}"
        )


    P_PLUS_0[i], P_MINUS_0[i] = (
        engine_sidebands(
            x=x,
            kv=0.0
        )
    )


    Z_0[i] = (
        heterodyne_from_sidebands(
            P_PLUS_0[i],
            P_MINUS_0[i]
        )
    )


# ============================================================
# 9. Thermal scan
# ============================================================

print()
print(
    "Calculating Maxwell-averaged third-order response..."
)


P_PLUS_D = np.zeros_like(
    P_PLUS_0
)

P_MINUS_D = np.zeros_like(
    P_PLUS_0
)

Z_D = np.zeros_like(
    P_PLUS_0
)


for i, x in enumerate(
    X_SCAN
):

    print(
        f"  Doppler "
        f"{i+1:3d}/{len(X_SCAN)}"
        f"   x={x:+.3f}"
    )


    result = thermal_response_at_x(
        x=x
    )


    P_PLUS_D[i] = (
        result["P_plus"]
    )

    P_MINUS_D[i] = (
        result["P_minus"]
    )

    Z_D[i] = (
        result["Z"]
    )


# ============================================================
# 10. Demodulation analysis
# ============================================================

SLOPE_0_COMPLEX = (
    complex_center_slope(
        X_SCAN,
        Z_0
    )
)


SLOPE_D_COMPLEX = (
    complex_center_slope(
        X_SCAN,
        Z_D
    )
)


PHI_0 = np.angle(
    SLOPE_0_COMPLEX
)


PHI_D = np.angle(
    SLOPE_D_COMPLEX
)


# ------------------------------------------------------------
# For direct physical comparison use ONE fixed mixer phase:
#
# the thermally optimized phase.
# ------------------------------------------------------------

S_0_FIXED = demodulate(
    Z_0,
    PHI_D
)


S_D_FIXED = demodulate(
    Z_D,
    PHI_D
)


# separately optimized signals for metric extraction

S_0_OPT = demodulate(
    Z_0,
    PHI_0
)


S_D_OPT = demodulate(
    Z_D,
    PHI_D
)


ZERO_0 = zero_crossing_near_zero(
    X_SCAN,
    S_0_OPT
)


ZERO_D = zero_crossing_near_zero(
    X_SCAN,
    S_D_OPT
)


(
    VPP_0,
    SEP_0,
    LEFT_0,
    RIGHT_0

) = nearest_extrema(
    X_SCAN,
    S_0_OPT,
    ZERO_0
)


(
    VPP_D,
    SEP_D,
    LEFT_D,
    RIGHT_D

) = nearest_extrema(
    X_SCAN,
    S_D_OPT,
    ZERO_D
)


SLOPE_FIXED_0 = abs(
    np.real(
        SLOPE_0_COMPLEX
        * np.exp(
            -1j * PHI_D
        )
    )
)


SLOPE_FIXED_D = abs(
    np.real(
        SLOPE_D_COMPLEX
        * np.exp(
            -1j * PHI_D
        )
    )
)


# far-wing background

EDGE_MASK = (
    np.abs(X_SCAN)
    > 3.5
)


FARWING_RATIO = (
    np.mean(
        np.abs(
            S_D_OPT[
                EDGE_MASK
            ]
        )
    )

    /

    max(
        np.max(
            np.abs(
                S_D_OPT
            )
        ),
        1e-30
    )
)


# ============================================================
# 11. Ordinary linear absorption control
#
# This is NOT the MTS model.
#
# It is only a control curve showing how strong the ordinary
# intermediate-state Doppler background is in the same
# velocity distribution.
# ============================================================

def ordinary_absorption(
    x_array,
    D=DOPPLER_D
):

    q, weight = make_q_grid(
        D=D,
        dq_target=DQ_TARGET
    )


    output = np.zeros(
        len(x_array)
    )


    for i, x in enumerate(
        x_array
    ):

        delta_ab = (
            engine.LEVEL_OFFSET
            + x
        )


        response = (
            engine.GAMMA_AB

            /

            (
                engine.GAMMA_AB**2

                +
                (
                    delta_ab
                    - q
                )**2
            )
        )


        output[i] = np.trapezoid(
            response * weight,
            q
        )


    return output


ABS_D = ordinary_absorption(
    X_SCAN
)


# Normalize to the ordinary Doppler absorption
# at the center of this scan for visual comparison.

ABS_D_NORM = (
    ABS_D
    / np.max(
        ABS_D
    )
)


# ============================================================
# 12. beta = 0 null test after Doppler averaging
# ============================================================

TEST_X = (
    RIGHT_D
    if np.isfinite(RIGHT_D)
    else 0.3
)


NORMAL_TEST = thermal_response_at_x(
    TEST_X,
    beta=engine.BETA
)


NULL_TEST = thermal_response_at_x(
    TEST_X,
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
        1e-30
    )
)


# ============================================================
# 13. Third-order scaling after Doppler averaging
# ============================================================

SCALING_S = np.array([
    0.50,
    0.75,
    1.00
])


P3_SCALE_VALUES = []
Z_SCALE_VALUES = []


if RUN_SCALING_CHECK:

    print()
    print(
        "Checking third-order scaling after Doppler averaging..."
    )


    for s in SCALING_S:

        print(
            "  field scale =",
            s
        )


        result = thermal_response_at_x(
            TEST_X,
            field_scale=s
        )


        P3_SCALE_VALUES.append(
            max(
                abs(
                    result["P_plus"]
                ),
                abs(
                    result["P_minus"]
                )
            )
        )


        Z_SCALE_VALUES.append(
            abs(
                result["Z"]
            )
        )


P3_SCALE_VALUES = np.asarray(
    P3_SCALE_VALUES
)

Z_SCALE_VALUES = np.asarray(
    Z_SCALE_VALUES
)


# ============================================================
# 14. Numerical q-grid convergence
# ============================================================

Q_CONVERGENCE_ERROR = np.nan
BOUNDARY_ERROR = np.nan


if RUN_CONVERGENCE_CHECK:

    print()
    print(
        "Checking Doppler integration convergence..."
    )


    TEST_POINTS = np.array([
        -0.5,
        0.0,
        +0.5
    ])


    coarse = []

    fine = []


    for x in TEST_POINTS:

        rc = thermal_response_at_x(
            x,
            dq_target=DQ_TARGET
        )


        rf = thermal_response_at_x(
            x,
            dq_target=DQ_TARGET / 2.0
        )


        coarse.append(
            rc["Z"]
        )

        fine.append(
            rf["Z"]
        )


    coarse = np.asarray(
        coarse
    )

    fine = np.asarray(
        fine
    )


    Q_CONVERGENCE_ERROR = (
        np.linalg.norm(
            coarse - fine
        )

        /

        max(
            np.linalg.norm(
                fine
            ),
            1e-30
        )
    )


    # --------------------------------------------------------
    # +/-4u versus +/-5u boundary check at x=0
    # --------------------------------------------------------

    def thermal_with_span(
        x,
        span
    ):

        D = DOPPLER_D


        qmax = (
            span * D
        )


        Nq = int(
            np.ceil(
                2.0
                * qmax
                / DQ_TARGET
            )
        ) + 1


        if Nq % 2 == 0:
            Nq += 1


        q = np.linspace(
            -qmax,
            qmax,
            Nq
        )


        w = (
            np.exp(
                -(q / D)**2
            )

            /
            (
                D
                * np.sqrt(np.pi)
            )
        )


        w /= np.trapezoid(
            w,
            q
        )


        Pp = np.empty(
            Nq,
            dtype=complex
        )

        Pm = np.empty(
            Nq,
            dtype=complex
        )


        for j, kv in enumerate(q):

            Pp[j], Pm[j] = (
                engine_sidebands(
                    x=x,
                    kv=kv
                )
            )


        Pp_avg = np.trapezoid(
            Pp * w,
            q
        )

        Pm_avg = np.trapezoid(
            Pm * w,
            q
        )


        return (
            heterodyne_from_sidebands(
                Pp_avg,
                Pm_avg
            )
        )


    Z4 = thermal_with_span(
        0.0,
        4.0
    )


    Z5 = thermal_with_span(
        0.0,
        5.0
    )


    BOUNDARY_ERROR = (
        abs(
            Z4 - Z5
        )

        /

        max(
            abs(Z5),
            1e-30
        )
    )


# ============================================================
# 15. Optional Doppler-width scan
# ============================================================

D_SCAN = np.array([
    0.0,
    10.0,
    20.0,
    50.0,
    100.0
])


D_SCAN_SLOPE = []


if RUN_DOPPLER_WIDTH_SCAN:

    print()
    print(
        "Running optional Doppler-width scan..."
    )


    X_LOCAL = np.linspace(
        -0.3,
        +0.3,
        7
    )


    for D_i in D_SCAN:

        Zi = []


        for x in X_LOCAL:

            Zi.append(
                thermal_response_at_x(
                    x,
                    D=D_i
                )["Z"]
            )


        Zi = np.asarray(
            Zi
        )


        slope_i = complex_center_slope(
            X_LOCAL,
            Zi,
            window=0.3
        )


        D_SCAN_SLOPE.append(
            abs(
                slope_i
            )
        )


    D_SCAN_SLOPE = np.asarray(
        D_SCAN_SLOPE
    )


# ============================================================
# 16. Console diagnostics
# ============================================================

print()
print(
    "============================================================"
)

print(
    "Stage E results"
)

print(
    "============================================================"
)


print()
print(
    "Doppler-free:"
)

print(
    "complex center slope =",
    SLOPE_0_COMPLEX
)

print(
    "|center slope| =",
    abs(
        SLOPE_0_COMPLEX
    )
)

print(
    "optimum phase =",
    np.degrees(
        PHI_0
    ),
    "deg"
)

print(
    "zero crossing =",
    ZERO_0
)

print(
    "nearest-extrema Vpp =",
    VPP_0
)

print(
    "nearest-extrema separation =",
    SEP_0
)


print()
print(
    "Doppler averaged:"
)

print(
    "complex center slope =",
    SLOPE_D_COMPLEX
)

print(
    "|center slope| =",
    abs(
        SLOPE_D_COMPLEX
    )
)

print(
    "optimum phase =",
    np.degrees(
        PHI_D
    ),
    "deg"
)

print(
    "zero crossing =",
    ZERO_D
)

print(
    "nearest-extrema Vpp =",
    VPP_D
)

print(
    "nearest-extrema separation =",
    SEP_D
)

print(
    "far-wing / peak =",
    FARWING_RATIO
)


print()
print(
    "Common-phase comparison:"
)

print(
    "Doppler-free slope at thermal optimum phase =",
    SLOPE_FIXED_0
)

print(
    "Doppler-averaged slope =",
    SLOPE_FIXED_D
)

print(
    "slope ratio Doppler/no-Doppler =",
    (
        SLOPE_FIXED_D
        /
        max(
            SLOPE_FIXED_0,
            1e-30
        )
    )
)


print()
print(
    "Structural validation:"
)

print(
    "test x =",
    TEST_X
)

print(
    "beta=0 null-test ratio =",
    NULL_RATIO
)


if RUN_SCALING_CHECK:

    print()
    print(
        "Third-order scaling after thermal averaging:"
    )

    print(
        f"{'s':>8} "
        f"{'|P3|/s^3':>18} "
        f"{'|Z|/s^4':>18}"
    )


    for (
        s,
        p,
        z
    ) in zip(
        SCALING_S,
        P3_SCALE_VALUES,
        Z_SCALE_VALUES
    ):

        print(
            f"{s:8.3f} "
            f"{p/s**3:18.10e} "
            f"{z/s**4:18.10e}"
        )


if RUN_CONVERGENCE_CHECK:

    print()
    print(
        "Doppler numerical convergence:"
    )

    print(
        "dq -> dq/2 relative Z error =",
        Q_CONVERGENCE_ERROR
    )

    print(
        "+/-4u -> +/-5u boundary error =",
        BOUNDARY_ERROR
    )


print()
print(
    "All finite =",
    (
        np.all(
            np.isfinite(
                P_PLUS_D
            )
        )

        and

        np.all(
            np.isfinite(
                P_MINUS_D
            )
        )

        and

        np.all(
            np.isfinite(
                Z_D
            )
        )
    )
)


print(
    "============================================================"
)


# ============================================================
# 17. Figure 1
#
# Maxwell velocity distribution in normalized q
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    Q_GRID,
    Q_WEIGHT
)


plt.axvline(
    +engine.LEVEL_OFFSET,
    linestyle="--",
    linewidth=0.8,
    label=r"$q=+\Delta_{ab}(x=0)$"
)

plt.axvline(
    -engine.LEVEL_OFFSET,
    linestyle="--",
    linewidth=0.8,
    label=r"$q=-\Delta_{ab}(x=0)$"
)


plt.xlabel(
    r"Normalized Doppler shift "
    r"$q=kv/\gamma_{\rm ref}$"
)

plt.ylabel(
    "Maxwell probability density"
)

plt.title(
    "Thermal velocity distribution"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 2
#
# Generated sidebands:
# common normalization preserves the amplitude change.
# ============================================================

SIDEBAND_SCALE = max(
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
    1e-30
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    X_SCAN,
    np.abs(
        P_PLUS_0
    ) / SIDEBAND_SCALE,
    label=r"No Doppler: $\omega+\Omega_m$"
)


plt.plot(
    X_SCAN,
    np.abs(
        P_MINUS_0
    ) / SIDEBAND_SCALE,
    label=r"No Doppler: $\omega-\Omega_m$"
)


plt.plot(
    X_SCAN,
    np.abs(
        P_PLUS_D
    ) / SIDEBAND_SCALE,
    linestyle="--",
    label=r"Doppler: $\omega+\Omega_m$"
)


plt.plot(
    X_SCAN,
    np.abs(
        P_MINUS_D
    ) / SIDEBAND_SCALE,
    linestyle="--",
    label=r"Doppler: $\omega-\Omega_m$"
)


plt.axvline(
    0,
    linestyle="--",
    linewidth=0.8
)


plt.xlabel(
    r"$x/\gamma_{\rm ref}$"
)

plt.ylabel(
    "Third-order sideband polarization "
    "(common normalization)"
)

plt.title(
    r"Thermal averaging of $P_\pm^{(3)}$"
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
# Complex heterodyne quadratures
# ============================================================

Z_SCALE = max(
    np.max(
        np.abs(
            Z_0
        )
    ),
    1e-30
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    X_SCAN,
    np.real(Z_0) / Z_SCALE,
    label="No Doppler Re(Z)"
)


plt.plot(
    X_SCAN,
    np.imag(Z_0) / Z_SCALE,
    label="No Doppler Im(Z)"
)


plt.plot(
    X_SCAN,
    np.real(Z_D) / Z_SCALE,
    linestyle="--",
    label="Doppler Re(Z)"
)


plt.plot(
    X_SCAN,
    np.imag(Z_D) / Z_SCALE,
    linestyle="--",
    label="Doppler Im(Z)"
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
    r"$x/\gamma_{\rm ref}$"
)

plt.ylabel(
    "RF response "
    "(common no-Doppler normalization)"
)

plt.title(
    "Doppler effect on complex MTS heterodyne response"
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
# MTS error signal:
# SAME mixer phase for both curves.
# ============================================================

ERROR_SCALE = max(
    np.max(
        np.abs(
            S_0_FIXED
        )
    ),
    1e-30
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    X_SCAN,
    S_0_FIXED / ERROR_SCALE,
    label="No Doppler"
)


plt.plot(
    X_SCAN,
    S_D_FIXED / ERROR_SCALE,
    linestyle="--",
    label="Maxwell averaged"
)


plt.axhline(
    0,
    linewidth=0.8
)


plt.axvline(
    0,
    linestyle="--",
    linewidth=0.8,
    label="Two-photon resonance"
)


plt.xlabel(
    r"$x/\gamma_{\rm ref}$"
)

plt.ylabel(
    "MTS error signal "
    "(common normalization)"
)

plt.title(
    "Doppler-free versus thermal MTS"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 5
#
# Ordinary Doppler background versus MTS.
#
# These are DIFFERENT physical observables, so each vertical
# scale is normalized separately. The comparison concerns
# baseline/linewidth, not absolute amplitudes.
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(8, 5)
)


line1 = ax1.plot(
    X_SCAN,
    ABS_D_NORM,
    label="Ordinary Doppler absorption"
)


ax1.set_xlabel(
    r"Two-photon-centered detuning "
    r"$x/\gamma_{\rm ref}$"
)


ax1.set_ylabel(
    "Ordinary absorption "
    "(self-normalized)"
)


ax2 = ax1.twinx()


MTS_SELF_SCALE = max(
    np.max(
        np.abs(
            S_D_OPT
        )
    ),
    1e-30
)


line2 = ax2.plot(
    X_SCAN,
    S_D_OPT / MTS_SELF_SCALE,
    label="Thermal MTS"
)


ax2.set_ylabel(
    "MTS signal "
    "(self-normalized)"
)


ax1.axvline(
    0,
    linestyle="--",
    linewidth=0.8
)


lines = (
    line1
    + line2
)


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
    "Ordinary Doppler background versus "
    "third-order MTS"
)


ax1.grid(
    alpha=0.3
)


plt.tight_layout()


# ============================================================
# Figure 6
#
# Optional D scan
# ============================================================

if RUN_DOPPLER_WIDTH_SCAN:

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        D_SCAN,
        D_SCAN_SLOPE
        / max(
            D_SCAN_SLOPE[0],
            1e-30
        ),
        marker="o"
    )


    plt.xlabel(
        r"$ku/\gamma_{\rm ref}$"
    )


    plt.ylabel(
        "Center-slope ratio"
    )


    plt.title(
        "Validated rho3 MTS versus Doppler width"
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


plt.show()