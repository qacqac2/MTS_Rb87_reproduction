# ============================================================
# 10_mts_rho3_vs_qutip.py
#
# Cross validation:
#
#   explicit perturbation rho^(3)
#           VS
#   full QuTiP master equation -> weak-field extrapolation a3
#
# Same three-level cascade model, same EOM convention,
# same decay model, same phase convention.
#
# Target:
#
#   P_±(s) = a3_± s^3 + a5_± s^5 + ...
#
# Therefore:
#
#   P_±(s) / s^3 = a3_± + a5_± s^2 + ...
#
# We extrapolate s -> 0 and compare a3 with explicit rho^(3).
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from scipy.special import jv


# ============================================================
# 0. Calculation mode
# ============================================================

# First run with True.
# If everything works, change to False for the final validation.
FAST_MODE = False

if FAST_MODE:
    X_SCAN = np.linspace(-1.2, 1.2, 9)

    FIELD_SCALES = np.array([
        0.50,
        0.70,
        0.90,
        1.10
    ])

    N_PHASE = 4

    N_SETTLE_PERIODS = 6
    N_SAMPLE_PERIODS = 2
    SAMPLES_PER_PERIOD = 64

else:
    X_SCAN = np.linspace(-1.5, 1.5, 17)

    FIELD_SCALES = np.array([
        0.40,
        0.55,
        0.70,
        0.85,
        1.00,
        1.15
    ])

    N_PHASE = 8

    N_SETTLE_PERIODS = 10
    N_SAMPLE_PERIODS = 4
    SAMPLES_PER_PERIOD = 128


# ============================================================
# 1. Normalized physical parameters
#
# Same convention as Stage 09
# ============================================================

GAMMA_REF = 1.0

# Modulation frequency
OMEGA_M = 0.6 * GAMMA_REF

# EOM modulation index
BETA = 0.28

# At two-photon resonance:
#
# (omega_ac / 2 - omega_ab) / gamma_ref = 20
#
LEVEL_OFFSET = 20.0

# Weak optical fields used in Stage 09
OMEGA_PUMP = 0.15 * GAMMA_REF
OMEGA_PROBE = 0.03 * GAMMA_REF


# ============================================================
# 2. Three-level atomic basis
#
# |a> -> |b> -> |c>
# ============================================================

a = qt.basis(3, 0)
b = qt.basis(3, 1)
c = qt.basis(3, 2)

Pa = a * a.dag()
Pb = b * b.dag()
Pc = c * c.dag()

# Raising dipole operator
#
# |a> -> |b>
# |b> -> |c>
#
SIGMA_PLUS = (
    b * a.dag()
    +
    c * b.dag()
)

SIGMA_MINUS = SIGMA_PLUS.dag()


# ============================================================
# 3. Dissipation
#
# We choose the Lindblad model so that approximately
#
# gamma_ab = gamma_bc = gamma_ac = 1
#
# matching the normalized Stage-09 model.
# ============================================================

GAMMA_BA = 1.0
GAMMA_CB = 1.0

# Additional pure dephasing of |a>.
#
# With
#
# Gamma_ba = Gamma_cb = 1
#
# the population-decay contributions are
#
# gamma_ab = 0.5
# gamma_bc = 1.0
# gamma_ac = 0.5
#
# Adding collapse sqrt(1)|a><a| gives another 0.5
# to coherences involving |a>, producing
#
# gamma_ab = gamma_bc = gamma_ac = 1.
#
GAMMA_PHI_A = 1.0

C_BA = np.sqrt(GAMMA_BA) * a * b.dag()
C_CB = np.sqrt(GAMMA_CB) * b * c.dag()
C_PHI = np.sqrt(GAMMA_PHI_A) * Pa

C_OPS = [
    C_BA,
    C_CB,
    C_PHI
]

RHO0 = Pa


# ============================================================
# 4. Bessel coefficients
# ============================================================

J0 = jv(0, BETA)
J1 = jv(1, BETA)


# ============================================================
# 5. Atomic detuning
#
# x is measured relative to two-photon resonance:
#
# x = omega - omega_ac/2
#
# Then
#
# Delta_ab = LEVEL_OFFSET + x
# Delta_ac = 2x
# ============================================================

def build_H0(x):

    delta_ab = LEVEL_OFFSET + x
    delta_ac = 2.0 * x

    H0 = (
        -delta_ab * Pb
        -delta_ac * Pc
    )

    return H0


# ============================================================
# 6. Optical modes
#
# Positive-frequency convention:
#
# Omega(t) = sum_m Omega_m exp(-i m Omega_M t)
#
# EOM:
#
# m = 0   : J0
# m = +1  : +J1
# m = -1  : -J1
#
# This matches Stage 09.
# ============================================================

def optical_modes(scale=1.0,
                  pump_phase=0.0,
                  beta=BETA):

    j0 = jv(0, beta)
    j1 = jv(1, beta)

    ph = np.exp(1j * pump_phase)

    modes = {}

    # lower sideband
    modes[-1] = (
        -scale
        * OMEGA_PUMP
        * j1
        * ph
    )

    # carrier:
    # pump carrier + unmodulated probe
    modes[0] = scale * (
        OMEGA_PROBE
        +
        OMEGA_PUMP
        * j0
        * ph
    )

    # upper sideband
    modes[+1] = (
        scale
        * OMEGA_PUMP
        * j1
        * ph
    )

    return modes


# ============================================================
# 7. Hamiltonian Fourier components
#
# H(t) =
#
# 1/2 [
#     Omega(t) Sigma_+
#     +
#     Omega*(t) Sigma_-
# ]
#
# For harmonic m:
#
# H_m =
#
# 1/2 [
#     Omega_m Sigma_+
#     +
#     Omega_-m* Sigma_-
# ]
# ============================================================

def interaction_harmonics(scale=1.0,
                          pump_phase=0.0,
                          beta=BETA):

    modes = optical_modes(
        scale=scale,
        pump_phase=pump_phase,
        beta=beta
    )

    Hm = {}

    for m in [-1, 0, +1]:

        Om = modes.get(m, 0.0)
        Ominus = modes.get(-m, 0.0)

        Hm[m] = 0.5 * (
            Om * SIGMA_PLUS
            +
            np.conj(Ominus) * SIGMA_MINUS
        )

    return Hm


# ============================================================
# 8. Matrix/vector utilities
# ============================================================

DIM = 3
SUPER_DIM = DIM * DIM

IDENTITY_SUPER = np.eye(
    SUPER_DIM,
    dtype=complex
)

RHO0_VEC = (
    qt.operator_to_vector(RHO0)
    .full()
    .ravel()
)

TRACE_ROW = (
    qt.operator_to_vector(qt.qeye(DIM))
    .dag()
    .full()
    .ravel()
)


def vec_to_matrix(v):

    # QuTiP uses column stacking
    return np.asarray(v).reshape(
        (DIM, DIM),
        order="F"
    )


def polarization_from_vec(v):

    rho = vec_to_matrix(v)

    return np.trace(
        SIGMA_PLUS.full() @ rho
    )


# ============================================================
# 9. Solve harmonic Liouvillian equation
#
# (-i n Omega_m I - L0) rho_n = RHS
#
# n = 0 is singular because of trace conservation.
# Higher perturbation orders must satisfy:
#
# Tr rho^(r) = 0
# ============================================================

def solve_harmonic(
        L0,
        harmonic_n,
        rhs):

    matrix = (
        -1j
        * harmonic_n
        * OMEGA_M
        * IDENTITY_SUPER
        -
        L0
    )

    rhs = rhs.copy()

    if harmonic_n == 0:

        matrix = matrix.copy()

        # Replace one equation by trace = 0
        matrix[0, :] = TRACE_ROW
        rhs[0] = 0.0

    return np.linalg.solve(
        matrix,
        rhs
    )


# ============================================================
# 10. Explicit perturbative rho^(3)
# ============================================================

def rho3_single_phase(
        x,
        pump_phase):

    H0 = build_H0(x)

    L0_q = qt.liouvillian(
        H0,
        C_OPS
    )

    L0 = L0_q.full()

    Hm = interaction_harmonics(
        scale=1.0,
        pump_phase=pump_phase,
        beta=BETA
    )

    V = {}

    for m in [-1, 0, +1]:

        Vq = -1j * (
            qt.spre(Hm[m])
            -
            qt.spost(Hm[m])
        )

        V[m] = Vq.full()

    # zeroth order
    previous = {
        0: RHO0_VEC
    }

    # Recursive perturbation expansion
    for order in range(1, 4):

        current = {}

        for n in range(
                -order,
                order + 1):

            rhs = np.zeros(
                SUPER_DIM,
                dtype=complex
            )

            for m in [-1, 0, +1]:

                previous_harmonic = n - m

                if previous_harmonic in previous:

                    rhs += (
                        V[m]
                        @ previous[
                            previous_harmonic
                        ]
                    )

            current[n] = solve_harmonic(
                L0,
                n,
                rhs
            )

        previous = current

    rho3_plus = previous[+1]
    rho3_minus = previous[-1]

    Pplus = polarization_from_vec(
        rho3_plus
    )

    Pminus = polarization_from_vec(
        rho3_minus
    )

    return Pplus, Pminus


# ============================================================
# 11. Pump optical-phase cycling
#
# Desired FWM term:
#
# pump * pump* * probe
#
# has zero net global pump phase.
#
# Averaging over pump phase suppresses pump-only
# and wrong-wavevector contributions.
# ============================================================

def rho3_phase_cycled(
        x,
        n_phase=N_PHASE):

    phases = np.linspace(
        0.0,
        2.0 * np.pi,
        n_phase,
        endpoint=False
    )

    Pplus = 0.0j
    Pminus = 0.0j

    for phi in phases:

        pp, pm = rho3_single_phase(
            x,
            phi
        )

        Pplus += pp
        Pminus += pm

    return (
        Pplus / n_phase,
        Pminus / n_phase
    )


# ============================================================
# 12. Full QuTiP master-equation solver
# ============================================================

def full_qutip_single_phase(
        x,
        scale,
        pump_phase,
        beta=BETA):

    H0 = build_H0(x)

    modes = optical_modes(
        scale=scale,
        pump_phase=pump_phase,
        beta=beta
    )

    def omega_t(t):

        return (
            modes[-1]
            * np.exp(+1j * OMEGA_M * t)
            +
            modes[0]
            +
            modes[+1]
            * np.exp(-1j * OMEGA_M * t)
        )

    def omega_t_conj(t):

        return np.conj(
            omega_t(t)
        )

    # QobjEvo avoids old f(t,args) callback convention
    H = qt.QobjEvo([
        H0,
        [
            0.5 * SIGMA_PLUS,
            omega_t
        ],
        [
            0.5 * SIGMA_MINUS,
            omega_t_conj
        ]
    ])

    period = 2.0 * np.pi / OMEGA_M

    settle_time = (
        N_SETTLE_PERIODS
        * period
    )

    # --------------------------------------------------------
    # First reach periodic steady state
    # --------------------------------------------------------

    settle_result = qt.mesolve(
        H,
        RHO0,
        [
            0.0,
            settle_time
        ],
        C_OPS,
        e_ops=[],
        options={
            "store_final_state": True,
            "progress_bar": "",
            "atol": 1e-11,
            "rtol": 1e-9,
            "nsteps": 100000
        }
    )

    rho_start = (
        settle_result.final_state
    )

    # --------------------------------------------------------
    # Sample an integer number of modulation periods
    # --------------------------------------------------------

    Nsample = (
        N_SAMPLE_PERIODS
        * SAMPLES_PER_PERIOD
    )

    times = (
        settle_time
        +
        np.arange(Nsample)
        * period
        / SAMPLES_PER_PERIOD
    )

    result = qt.mesolve(
        H,
        rho_start,
        times,
        C_OPS,
        e_ops=[
            SIGMA_PLUS
        ],
        options={
            "progress_bar": "",
            "atol": 1e-11,
            "rtol": 1e-9,
            "nsteps": 100000
        }
    )

    P_t = np.asarray(
        result.expect[0],
        dtype=complex
    )

    # --------------------------------------------------------
    # Fourier components:
    #
    # P(t) = sum_n P_n exp(-i n Omega t)
    #
    # therefore
    #
    # P_n = <P exp(+i n Omega t)>
    # --------------------------------------------------------

    Pplus = np.mean(
        P_t
        * np.exp(
            +1j
            * OMEGA_M
            * times
        )
    )

    Pminus = np.mean(
        P_t
        * np.exp(
            -1j
            * OMEGA_M
            * times
        )
    )

    return Pplus, Pminus


# ============================================================
# 13. Full-master-equation phase cycling
# ============================================================

def full_qutip_phase_cycled(
        x,
        scale,
        beta=BETA,
        n_phase=N_PHASE):

    phases = np.linspace(
        0.0,
        2.0 * np.pi,
        n_phase,
        endpoint=False
    )

    Pplus = 0.0j
    Pminus = 0.0j

    for phi in phases:

        pp, pm = full_qutip_single_phase(
            x=x,
            scale=scale,
            pump_phase=phi,
            beta=beta
        )

        Pplus += pp
        Pminus += pm

    return (
        Pplus / n_phase,
        Pminus / n_phase
    )


# ============================================================
# 14. Extract a3 from full master equation
#
# P(s)/s^3 =
#
# a3 + a5 s^2 + a7 s^4 + ...
# ============================================================

def extrapolate_a3(
        scales,
        values):

    scales = np.asarray(
        scales,
        dtype=float
    )

    values = np.asarray(
        values,
        dtype=complex
    )

    u = scales ** 2

    y = (
        values
        /
        scales ** 3
    )

    # fit:
    #
    # y = a3 + a5 u + a7 u^2
    #
    X = np.column_stack([
        np.ones_like(u),
        u,
        u ** 2
    ])

    coefficients, _, _, _ = (
        np.linalg.lstsq(
            X,
            y,
            rcond=None
        )
    )

    a3 = coefficients[0]

    predicted_y = (
        X
        @ coefficients
    )

    predicted_values = (
        scales ** 3
        * predicted_y
    )

    denominator = max(
        np.linalg.norm(values),
        1e-30
    )

    residual = (
        np.linalg.norm(
            values
            -
            predicted_values
        )
        /
        denominator
    )

    return (
        a3,
        coefficients,
        residual
    )


# ============================================================
# 15. Complex comparison
# ============================================================

def compare_complex(
        reference,
        test):

    reference = np.asarray(
        reference,
        dtype=complex
    )

    test = np.asarray(
        test,
        dtype=complex
    )

    raw_error = (
        np.linalg.norm(
            test - reference
        )
        /
        max(
            np.linalg.norm(reference),
            1e-30
        )
    )

    # Best single complex scale factor:
    #
    # test ~= C * reference
    #
    C = (
        np.vdot(
            reference,
            test
        )
        /
        max(
            np.vdot(
                reference,
                reference
            ).real,
            1e-30
        )
    )

    shape_error = (
        np.linalg.norm(
            test
            -
            C * reference
        )
        /
        max(
            np.linalg.norm(test),
            1e-30
        )
    )

    return (
        raw_error,
        C,
        shape_error
    )


# ============================================================
# 16. Main calculation
# ============================================================

print()
print("=" * 70)
print("Stage D-2C: rho^(3) VS full QuTiP weak-field validation")
print("=" * 70)

print()
print("Mode:")
print("FAST_MODE =", FAST_MODE)

print()
print("Three-level model:")
print("gamma_ref =", GAMMA_REF)
print("Omega_m/gamma =", OMEGA_M / GAMMA_REF)
print("level offset/gamma =", LEVEL_OFFSET)

print()
print("EOM:")
print("beta =", BETA)
print("J0 =", J0)
print("J1 =", J1)

print()
print("Base Rabi frequencies:")
print("Omega_pump/gamma =", OMEGA_PUMP)
print("Omega_probe/gamma =", OMEGA_PROBE)

print()
print("Weak-field scaling points:")
print(FIELD_SCALES)

print()
print("Phase-cycle points =", N_PHASE)

print()
print("Calculating explicit rho^(3)...")

rho3_plus = []
rho3_minus = []

for i, x in enumerate(X_SCAN):

    print(
        f"rho3 {i+1:2d}/{len(X_SCAN)}"
        f"   x = {x:+.4f}"
    )

    pp, pm = rho3_phase_cycled(
        x
    )

    rho3_plus.append(pp)
    rho3_minus.append(pm)

rho3_plus = np.asarray(
    rho3_plus
)

rho3_minus = np.asarray(
    rho3_minus
)


# ============================================================
# 17. Full QuTiP scans
# ============================================================

qutip_a3_plus = []
qutip_a3_minus = []

fit_res_plus = []
fit_res_minus = []

# save scale dependence at the point closest to x=0
ic = np.argmin(
    np.abs(X_SCAN)
)

center_full_plus = None
center_full_minus = None


print()
print("Calculating full QuTiP weak-field scans...")

for ix, x in enumerate(X_SCAN):

    print()
    print(
        f"x point {ix+1}/{len(X_SCAN)}"
        f" : x = {x:+.4f}"
    )

    vals_plus = []
    vals_minus = []

    for s in FIELD_SCALES:

        print(
            f"    field scale = {s:.3f}"
        )

        pp, pm = full_qutip_phase_cycled(
            x=x,
            scale=s
        )

        vals_plus.append(pp)
        vals_minus.append(pm)

    vals_plus = np.asarray(
        vals_plus
    )

    vals_minus = np.asarray(
        vals_minus
    )

    a3p, _, resp = extrapolate_a3(
        FIELD_SCALES,
        vals_plus
    )

    a3m, _, resm = extrapolate_a3(
        FIELD_SCALES,
        vals_minus
    )

    qutip_a3_plus.append(a3p)
    qutip_a3_minus.append(a3m)

    fit_res_plus.append(resp)
    fit_res_minus.append(resm)

    if ix == ic:

        center_full_plus = vals_plus.copy()
        center_full_minus = vals_minus.copy()


qutip_a3_plus = np.asarray(
    qutip_a3_plus
)

qutip_a3_minus = np.asarray(
    qutip_a3_minus
)

fit_res_plus = np.asarray(
    fit_res_plus
)

fit_res_minus = np.asarray(
    fit_res_minus
)


# ============================================================
# 18. Heterodyne combination
#
# Probe carrier is taken real:
#
# Z ~ Eprobe* [
#       P_(+1)
#       +
#       P_(-1)^*
#     ]
# ============================================================

Z_rho3 = (
    OMEGA_PROBE
    * (
        rho3_plus
        +
        np.conj(
            rho3_minus
        )
    )
)

Z_qutip = (
    OMEGA_PROBE
    * (
        qutip_a3_plus
        +
        np.conj(
            qutip_a3_minus
        )
    )
)


# ============================================================
# 19. Comparison diagnostics
# ============================================================

err_p, Cp, shape_p = (
    compare_complex(
        rho3_plus,
        qutip_a3_plus
    )
)

err_m, Cm, shape_m = (
    compare_complex(
        rho3_minus,
        qutip_a3_minus
    )
)

err_z, Cz, shape_z = (
    compare_complex(
        Z_rho3,
        Z_qutip
    )
)


print()
print("=" * 70)
print("Cross-validation results")
print("=" * 70)

print()
print("Upper generated sideband:")
print(
    "raw relative L2 error =",
    err_p
)
print(
    "best complex scale =",
    Cp
)
print(
    "shape error after complex scaling =",
    shape_p
)

print()
print("Lower generated sideband:")
print(
    "raw relative L2 error =",
    err_m
)
print(
    "best complex scale =",
    Cm
)
print(
    "shape error after complex scaling =",
    shape_m
)

print()
print("Complex heterodyne signal Z:")
print(
    "raw relative L2 error =",
    err_z
)
print(
    "best complex scale =",
    Cz
)
print(
    "shape error after complex scaling =",
    shape_z
)

print()
print("Weak-field extrapolation residual:")
print(
    "max upper-sideband fit residual =",
    np.max(fit_res_plus)
)
print(
    "max lower-sideband fit residual =",
    np.max(fit_res_minus)
)


# ============================================================
# 20. Effective scaling exponent at x ~= 0
# ============================================================

abs_center = np.abs(
    center_full_plus
)

valid = (
    abs_center > 1e-20
)

if np.sum(valid) >= 2:

    ss = FIELD_SCALES[valid]
    yy = abs_center[valid]

    slope_log, intercept_log = (
        np.polyfit(
            np.log(ss),
            np.log(yy),
            1
        )
    )

    print()
    print(
        "Effective weak-field exponent"
        " for |P_+| at x~0 =",
        slope_log
    )
    print(
        "Ideal third-order value = 3"
    )


# ============================================================
# 21. beta = 0 full-master null test
# ============================================================

print()
print("Running beta=0 null test...")

P0p, P0m = full_qutip_phase_cycled(
    x=X_SCAN[ic],
    scale=1.0,
    beta=0.0
)

Pbp, Pbm = full_qutip_phase_cycled(
    x=X_SCAN[ic],
    scale=1.0,
    beta=BETA
)

null_num = (
    abs(P0p)
    +
    abs(P0m)
)

null_den = max(
    abs(Pbp)
    +
    abs(Pbm),
    1e-30
)

null_ratio = (
    null_num
    /
    null_den
)

print(
    "beta=0 full-master null ratio =",
    null_ratio
)


# ============================================================
# 22. Optimum demodulation phase
# ============================================================

dZdx = np.gradient(
    Z_rho3,
    X_SCAN
)

central_complex_slope = (
    dZdx[ic]
)

phi_opt = np.angle(
    central_complex_slope
)

phi_deg = np.degrees(
    phi_opt
)

S_rho3 = np.real(
    Z_rho3
    * np.exp(
        -1j * phi_opt
    )
)

S_qutip = np.real(
    Z_qutip
    * np.exp(
        -1j * phi_opt
    )
)

print()
print("Demodulation:")
print(
    "rho3 central complex slope =",
    central_complex_slope
)
print(
    "optimum phase =",
    phi_deg,
    "deg"
)


# ============================================================
# 23. Simple pass/fail interpretation
# ============================================================

print()
print("=" * 70)
print("Interpretation")
print("=" * 70)

if shape_z < 0.05:

    print(
        "PASS: perturbative rho3 and full-master"
        " heterodyne lineshapes agree very well."
    )

elif shape_z < 0.15:

    print(
        "PARTIAL PASS: overall structure agrees,"
        " but quantitative differences remain."
    )

else:

    print(
        "CHECK REQUIRED: disagreement is too large."
    )


if np.max(
        np.concatenate([
            fit_res_plus,
            fit_res_minus
        ])
) < 0.02:

    print(
        "PASS: weak-field a3 extrapolation is stable."
    )

else:

    print(
        "CHECK: weak-field polynomial fit residual"
        " is relatively large."
    )


if null_ratio < 1e-2:

    print(
        "PASS: beta=0 null test."
    )

else:

    print(
        "CHECK: beta=0 residual is too large."
    )

print()
print(
    "All finite =",
    np.all(
        np.isfinite(
            np.concatenate([
                rho3_plus,
                rho3_minus,
                qutip_a3_plus,
                qutip_a3_minus,
                Z_rho3,
                Z_qutip
            ])
        )
    )
)


# ============================================================
# 24. Figure 1:
# Complex heterodyne comparison
# ============================================================

common_Z = max(
    np.max(np.abs(Z_rho3)),
    1e-30
)

plt.figure()

plt.plot(
    X_SCAN,
    np.real(Z_rho3) / common_Z,
    "o-",
    label="rho3 Re(Z)"
)

plt.plot(
    X_SCAN,
    np.real(Z_qutip) / common_Z,
    "s--",
    label="QuTiP a3 Re(Z)"
)

plt.plot(
    X_SCAN,
    np.imag(Z_rho3) / common_Z,
    "o-",
    label="rho3 Im(Z)"
)

plt.plot(
    X_SCAN,
    np.imag(Z_qutip) / common_Z,
    "s--",
    label="QuTiP a3 Im(Z)"
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
    r"Two-photon detuning $x/\gamma_{\rm ref}$"
)

plt.ylabel(
    "Common-normalized complex response"
)

plt.title(
    r"$\rho^{(3)}$ versus full-master weak-field $a_3$"
)

plt.legend()

plt.tight_layout()


# ============================================================
# 25. Figure 2:
# Generated sidebands
# ============================================================

common_P = max(
    np.max(
        np.abs(
            np.concatenate([
                rho3_plus,
                rho3_minus
            ])
        )
    ),
    1e-30
)

plt.figure()

plt.plot(
    X_SCAN,
    np.abs(rho3_plus) / common_P,
    "o-",
    label=r"$\rho^{(3)}:\ \omega+\Omega_m$"
)

plt.plot(
    X_SCAN,
    np.abs(qutip_a3_plus) / common_P,
    "s--",
    label=r"QuTiP $a_3:\ \omega+\Omega_m$"
)

plt.plot(
    X_SCAN,
    np.abs(rho3_minus) / common_P,
    "o-",
    label=r"$\rho^{(3)}:\ \omega-\Omega_m$"
)

plt.plot(
    X_SCAN,
    np.abs(qutip_a3_minus) / common_P,
    "s--",
    label=r"QuTiP $a_3:\ \omega-\Omega_m$"
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
    "Generated sideband magnitude"
)

plt.title(
    "Third-order generated-sideband cross validation"
)

plt.legend()

plt.tight_layout()


# ============================================================
# 26. Figure 3:
# Demodulated error signal
# ============================================================

norm_s = max(
    np.max(np.abs(S_rho3)),
    1e-30
)

plt.figure()

plt.plot(
    X_SCAN,
    S_rho3 / norm_s,
    "o-",
    label="explicit rho3"
)

plt.plot(
    X_SCAN,
    S_qutip / norm_s,
    "s--",
    label="full QuTiP -> a3"
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
    "Common-normalized MTS signal"
)

plt.title(
    "MTS error-signal cross validation"
)

plt.legend()

plt.tight_layout()


# ============================================================
# 27. Figure 4:
# Relative discrepancy
# ============================================================

relative_Z_error = (
    np.abs(
        Z_qutip
        -
        Z_rho3
    )
    /
    np.maximum(
        np.abs(Z_rho3),
        1e-15
    )
)

plt.figure()

plt.plot(
    X_SCAN,
    relative_Z_error,
    "o-"
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
    r"$|Z_{\rm Qutip}-Z_{\rho3}|/|Z_{\rho3}|$"
)

plt.title(
    "Point-by-point rho3 / full-master discrepancy"
)

plt.tight_layout()


# ============================================================
# 28. Figure 5:
# QuTiP weak-field scaling at resonance
# ============================================================

plt.figure()

center_mag = np.abs(
    center_full_plus
)

ref = (
    center_mag[0]
    * (
        FIELD_SCALES
        /
        FIELD_SCALES[0]
    ) ** 3
)

plt.loglog(
    FIELD_SCALES,
    center_mag,
    "o-",
    label="full QuTiP"
)

plt.loglog(
    FIELD_SCALES,
    ref,
    "--",
    label=r"cubic reference $\propto s^3$"
)

plt.xlabel(
    "Common field-amplitude scale s"
)

plt.ylabel(
    r"$|P_{+1}|$"
)

plt.title(
    "Weak-field third-order scaling"
)

plt.legend()

plt.tight_layout()


plt.show()