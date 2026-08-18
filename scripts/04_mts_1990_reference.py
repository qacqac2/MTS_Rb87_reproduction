import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 0. 1990 文献参数与归一化约定
# ============================================================
#
# 文献 Fig.2 明确给出：
#
#   delta / gamma_ac = 0.6
#   delta / gamma_ab = 0.6
#
# 因此：
#
#   gamma_ac = gamma_ab
#
# 我们取 gamma_ac 为频率单位：
#
#   gamma_ac = 1
#   gamma_ab = 1
#
# 文献还给出：
#
#   omega_ac/2 - omega_ab = 20 * gamma_ac
#
# ------------------------------------------------------------
# 以下参数文献没有逐一给出：
#
#   gamma_a
#   gamma_b
#   gamma_bc
#
# 因此采用最简单的对称归一化假设 = 1。
#
# mu_ab, mu_bc 只影响整体幅度，
# 对归一化线型无影响，也设为 1。
# ============================================================

gamma_ac = 1.0
gamma_ab = 1.0

# ---- unresolved model assumptions ----
gamma_a = 1.0
gamma_b = 1.0
gamma_bc = 1.0

mu_ab = 1.0
mu_bc = 1.0

gamma_bar = (
    gamma_ab + gamma_bc
) / 2.0


# ============================================================
# 1. 1990 Fig.2 的明确参数
# ============================================================

delta_fig2 = 0.6 * gamma_ac

level_offset = 20.0 * gamma_ac


# ============================================================
# 2. 三种失谐
# ============================================================

def detunings(x):
    """
    x = Delta_ab / gamma_ac

    当前 gamma_ac = 1，因此直接使用无量纲数值。

    定义：

        Delta_ab = omega - omega_ab

        Delta_ac = 2*omega - omega_ac

        Delta_bar =
            omega - (omega_ab + omega_bc)/2
            = omega - omega_ac/2

    又因为：

        omega_ac/2 - omega_ab
            = level_offset

    所以：

        Delta_bar = Delta_ab - level_offset

        Delta_ac = 2 * Delta_bar
    """

    Delta_ab = x

    Delta_bar = (
        x - level_offset
    )

    Delta_ac = (
        2.0 * Delta_bar
    )

    return (
        Delta_ab,
        Delta_bar,
        Delta_ac
    )


# ============================================================
# 3. S1(delta)
#
# 1990 文献式(9)
#
# 中间能级的单光子饱和吸收 MTS 分量
# ============================================================

def S1_delta(
    x,
    delta
):

    Delta_ab, _, _ = detunings(
        x
    )

    population_factor = (

        mu_ab**2
        / (
            gamma_a
            + 1j * delta
        )

        +

        mu_ab**2
        / (
            gamma_b
            + 1j * delta
        )
    )

    bracket = (

        1.0 / (
            gamma_ab
            + 1j * (
                Delta_ab
                + delta / 2.0
            )
        )

        -

        1.0 / (
            gamma_ab
            + 1j * (
                Delta_ab
                + delta
            )
        )

        +

        1.0 / (
            gamma_ab
            - 1j * (
                Delta_ab
                - delta
            )
        )

        -

        1.0 / (
            gamma_ab
            - 1j * (
                Delta_ab
                - delta / 2.0
            )
        )
    )

    return (
        -population_factor
        * bracket
    )


# ============================================================
# 4. S2(delta)
#
# 1990 文献式(10)
#
# 高能级的双光子 MTS 分量
# ============================================================

def S2_delta(
    x,
    delta
):

    (
        Delta_ab,
        Delta_bar,
        Delta_ac
    ) = detunings(
        x
    )

    gbar = (
        gamma_ab
        + gamma_bc
    ) / 2.0


    # --------------------------------------------------------
    # Term 1
    # --------------------------------------------------------

    term1 = (

        -mu_bc**2
        / (
            gamma_b
            + 1j * delta
        )

        *

        (

            1.0 / (
                gbar
                + 1j * (
                    Delta_bar
                    + delta
                )
            )

            -

            1.0 / (
                gbar
                + 1j * (
                    Delta_bar
                    + delta / 2.0
                )
            )

            +

            1.0 / (
                gbar
                - 1j * (
                    Delta_bar
                    - delta / 2.0
                )
            )

            -

            1.0 / (
                gbar
                - 1j * (
                    Delta_bar
                    - delta
                )
            )
        )
    )


    # --------------------------------------------------------
    # Term 2
    # --------------------------------------------------------

    term2 = (

        mu_bc**2
        / (
            gamma_ac
            + 1j * (
                Delta_ac
                + delta
            )
        )

        *

        (

            1.0 / (
                gbar
                + 1j * (
                    Delta_bar
                    + delta
                )
            )

            -

            1.0 / (
                gamma_ab
                + 1j * (
                    Delta_ab
                    + delta
                )
            )
        )
    )


    # --------------------------------------------------------
    # Term 3
    # --------------------------------------------------------

    term3 = (

        mu_bc**2
        / (
            gamma_ac
            + 1j * Delta_ac
        )

        *

        (

            1.0 / (
                gamma_ab
                + 1j * (
                    Delta_ab
                    + delta / 2.0
                )
            )

            -

            1.0 / (
                gbar
                + 1j * (
                    Delta_bar
                    + delta / 2.0
                )
            )
        )
    )


    # --------------------------------------------------------
    # Term 4
    # --------------------------------------------------------

    term4 = (

        mu_bc**2
        / (
            gamma_ac
            - 1j * (
                Delta_ac
                - delta
            )
        )

        *

        (

            1.0 / (
                gamma_ab
                - 1j * (
                    Delta_ab
                    - delta
                )
            )

            -

            1.0 / (
                gbar
                - 1j * (
                    Delta_bar
                    - delta
                )
            )
        )
    )


    # --------------------------------------------------------
    # Term 5
    # --------------------------------------------------------

    term5 = (

        mu_bc**2
        / (
            gamma_ac
            - 1j * Delta_ac
        )

        *

        (

            1.0 / (
                gbar
                - 1j * (
                    Delta_bar
                    - delta / 2.0
                )
            )

            -

            1.0 / (
                gamma_ab
                - 1j * (
                    Delta_ab
                    - delta / 2.0
                )
            )
        )
    )


    return (
        term1
        + term2
        + term3
        + term4
        + term5
    )


# ============================================================
# 5. 同步解调
# ============================================================

def demodulate(
    S,
    phi
):
    """
    若复外差信号为：

        I ~ S exp(i delta t) + c.c.

    则解调相位 phi 下：

        V_phi = Re[S exp(i phi)]
    """

    return np.real(
        S
        * np.exp(1j * phi)
    )


# ============================================================
# 6. 计算复中心斜率
# ============================================================

def complex_center_slope(
    signal_function,
    delta,
    x0,
    h=1e-4
):
    """
    数值计算：

        dS/dx | x0

    返回复数。
    """

    S_plus = signal_function(
        x0 + h,
        delta
    )

    S_minus = signal_function(
        x0 - h,
        delta
    )

    return (
        S_plus - S_minus
    ) / (
        2.0 * h
    )


# ============================================================
# 7. 给定复斜率的最佳解调相位
# ============================================================

def optimum_phase_from_slope(
    complex_slope
):
    """
    要使：

        Re[
            (dS/dx) exp(i phi)
        ]

    为最大正值：

        phi = -arg(dS/dx)
    """

    return (
        -np.angle(
            complex_slope
        )
    )


# ============================================================
# 8. 1990 Fig.2(a)
#
# 双光子跃迁 S2(delta)
# ============================================================

# 双光子共振在：
#
#   Delta_bar = 0
#
# 即：
#
#   Delta_ab = level_offset = 20
#
# 为方便作图，以 Delta_bar/gamma_ac 为横轴。

two_photon_detuning = np.linspace(
    -5.0,
    5.0,
    5001
)

x_two = (
    level_offset
    + two_photon_detuning
)

S2_fig2 = S2_delta(
    x_two,
    delta_fig2
)


# ------------------------------------------------------------
# 在论文 Fig.2 标准条件 delta/gamma_ac=0.6
# 确定一次 lock-in 相位。
#
# 后续 Fig.3 扫描 delta 时保持此相位不变。
# ------------------------------------------------------------

S2_slope_complex = (
    complex_center_slope(
        S2_delta,
        delta_fig2,
        x0=level_offset
    )
)

phi_reference = (
    optimum_phase_from_slope(
        S2_slope_complex
    )
)

V2_fig2 = demodulate(
    S2_fig2,
    phi_reference
)

V2_fig2 = (
    V2_fig2
    / np.max(
        np.abs(
            V2_fig2
        )
    )
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    two_photon_detuning,
    V2_fig2
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
    r"Laser detuning "
    r"$\bar{\Delta}/\gamma_{ac}$"
)

plt.ylabel(
    "Normalized MTS signal"
)

plt.title(
    r"1990 Fig.2(a): "
    r"two-photon $S_2(\delta)$"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 9. 1990 Fig.2(b)
#
# 中间能级饱和吸收 S1(delta)
# ============================================================

single_photon_detuning = np.linspace(
    -5.0,
    5.0,
    5001
)

x_single = (
    single_photon_detuning
)

S1_fig2 = S1_delta(
    x_single,
    delta_fig2
)


# ------------------------------------------------------------
# Fig.2(b) 是理论线型图。
#
# 因原文没有给出两幅理论图采用的绝对
# electronic phase zero，这里为 S1 单独选择
# 最大中心斜率正交分量，再翻转符号使之
# 与原图方向一致。
# ------------------------------------------------------------

S1_slope_complex = (
    complex_center_slope(
        S1_delta,
        delta_fig2,
        x0=0.0
    )
)

phi_S1 = (
    optimum_phase_from_slope(
        S1_slope_complex
    )
)

V1_fig2 = demodulate(
    S1_fig2,
    phi_S1
)

# 原论文 Fig.2(b) 的斜率方向与 Fig.2(a) 相反。
# 整体 ± 号属于解调相位 180° 的约定。
V1_fig2 = -V1_fig2

V1_fig2 = (
    V1_fig2
    / np.max(
        np.abs(
            V1_fig2
        )
    )
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    single_photon_detuning,
    V1_fig2
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
    r"Laser detuning "
    r"$\Delta_{ab}/\gamma_{ab}$"
)

plt.ylabel(
    "Normalized MTS signal"
)

plt.title(
    r"1990 Fig.2(b): "
    r"saturation absorption $S_1(\delta)$"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 10. 1990 Fig.3
#
# 双光子共振中心斜率
# vs
# delta / gamma_ac
# ============================================================

ratio_scan = np.linspace(
    0.05,
    2.5,
    1200
)

fixed_phase_slopes = []
optimized_phase_envelope = []


for ratio in ratio_scan:

    delta_i = (
        ratio * gamma_ac
    )

    complex_slope_i = (
        complex_center_slope(
            S2_delta,
            delta_i,
            x0=level_offset
        )
    )


    # --------------------------------------------------------
    # 真正用于和 1990 Fig.3 对比的量：
    #
    # 保持 Fig.2 的 lock-in 相位不变。
    # --------------------------------------------------------

    slope_fixed = np.real(
        complex_slope_i
        * np.exp(
            1j * phi_reference
        )
    )

    fixed_phase_slopes.append(
        slope_fixed
    )


    # --------------------------------------------------------
    # 同时计算理论“上包络”：
    #
    # 每一个 delta 都重新优化相位。
    #
    # 这不是 Fig.3 的主要比较对象，
    # 只是用于展示两种定义的区别。
    # --------------------------------------------------------

    optimized_phase_envelope.append(
        np.abs(
            complex_slope_i
        )
    )


fixed_phase_slopes = np.array(
    fixed_phase_slopes
)

optimized_phase_envelope = np.array(
    optimized_phase_envelope
)


# ============================================================
# 11. 找固定相位下最大斜率位置
# ============================================================

idx_fixed = np.argmax(
    fixed_phase_slopes
)

best_ratio_fixed = (
    ratio_scan[idx_fixed]
)


idx_envelope = np.argmax(
    optimized_phase_envelope
)

best_ratio_envelope = (
    ratio_scan[idx_envelope]
)


# ============================================================
# 12. Figure 3 归一化
# ============================================================

fixed_norm = (
    fixed_phase_slopes
    / np.max(
        fixed_phase_slopes
    )
)

envelope_norm = (
    optimized_phase_envelope
    / np.max(
        optimized_phase_envelope
    )
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    ratio_scan,
    fixed_norm,
    label="Fixed lock-in phase"
)

plt.plot(
    ratio_scan,
    envelope_norm,
    linestyle="--",
    label="Phase-optimized envelope"
)

# 1990 文献理论结果：
# maximum slope near delta/gamma_ac = 0.76
plt.axvline(
    0.76,
    linestyle=":",
    linewidth=1.2,
    label="Paper: 0.76"
)

plt.axvline(
    best_ratio_fixed,
    linestyle="--",
    linewidth=0.9,
    label=(
        "Simulation max = "
        f"{best_ratio_fixed:.3f}"
    )
)

plt.xlabel(
    r"$\delta/\gamma_{ac}$"
)

plt.ylabel(
    "Normalized center slope"
)

plt.title(
    "1990 Fig.3: "
    "two-photon MTS center slope"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 13. 可选：把 9 MHz 实验频率换算成物理尺度
# ============================================================
#
# 1990 实验使用的相位调制频率为 9 MHz。
#
# 若把 Fig.2 的：
#
#   delta/gamma_ac = 0.6
#
# 与该实验频率直接对应，则：
#
#   gamma_ac/(2pi) ≈ 9 / 0.6 = 15 MHz
#
# 注意：
# 这是一个“按 Fig.2 比值映射”的推断值，
# 不是文章直接报告的 gamma_ac。
# ============================================================

experimental_fm_MHz = 9.0

gamma_ac_MHz_inferred = (
    experimental_fm_MHz
    / 0.6
)

offset_MHz_inferred = (
    20.0
    * gamma_ac_MHz_inferred
)


# ============================================================
# 14. 数值诊断输出
# ============================================================

print()
print(
    "============================================="
)

print(
    "1990 two-photon MTS reference simulation"
)

print(
    "============================================="
)

print()

print(
    "Paper-fixed parameters:"
)

print(
    "delta/gamma_ac =",
    delta_fig2 / gamma_ac
)

print(
    "delta/gamma_ab =",
    delta_fig2 / gamma_ab
)

print(
    "gamma_ac/gamma_ab =",
    gamma_ac / gamma_ab
)

print(
    "(omega_ac/2 - omega_ab)/gamma_ac =",
    level_offset / gamma_ac
)


print()
print(
    "Assumed normalized parameters:"
)

print(
    "gamma_a/gamma_ac =",
    gamma_a / gamma_ac
)

print(
    "gamma_b/gamma_ac =",
    gamma_b / gamma_ac
)

print(
    "gamma_bc/gamma_ac =",
    gamma_bc / gamma_ac
)

print(
    "gamma_bar/gamma_ac =",
    gamma_bar / gamma_ac
)


print()
print(
    "Fig.2 demodulation:"
)

print(
    "S2 reference phase = "
    f"{np.degrees(phi_reference):.2f} deg"
)

print(
    "S1 slope-max phase = "
    f"{np.degrees(phi_S1):.2f} deg"
)


print()
print(
    "Fig.3 validation:"
)

print(
    "Paper maximum slope near "
    "delta/gamma_ac = 0.76"
)

print(
    "Simulation, fixed phase maximum at "
    "delta/gamma_ac =",
    best_ratio_fixed
)

print(
    "Phase-optimized envelope maximum at "
    "delta/gamma_ac =",
    best_ratio_envelope
)


print()
print(
    "Optional physical-scale inference:"
)

print(
    "Experimental modulation frequency =",
    experimental_fm_MHz,
    "MHz"
)

print(
    "If 9 MHz corresponds to "
    "delta/gamma_ac = 0.6:"
)

print(
    "gamma_ac/(2pi) inferred ≈",
    gamma_ac_MHz_inferred,
    "MHz"
)

print(
    "(omega_ac/2 - omega_ab)/(2pi) inferred ≈",
    offset_MHz_inferred,
    "MHz"
)


print()
print(
    "All finite:",
    np.all(
        np.isfinite(
            fixed_phase_slopes
        )
    )
)

print(
    "============================================="
)


plt.show()