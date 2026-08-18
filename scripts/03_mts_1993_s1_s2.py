import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 0. 基准参数
# ============================================================
#
# 论文真正用作归一化线宽的是：
#
#     gamma_bar = (gamma_ab + gamma_bc) / 2
#
# 我们设 gamma_bar = 1。
#
# 论文没有明确给出 Fig.4 / Fig.5 中所有 gamma 的具体比值，
# 因此第一版仍采用最少参数的对称基准：
#
# gamma_ab = gamma_bc = 1
# gamma_a  = gamma_b  = gamma_ac = 1
#
# 这是“模型假设”，不是论文给出的实验数值。
# ============================================================

gamma_a = 1.0
gamma_b = 1.0

gamma_ab = 1.0
gamma_bc = 1.0
gamma_ac = 1.0

mu_ab = 1.0
mu_bc = 1.0

gamma_bar = (gamma_ab + gamma_bc) / 2.0


# ============================================================
# 1. 基本检查
# ============================================================

if not np.isclose(gamma_bar, 1.0):
    raise ValueError(
        "Current normalization requires gamma_bar = 1."
    )


# ============================================================
# 2. 三种失谐
# ============================================================

def detunings(x, level_offset):
    """
    所有频率均以 gamma_bar 为单位。

    x =
        Delta_ab / gamma_bar

    level_offset =
        (omega_ac/2 - omega_ab) / gamma_bar

    因此：

        Delta_ab = x

        Delta_bar
          = omega - omega_ac/2
          = x - level_offset

        Delta_ac
          = 2*omega - omega_ac
          = 2 * Delta_bar
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
#    1993论文式(20)
# ============================================================

def S1_delta(
    x,
    delta,
    level_offset,
    gamma_a=1.0,
    gamma_b=1.0,
    gamma_ab=1.0,
    mu_ab=1.0
):

    Delta_ab, _, _ = detunings(
        x,
        level_offset
    )

    population_factor = (
        mu_ab**2
        / (gamma_a + 1j * delta)

        +

        mu_ab**2
        / (gamma_b + 1j * delta)
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
#    1993论文式(21)
# ============================================================

def S2_delta(
    x,
    delta,
    level_offset,
    gamma_b=1.0,
    gamma_ab=1.0,
    gamma_bc=1.0,
    gamma_ac=1.0,
    mu_bc=1.0
):

    (
        Delta_ab,
        Delta_bar,
        Delta_ac
    ) = detunings(
        x,
        level_offset
    )

    gbar = (
        gamma_ab + gamma_bc
    ) / 2.0


    # --------------------------------------------------------
    # term 1
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
    # term 2
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
    # term 3
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
    # term 4
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
    # term 5
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
# 5. S1(2delta)
#    论文式(23)
# ============================================================

def S1_2delta(
    x,
    delta,
    level_offset,
    gamma_a=1.0,
    gamma_b=1.0,
    gamma_ab=1.0,
    mu_ab=1.0
):

    Delta_ab, _, _ = detunings(
        x,
        level_offset
    )

    population_factor = (

        mu_ab**2
        / (
            gamma_a
            + 2j * delta
        )

        +

        mu_ab**2
        / (
            gamma_b
            + 2j * delta
        )
    )

    bracket = (

        1.0 / (
            gamma_ab
            - 1j * (
                Delta_ab
                - 3.0 * delta / 2.0
            )
        )

        -

        1.0 / (
            gamma_ab
            + 1j * (
                Delta_ab
                + 3.0 * delta / 2.0
            )
        )
    )

    return (
        -population_factor
        * bracket
    )


# ============================================================
# 6. S2(2delta)
#    论文式(24)
# ============================================================

def S2_2delta(
    x,
    delta,
    level_offset,
    gamma_b=1.0,
    gamma_ab=1.0,
    gamma_bc=1.0,
    gamma_ac=1.0,
    mu_bc=1.0
):

    (
        Delta_ab,
        Delta_bar,
        Delta_ac
    ) = detunings(
        x,
        level_offset
    )

    gbar = (
        gamma_ab + gamma_bc
    ) / 2.0


    term1 = (

        -mu_bc**2
        / (
            gamma_b
            + 2j * delta
        )

        *

        (

            1.0 / (
                gbar
                + 1j * (
                    Delta_bar
                    + 3.0 * delta / 2.0
                )
            )

            -

            1.0 / (
                gbar
                - 1j * (
                    Delta_bar
                    - 3.0 * delta / 2.0
                )
            )
        )
    )


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
                    + 3.0 * delta / 2.0
                )
            )

            -

            1.0 / (
                gamma_ab
                + 1j * (
                    Delta_ab
                    + 3.0 * delta / 2.0
                )
            )
        )
    )


    term3 = (

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
                    - 3.0 * delta / 2.0
                )
            )

            -

            1.0 / (
                gbar
                - 1j * (
                    Delta_bar
                    - 3.0 * delta / 2.0
                )
            )
        )
    )


    return (
        term1
        + term2
        + term3
    )


# ============================================================
# 7. 解调函数
# ============================================================

def demodulate(S, phi):
    """
    复外差信号：

        I ~ S exp(i delta t) + c.c.

    经过相位为 phi 的同步解调：

        V_phi = Re[S exp(i phi)]

    等价于：

        Re(S) cos(phi)
        - Im(S) sin(phi)
    """

    return np.real(
        S * np.exp(1j * phi)
    )


# ============================================================
# 8. 在指定共振中心计算复斜率
# ============================================================

def fit_complex_slope(
    x,
    S,
    x0,
    window=0.20
):
    """
    在 x0 附近分别对 Re(S)、Im(S)
    作三次多项式拟合。

    返回：

        dS/dx | x0

    的复数值。
    """

    mask = (
        np.abs(x - x0)
        <= window
    )

    if np.count_nonzero(mask) < 7:
        raise ValueError(
            "Too few points for slope fit."
        )

    xx = (
        x[mask] - x0
    )

    coeff_r = np.polyfit(
        xx,
        np.real(S[mask]),
        3
    )

    coeff_i = np.polyfit(
        xx,
        np.imag(S[mask]),
        3
    )

    slope_complex = (
        coeff_r[-2]
        + 1j * coeff_i[-2]
    )

    return slope_complex


# ============================================================
# 9. 自动寻找“最大中心斜率”解调相位
# ============================================================

def optimum_dispersion_phase(
    x,
    S,
    x0,
    window=0.20
):
    """
    选择 phi，使得：

        dV_phi/dx

    在 x0 处最大且为正。

    由于：

        V_phi = Re[S exp(i phi)]

    若：

        dS/dx = |dS/dx| exp(i theta)

    则取：

        phi = -theta

    即可得到最大正斜率。
    """

    complex_slope = fit_complex_slope(
        x,
        S,
        x0,
        window
    )

    phi = (
        -np.angle(
            complex_slope
        )
    )

    # wrap 到 [-pi, pi]
    phi = (
        (phi + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )

    max_slope = (
        np.abs(
            complex_slope
        )
    )

    return (
        phi,
        max_slope
    )


# ============================================================
# 10. 找整条复信号中最强变化的位置
# ============================================================

def strongest_gradient_center(
    x,
    S,
    edge_fraction=0.08
):
    """
    用于没有明显“中心=0”的复杂高阶线型。

    寻找 |dS/dx| 最大的位置。
    """

    dSdx = np.gradient(
        S,
        x
    )

    strength = np.abs(
        dSdx
    )

    n = len(x)

    i0 = int(
        edge_fraction * n
    )

    i1 = int(
        (1.0 - edge_fraction) * n
    )

    idx = (
        i0
        + np.argmax(
            strength[i0:i1]
        )
    )

    return x[idx]


# ============================================================
# 11. 画“斜率最优正交分量 + 正交分量”
# ============================================================

def plot_phase_optimized_pair(
    x,
    S,
    x0,
    xlabel,
    title,
    fit_window=0.20
):
    """
    不再简单把 Re(S) 称作吸收，
    Im(S) 称作色散。

    先用共振中心斜率寻找最佳解调相位：

        phi_disp

    再定义其正交方向：

        phi_orth = phi_disp + pi/2
    """

    (
        phi_disp,
        max_slope
    ) = optimum_dispersion_phase(
        x,
        S,
        x0,
        fit_window
    )

    phi_orth = (
        phi_disp
        + np.pi / 2.0
    )

    signal_disp = demodulate(
        S,
        phi_disp
    )

    signal_orth = demodulate(
        S,
        phi_orth
    )

    # 共同归一化
    scale = max(
        np.max(
            np.abs(signal_disp)
        ),
        np.max(
            np.abs(signal_orth)
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        x,
        signal_disp / scale,
        label=(
            "Slope-max quadrature "
            f"({np.degrees(phi_disp):.1f}°)"
        )
    )

    plt.plot(
        x,
        signal_orth / scale,
        label="Orthogonal quadrature"
    )

    plt.axhline(
        0,
        linewidth=0.8
    )

    plt.axvline(
        x0,
        linestyle="--",
        linewidth=0.8,
        label="Reference resonance"
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        "Response "
        "(common normalization)"
    )

    plt.title(
        title
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    return (
        phi_disp,
        max_slope
    )


# ============================================================
# 12. Figure 5：
#
# delta / gamma_bar = 0.6
# level offset       = 20
#
# ============================================================

delta_fig5 = 0.6
offset_fig5 = 20.0


# ------------------------------------------------------------
# S1(delta)
#
# 单光子共振：
# Delta_ab / gamma_bar = 0
# ------------------------------------------------------------

x_s1_f5 = np.linspace(
    -6.0,
    6.0,
    6001
)

S1_f5 = S1_delta(
    x_s1_f5,
    delta_fig5,
    offset_fig5,
    gamma_a,
    gamma_b,
    gamma_ab,
    mu_ab
)

phi_s1_f5, slope_s1_f5 = (
    plot_phase_optimized_pair(
        x_s1_f5,
        S1_f5,
        x0=0.0,
        xlabel=(
            r"Single-photon detuning "
            r"$\Delta_{ab}/\bar{\gamma}$"
        ),
        title=(
            r"1993 model: "
            r"$S_1(\delta)$, "
            r"$\delta/\bar{\gamma}=0.6$"
        )
    )
)


# ------------------------------------------------------------
# S2(delta)
#
# 用双光子失谐作为横轴：
#
# y = Delta_bar / gamma_bar
#
# 共振中心就是 y = 0。
# ------------------------------------------------------------

y_s2_f5 = np.linspace(
    -6.0,
    6.0,
    6001
)

x_s2_f5 = (
    y_s2_f5
    + offset_fig5
)

S2_f5 = S2_delta(
    x_s2_f5,
    delta_fig5,
    offset_fig5,
    gamma_b,
    gamma_ab,
    gamma_bc,
    gamma_ac,
    mu_bc
)

phi_s2_f5, slope_s2_f5 = (
    plot_phase_optimized_pair(
        y_s2_f5,
        S2_f5,
        x0=0.0,
        xlabel=(
            r"Two-photon detuning "
            r"$\bar{\Delta}/\bar{\gamma}$"
        ),
        title=(
            r"1993 model: "
            r"$S_2(\delta)$, "
            r"$\delta/\bar{\gamma}=0.6$"
        )
    )
)


# ============================================================
# 13. gamma_ac 灵敏度扫描
# ============================================================
#
# 这是非常重要的新图。
#
# 保持论文定义的：
#
# gamma_ab = gamma_bc = 1
# gamma_bar = 1
#
# 只改变高低能级相干退相干：
#
# gamma_ac / gamma_bar
#
# 观察双光子线型是否强烈变化。
# ============================================================

gamma_ac_values = [
    0.25,
    0.50,
    1.00,
    2.00
]

sensitivity_curves = []

for gac in gamma_ac_values:

    S_test = S2_delta(
        x_s2_f5,
        delta_fig5,
        offset_fig5,
        gamma_b,
        gamma_ab,
        gamma_bc,
        gac,
        mu_bc
    )

    phi_test, _ = (
        optimum_dispersion_phase(
            y_s2_f5,
            S_test,
            x0=0.0,
            window=0.20
        )
    )

    V_test = demodulate(
        S_test,
        phi_test
    )

    # 注意：
    # 此图单独归一化每一条曲线，
    # 仅用于比较“线型”，不用于比较幅度。
    V_test = (
        V_test
        / np.max(
            np.abs(V_test)
        )
    )

    sensitivity_curves.append(
        V_test
    )


plt.figure(
    figsize=(8, 5)
)

for gac, curve in zip(
    gamma_ac_values,
    sensitivity_curves
):

    plt.plot(
        y_s2_f5,
        curve,
        label=(
            rf"$\gamma_{{ac}}/"
            rf"\bar{{\gamma}}={gac}$"
        )
    )

plt.axhline(
    0,
    linewidth=0.8
)

plt.axvline(
    0,
    linewidth=0.8
)

plt.xlabel(
    r"$\bar{\Delta}/\bar{\gamma}$"
)

plt.ylabel(
    "Individually normalized shape"
)

plt.title(
    r"Sensitivity of "
    r"$S_2(\delta)$ "
    r"to $\gamma_{ac}$"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 14. Figure 4：
#
# delta / gamma_bar = 10
# level offset       = 10
#
# 扩大扫描范围：
#
# 原脚本 -10...30 会截掉
#
# S1(2delta) 在 -3delta/2 = -15 的共振。
#
# 现在统一扩大到足够范围。
# ============================================================

delta_fig4 = 10.0
offset_fig4 = 10.0


# ------------------------------------------------------------
# S1(delta)
# ------------------------------------------------------------

x_s1_f4 = np.linspace(
    -30.0,
    30.0,
    12001
)

S1_f4 = S1_delta(
    x_s1_f4,
    delta_fig4,
    offset_fig4,
    gamma_a,
    gamma_b,
    gamma_ab,
    mu_ab
)

phi_s1_f4, slope_s1_f4 = (
    plot_phase_optimized_pair(
        x_s1_f4,
        S1_f4,
        x0=0.0,
        xlabel=(
            r"$\Delta_{ab}/\bar{\gamma}$"
        ),
        title=(
            r"1993 model: "
            r"$S_1(\delta)$, "
            r"$\delta/\bar{\gamma}=10$"
        ),
        fit_window=0.40
    )
)


# ------------------------------------------------------------
# S2(delta)
#
# 双光子失谐 y = Delta_bar / gamma_bar
# ------------------------------------------------------------

y_s2_f4 = np.linspace(
    -30.0,
    30.0,
    12001
)

x_s2_f4 = (
    y_s2_f4
    + offset_fig4
)

S2_f4 = S2_delta(
    x_s2_f4,
    delta_fig4,
    offset_fig4,
    gamma_b,
    gamma_ab,
    gamma_bc,
    gamma_ac,
    mu_bc
)

phi_s2_f4, slope_s2_f4 = (
    plot_phase_optimized_pair(
        y_s2_f4,
        S2_f4,
        x0=0.0,
        xlabel=(
            r"$\bar{\Delta}/\bar{\gamma}$"
        ),
        title=(
            r"1993 model: "
            r"$S_2(\delta)$, "
            r"$\delta/\bar{\gamma}=10$"
        ),
        fit_window=0.40
    )
)


# ------------------------------------------------------------
# S1(2delta)
#
# 主要单光子结构位于：
#
# Delta_ab = +/- 3delta/2
#
# 即 +/-15。
#
# 为了固定相位选择，取 +15 作为参考共振。
# ------------------------------------------------------------

x_s1_2_f4 = np.linspace(
    -30.0,
    30.0,
    12001
)

S1_2_f4 = S1_2delta(
    x_s1_2_f4,
    delta_fig4,
    offset_fig4,
    gamma_a,
    gamma_b,
    gamma_ab,
    mu_ab
)

reference_s1_2 = (
    3.0
    * delta_fig4
    / 2.0
)

phi_s1_2_f4, slope_s1_2_f4 = (
    plot_phase_optimized_pair(
        x_s1_2_f4,
        S1_2_f4,
        x0=reference_s1_2,
        xlabel=(
            r"$\Delta_{ab}/\bar{\gamma}$"
        ),
        title=(
            r"1993 model: "
            r"$S_1(2\delta)$, "
            r"$\delta/\bar{\gamma}=10$"
        ),
        fit_window=0.40
    )
)


# ------------------------------------------------------------
# S2(2delta)
#
# 这是最复杂的一类。
#
# 不人为假定某一个峰一定是“中心”，
# 先让程序找 |dS/dy| 最大的位置。
# ------------------------------------------------------------

y_s2_2_f4 = np.linspace(
    -30.0,
    30.0,
    12001
)

x_s2_2_f4 = (
    y_s2_2_f4
    + offset_fig4
)

S2_2_f4 = S2_2delta(
    x_s2_2_f4,
    delta_fig4,
    offset_fig4,
    gamma_b,
    gamma_ab,
    gamma_bc,
    gamma_ac,
    mu_bc
)

reference_s2_2 = (
    strongest_gradient_center(
        y_s2_2_f4,
        S2_2_f4
    )
)

phi_s2_2_f4, slope_s2_2_f4 = (
    plot_phase_optimized_pair(
        y_s2_2_f4,
        S2_2_f4,
        x0=reference_s2_2,
        xlabel=(
            r"$\bar{\Delta}/\bar{\gamma}$"
        ),
        title=(
            r"1993 model: "
            r"$S_2(2\delta)$, "
            r"$\delta/\bar{\gamma}=10$"
        ),
        fit_window=0.40
    )
)


# ============================================================
# 15. 数值稳定性检查
# ============================================================

all_signals = [
    S1_f5,
    S2_f5,
    S1_f4,
    S2_f4,
    S1_2_f4,
    S2_2_f4
]

finite_test = all(
    np.all(
        np.isfinite(S)
    )
    for S in all_signals
)


# ============================================================
# 16. 控制台输出
# ============================================================

print()
print(
    "=============================================="
)
print(
    "1993 MTS Stage-B validation"
)
print(
    "=============================================="
)

print()

print(
    "Decay-rate model:"
)

print(
    f"gamma_a   / gbar = {gamma_a:.3f}"
)

print(
    f"gamma_b   / gbar = {gamma_b:.3f}"
)

print(
    f"gamma_ab  / gbar = {gamma_ab:.3f}"
)

print(
    f"gamma_bc  / gbar = {gamma_bc:.3f}"
)

print(
    f"gamma_ac  / gbar = {gamma_ac:.3f}"
)

print(
    f"gamma_bar        = {gamma_bar:.3f}"
)


print()
print(
    "IMPORTANT:"
)

print(
    "Individual gamma values above are "
    "model assumptions; the paper only fixes "
    "ratios relative to gamma_bar for Fig.4/Fig.5."
)


print()
print(
    "Figure 5:"
)

print(
    f"delta/gamma_bar = {delta_fig5}"
)

print(
    "level offset / gamma_bar =",
    offset_fig5
)

print(
    "S1 optimum demod phase = "
    f"{np.degrees(phi_s1_f5):.2f} deg"
)

print(
    "S1 maximum center slope = "
    f"{slope_s1_f5:.6f}"
)

print(
    "S2 optimum demod phase = "
    f"{np.degrees(phi_s2_f5):.2f} deg"
)

print(
    "S2 maximum center slope = "
    f"{slope_s2_f5:.6f}"
)


print()
print(
    "Figure 4:"
)

print(
    f"delta/gamma_bar = {delta_fig4}"
)

print(
    "level offset / gamma_bar =",
    offset_fig4
)

print(
    "S1(delta) phase = "
    f"{np.degrees(phi_s1_f4):.2f} deg"
)

print(
    "S2(delta) phase = "
    f"{np.degrees(phi_s2_f4):.2f} deg"
)

print(
    "S1(2delta) reference = "
    f"{reference_s1_2:.3f}"
)

print(
    "S1(2delta) phase = "
    f"{np.degrees(phi_s1_2_f4):.2f} deg"
)

print(
    "S2(2delta) strongest-gradient "
    "reference = "
    f"{reference_s2_2:.3f}"
)

print(
    "S2(2delta) phase = "
    f"{np.degrees(phi_s2_2_f4):.2f} deg"
)


print()
print(
    "All numerical values finite:",
    finite_test
)

print()
print(
    "Expected S1(2delta) resonances:"
)

print(
    "Delta_ab/gamma_bar = +/-",
    3.0 * delta_fig4 / 2.0
)

print()
print(
    "=============================================="
)


plt.show()