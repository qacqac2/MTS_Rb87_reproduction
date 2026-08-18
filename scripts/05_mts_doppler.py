import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# 0. 基本参数
# ============================================================

# ----------------------------
# Stage B 延续的归一化参数
# ----------------------------

gamma_ac = 1.0
gamma_ab = 1.0

# 文献未完全指定，继续采用对称模型假设
gamma_a = 1.0
gamma_b = 1.0
gamma_bc = 1.0

mu_ab = 1.0
mu_bc = 1.0

gamma_bar = (
    gamma_ab + gamma_bc
) / 2.0


# ----------------------------
# 1990 文献参数
# ----------------------------

delta = 0.6 * gamma_ac

# omega_ac/2 - omega_ab = 20 gamma_ac
level_offset = 20.0 * gamma_ac


# ============================================================
# 1. Doppler 参数
# ============================================================

# 原脚本的参考工作点
KU_REFERENCE = 50.0

# 新增：Doppler 宽度扫描
#
# ku/gamma_ac = 0
# 表示完全无 Doppler 情况
KU_SCAN = np.array([
    0.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0
])

# 高精度参考积分
NQ_REFERENCE = 8001

# Doppler 扫描积分点数
#
# 前一版本已经验证 Nq=2001 在 ku=50 时
# 就已高度收敛，因此这里 4001 足够。
NQ_SCAN = 4001


# ============================================================
# 2. 三种失谐
# ============================================================

def detunings_from_two_photon(y):
    """
    y = Delta_bar / gamma_ac

    Delta_bar =
        omega - omega_ac/2

    则：

        Delta_ab = y + level_offset

        Delta_bc = y - level_offset

        Delta_ac = 2*y
    """

    Delta_bar = y

    Delta_ab = (
        y + level_offset
    )

    Delta_bc = (
        y - level_offset
    )

    Delta_ac = (
        2.0 * y
    )

    return (
        Delta_ab,
        Delta_bc,
        Delta_ac
    )


# ============================================================
# 3. 含速度项的三阶极化
# ============================================================

def rho3_polarization_components(
    y,
    q,
    side
):
    """
    q = kv / gamma_ac

    side = +1:
        omega + delta

    side = -1:
        omega - delta

    返回：

        P1：
        中间态饱和吸收 MTS 部分

        P2：
        双光子 MTS 部分

    所有共同的整体比例常数被省略。
    """

    (
        Delta_ab,
        Delta_bc,
        Delta_ac
    ) = detunings_from_two_photon(
        y
    )


    # ========================================================
    # 光学相干共振分母
    # ========================================================

    A1 = 1.0 / (
        gamma_ab
        + 1j * (
            Delta_ab
            + side * delta
            - q
        )
    )

    A2 = 1.0 / (
        gamma_ab
        - 1j * (
            Delta_ab
            - q
        )
    )

    A3 = 1.0 / (
        gamma_ab
        + 1j * (
            Delta_ab
            - q
        )
    )

    A4 = 1.0 / (
        gamma_ab
        - 1j * (
            Delta_ab
            - side * delta
            - q
        )
    )


    # ========================================================
    # 布居相关组合
    # ========================================================

    B_population = (
        A1
        + A2
        - A3
        - A4
    )


    # ========================================================
    # 双光子相关组合
    # ========================================================

    B_two_shifted = (

        1.0 / (
            gamma_ab
            + 1j * (
                Delta_ab
                + side * delta
                - q
            )
        )

        +

        1.0 / (
            gamma_ab
            + 1j * (
                Delta_ab
                + q
            )
        )
    )


    B_two_center = (

        1.0 / (
            gamma_ab
            + 1j * (
                Delta_ab
                - q
            )
        )

        +

        1.0 / (
            gamma_ab
            + 1j * (
                Delta_ab
                + q
            )
        )
    )


    # ========================================================
    # rho_ab^(3)
    # ========================================================

    outer_ab = (
        gamma_ab
        + 1j * (
            Delta_ab
            + side * delta
            + q
        )
    )


    population_factor = (

        mu_ab**2
        / (
            gamma_a
            + 1j * side * delta
        )

        +

        mu_ab**2
        / (
            gamma_b
            + 1j * side * delta
        )
    )


    common_phase = (
        1j**3
    )


    # ----------------------------
    # S1 contribution
    # ----------------------------

    rho_ab_S1 = (

        (-side)
        * common_phase
        * mu_ab
        / outer_ab

        * population_factor
        * B_population
    )


    # ----------------------------
    # S2 contribution in rho_ab
    # ----------------------------

    rho_ab_S2 = (

        (-side)
        * common_phase
        * mu_ab
        / outer_ab

        *

        (

            mu_bc**2
            / (
                gamma_ac
                + 1j * (
                    Delta_ac
                    + side * delta
                )
            )
            * B_two_shifted

            -

            mu_bc**2
            / (
                gamma_ac
                + 1j * Delta_ac
            )
            * B_two_center
        )
    )


    # ========================================================
    # rho_bc^(3)
    # ========================================================

    outer_bc = (

        gamma_bc
        + 1j * (
            Delta_bc
            + side * delta
            + q
        )
    )


    rho_bc_S2 = (

        side
        * common_phase
        * mu_ab**2
        * mu_bc
        / outer_bc

        *

        (

            1.0
            / (
                gamma_b
                + 1j * side * delta
            )
            * B_population

            +

            1.0
            / (
                gamma_ac
                + 1j * (
                    Delta_ac
                    + side * delta
                )
            )
            * B_two_shifted

            -

            1.0
            / (
                gamma_ac
                + 1j * Delta_ac
            )
            * B_two_center
        )
    )


    # ========================================================
    # 三阶极化
    # ========================================================

    P1 = (
        mu_ab
        * rho_ab_S1
    )

    P2 = (

        mu_ab
        * rho_ab_S2

        +

        mu_bc
        * rho_bc_S2
    )


    return (
        P1,
        P2
    )


# ============================================================
# 4. 无 Doppler MTS
# ============================================================

def no_doppler_mts(
    y_grid
):

    Pp1, Pp2 = (
        rho3_polarization_components(
            y_grid,
            0.0,
            +1
        )
    )

    Pm1, Pm2 = (
        rho3_polarization_components(
            y_grid,
            0.0,
            -1
        )
    )


    Z1 = (

        -1j * Pp1

        +

        np.conj(
            -1j * Pm1
        )
    )


    Z2 = (

        -1j * Pp2

        +

        np.conj(
            -1j * Pm2
        )
    )


    return (
        Z1,
        Z2
    )


# ============================================================
# 5. Maxwell Doppler 平均
# ============================================================

def doppler_average_mts(
    y_grid,
    ku_ratio,
    Nq=4001,
    chunk_size=40
):
    """
    显式计算：

        integral f(q) P(q) dq

    其中：

        q = kv/gamma_ac

        f(q)
        =
        exp[-(q/ku)^2]
        /(ku*sqrt(pi))

    若 ku_ratio = 0，
    自动返回无 Doppler 结果。
    """

    if ku_ratio == 0:

        return no_doppler_mts(
            y_grid
        )


    q = np.linspace(
        -4.0 * ku_ratio,
        +4.0 * ku_ratio,
        Nq
    )


    weight = (

        np.exp(
            -(q / ku_ratio)**2
        )

        /

        (
            ku_ratio
            * np.sqrt(np.pi)
        )
    )


    Z1 = np.zeros(
        len(y_grid),
        dtype=np.complex128
    )

    Z2 = np.zeros(
        len(y_grid),
        dtype=np.complex128
    )


    # ========================================================
    # 分块计算避免占用过多内存
    # ========================================================

    for i0 in range(
        0,
        len(y_grid),
        chunk_size
    ):

        y = (
            y_grid[
                i0:i0 + chunk_size
            ][:, None]
        )

        qq = q[None, :]


        # ----------------------------
        # omega + delta
        # ----------------------------

        Pp1, Pp2 = (
            rho3_polarization_components(
                y,
                qq,
                +1
            )
        )


        # ----------------------------
        # omega - delta
        # ----------------------------

        Pm1, Pm2 = (
            rho3_polarization_components(
                y,
                qq,
                -1
            )
        )


        # ----------------------------
        # Maxwell 平均
        # ----------------------------

        Pp1_avg = np.trapezoid(
            Pp1 * weight,
            q,
            axis=1
        )

        Pp2_avg = np.trapezoid(
            Pp2 * weight,
            q,
            axis=1
        )

        Pm1_avg = np.trapezoid(
            Pm1 * weight,
            q,
            axis=1
        )

        Pm2_avg = np.trapezoid(
            Pm2 * weight,
            q,
            axis=1
        )


        # ----------------------------
        # Er ~ -i P
        # ----------------------------

        Ep1 = -1j * Pp1_avg
        Ep2 = -1j * Pp2_avg

        Em1 = -1j * Pm1_avg
        Em2 = -1j * Pm2_avg


        # ----------------------------
        # RF 外差复包络
        # ----------------------------

        block_len = len(
            Pp1_avg
        )


        Z1[
            i0:i0 + block_len
        ] = (

            Ep1

            +

            np.conj(
                Em1
            )
        )


        Z2[
            i0:i0 + block_len
        ] = (

            Ep2

            +

            np.conj(
                Em2
            )
        )


    return (
        Z1,
        Z2
    )


# ============================================================
# 6. 解调函数
# ============================================================

def demodulate(
    Z,
    phi
):

    return np.real(
        Z
        * np.exp(
            1j * phi
        )
    )


# ============================================================
# 7. 复中心斜率和最佳相位
# ============================================================

def complex_center_slope(
    x,
    Z,
    window=0.20
):

    mask = (
        np.abs(x)
        <= window
    )


    coeff_r = np.polyfit(
        x[mask],
        np.real(
            Z[mask]
        ),
        3
    )

    coeff_i = np.polyfit(
        x[mask],
        np.imag(
            Z[mask]
        ),
        3
    )


    return (
        coeff_r[-2]
        + 1j * coeff_i[-2]
    )


def optimum_phase(
    x,
    Z,
    window=0.20
):

    slope_complex = (
        complex_center_slope(
            x,
            Z,
            window
        )
    )


    phi = (
        -np.angle(
            slope_complex
        )
    )


    return (
        phi,
        np.abs(
            slope_complex
        )
    )


# ============================================================
# 8. 固定相位下中心斜率
# ============================================================

def real_center_slope(
    x,
    signal,
    window=0.20
):

    mask = (
        np.abs(x)
        <= window
    )


    coeff = np.polyfit(
        x[mask],
        signal[mask],
        3
    )


    return coeff[-2]


# ============================================================
# 9. 零点
# ============================================================

def zero_crossing_near_zero(
    x,
    y
):

    idx = np.where(
        y[:-1] * y[1:] <= 0
    )[0]


    if len(idx) == 0:

        return np.nan


    candidate_centers = (

        x[idx]
        + x[idx + 1]

    ) / 2.0


    j = idx[
        np.argmin(
            np.abs(
                candidate_centers
            )
        )
    ]


    x1 = x[j]
    x2 = x[j + 1]

    y1 = y[j]
    y2 = y[j + 1]


    if y2 == y1:

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


# ============================================================
# 10. 零点附近两个最近极值的间距
# ============================================================

def peak_separation_near_zero(
    x,
    y
):
    """
    对类似色散型 MTS 线型：

        negative peak
            -> zero
            -> positive peak

    找零点左右最近的局部极值。

    返回：

        x_right - x_left

    这是一个稳定的“特征线宽”指标，
    比对奇函数使用 FWHM 更合适。
    """

    maxima, _ = find_peaks(
        y
    )

    minima, _ = find_peaks(
        -y
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
        x[extrema] < 0
    ]

    right = extrema[
        x[extrema] > 0
    ]


    if (
        len(left) == 0
        or len(right) == 0
    ):

        return np.nan


    left_idx = left[-1]
    right_idx = right[0]


    return (
        x[right_idx]
        - x[left_idx]
    )


# ============================================================
# 11. 普通线性 Doppler 吸收
# ============================================================

def linear_absorption_doppler(
    Delta_ab_grid,
    ku_ratio,
    Nq=4001,
    chunk_size=80
):

    # ----------------------------
    # 无 Doppler
    # ----------------------------

    if ku_ratio == 0:

        return (

            gamma_ab

            /

            (
                gamma_ab**2

                +

                Delta_ab_grid**2
            )
        )


    q = np.linspace(
        -4.0 * ku_ratio,
        +4.0 * ku_ratio,
        Nq
    )


    weight = (

        np.exp(
            -(q / ku_ratio)**2
        )

        /

        (
            ku_ratio
            * np.sqrt(np.pi)
        )
    )


    result = np.zeros(
        len(
            Delta_ab_grid
        )
    )


    for i0 in range(
        0,
        len(
            Delta_ab_grid
        ),
        chunk_size
    ):

        Delta = (

            Delta_ab_grid[
                i0:i0 + chunk_size
            ][:, None]
        )


        response = (

            gamma_ab

            /

            (
                gamma_ab**2

                +

                (
                    Delta
                    - q[None, :]
                )**2
            )
        )


        block = np.trapezoid(
            response
            * weight,
            q,
            axis=1
        )


        result[
            i0:i0 + len(block)
        ] = block


    return result


# ============================================================
# 12. 参考扫描网格
# ============================================================

y_two = np.linspace(
    -5.0,
    +5.0,
    801
)


# ============================================================
# 13. ku = 50 的高精度参考结果
# ============================================================

Z1_ref, Z2_ref = (
    doppler_average_mts(
        y_two,
        ku_ratio=KU_REFERENCE,
        Nq=NQ_REFERENCE
    )
)


Z1_0, Z2_0 = (
    no_doppler_mts(
        y_two
    )
)


# ============================================================
# 14. 固定实验解调相位
#
# 用 ku/gamma=50 的 MTS
# 找一次最佳相位。
#
# Doppler 扫描中保持这个相位不变，
# 这样才能单独研究 Doppler 参数的影响。
# ============================================================

phi_S2_reference, _ = (
    optimum_phase(
        y_two,
        Z2_ref
    )
)


V2_ref = demodulate(
    Z2_ref,
    phi_S2_reference
)

V2_0 = demodulate(
    Z2_0,
    phi_S2_reference
)


# ============================================================
# 15. S1 共振
# ============================================================

x_single = np.linspace(
    -5.0,
    +5.0,
    801
)

y_for_single = (
    x_single
    - level_offset
)


Z1_single_ref, _ = (
    doppler_average_mts(
        y_for_single,
        ku_ratio=KU_REFERENCE,
        Nq=NQ_REFERENCE
    )
)


Z1_single_0, _ = (
    no_doppler_mts(
        y_for_single
    )
)


phi_S1_reference, _ = (
    optimum_phase(
        x_single,
        Z1_single_ref
    )
)


V1_ref = demodulate(
    Z1_single_ref,
    phi_S1_reference
)

V1_0 = demodulate(
    Z1_single_0,
    phi_S1_reference
)


# ============================================================
# 16. 共同归一化常数
#
# 所有 S2 都用无 Doppler S2 最大值；
# 所有 S1 都用无 Doppler S1 最大值。
# ============================================================

S2_COMMON_SCALE = np.max(
    np.abs(
        V2_0
    )
)

S1_COMMON_SCALE = np.max(
    np.abs(
        V1_0
    )
)


# ============================================================
# 17. Figure 1
#
# 普通吸收：
# 统一使用无 Doppler 吸收峰作为归一化基准
# ============================================================

Delta_wide = np.linspace(
    -150.0,
    +150.0,
    1201
)


linear_no = (
    linear_absorption_doppler(
        Delta_wide,
        ku_ratio=0.0
    )
)


linear_ref = (
    linear_absorption_doppler(
        Delta_wide,
        ku_ratio=KU_REFERENCE,
        Nq=NQ_REFERENCE
    )
)


LINEAR_COMMON_SCALE = np.max(
    linear_no
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    Delta_wide,
    linear_no
    / LINEAR_COMMON_SCALE,
    label="No Doppler"
)

plt.plot(
    Delta_wide,
    linear_ref
    / LINEAR_COMMON_SCALE,
    label=(
        rf"$ku/\gamma_{{ac}}="
        rf"{KU_REFERENCE:.0f}$"
    )
)

plt.axvline(
    level_offset,
    linestyle="--",
    linewidth=0.9,
    label="Two-photon resonance location"
)

plt.xlabel(
    r"Single-photon detuning "
    r"$\Delta_{ab}/\gamma_{ac}$"
)

plt.ylabel(
    "Absorption / no-Doppler peak"
)

plt.title(
    "Ordinary absorption: "
    "common normalization"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 18. Figure 2
#
# Doppler 背景 vs 双光子 MTS
#
# 注意：
# 线性吸收和三阶 MTS 不是同一个物理量，
# 不能使用同一个绝对比例常数。
#
# 所以使用双纵轴：
#
# 左：吸收 / 无 Doppler 吸收峰
# 右：MTS / 无 Doppler MTS 峰
# ============================================================

Delta_ab_two = (
    level_offset
    + y_two
)


background_ref = (
    linear_absorption_doppler(
        Delta_ab_two,
        ku_ratio=KU_REFERENCE,
        Nq=NQ_REFERENCE
    )
)


fig, ax1 = plt.subplots(
    figsize=(8, 5)
)


line1 = ax1.plot(
    y_two,
    background_ref
    / LINEAR_COMMON_SCALE,
    label="Intermediate-state absorption"
)

ax1.set_xlabel(
    r"Two-photon detuning "
    r"$\bar{\Delta}/\gamma_{ac}$"
)

ax1.set_ylabel(
    "Absorption / no-Doppler peak"
)


ax2 = ax1.twinx()


line2 = ax2.plot(
    y_two,
    V2_ref
    / S2_COMMON_SCALE,
    label=r"Two-photon MTS $S_2$"
)

ax2.set_ylabel(
    "MTS / no-Doppler MTS peak"
)


ax1.axvline(
    0,
    linestyle="--",
    linewidth=0.8
)

ax1.axhline(
    0,
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
    "Doppler background versus "
    "two-photon MTS"
)

ax1.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 19. Figure 3
#
# S2：
# 无 Doppler 与 ku=50
#
# 严格共同归一化
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    y_two,
    V2_0
    / S2_COMMON_SCALE,
    label="No Doppler"
)

plt.plot(
    y_two,
    V2_ref
    / S2_COMMON_SCALE,
    linestyle="--",
    label=(
        rf"$ku/\gamma_{{ac}}="
        rf"{KU_REFERENCE:.0f}$"
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
    r"$\bar{\Delta}/\gamma_{ac}$"
)

plt.ylabel(
    "MTS signal / no-Doppler peak"
)

plt.title(
    r"$S_2$: common-normalized "
    r"Doppler comparison"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 20. Figure 4
#
# S1：
# 无 Doppler 与 ku=50
#
# 严格共同归一化
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    x_single,
    V1_0
    / S1_COMMON_SCALE,
    label="No Doppler"
)

plt.plot(
    x_single,
    V1_ref
    / S1_COMMON_SCALE,
    linestyle="--",
    label=(
        rf"$ku/\gamma_{{ac}}="
        rf"{KU_REFERENCE:.0f}$"
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
    r"$\Delta_{ab}/\gamma_{ac}$"
)

plt.ylabel(
    "MTS signal / no-Doppler peak"
)

plt.title(
    r"$S_1$: common-normalized "
    r"Doppler comparison"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 21. Doppler 参数扫描
# ============================================================

scan_signals = []

amplitude_ratio = []
slope_ratio = []

peak_separation = []
zero_shift = []

farwing_ratio = []

optimum_phase_values = []
phase_shift_from_reference = []

conventional_background_fraction = []


# 无 Doppler 基准
amp0 = np.max(
    np.abs(
        V2_0
    )
)

slope0 = real_center_slope(
    y_two,
    V2_0
)


for ku_ratio in KU_SCAN:

    # --------------------------------------------------------
    # MTS
    # --------------------------------------------------------

    Z1_i, Z2_i = (
        doppler_average_mts(
            y_two,
            ku_ratio=ku_ratio,
            Nq=NQ_SCAN
        )
    )


    # --------------------------------------------------------
    # 固定 ku=50 时确定的解调相位
    # --------------------------------------------------------

    V2_i = demodulate(
        Z2_i,
        phi_S2_reference
    )


    scan_signals.append(
        V2_i
    )


    # --------------------------------------------------------
    # 峰值
    # --------------------------------------------------------

    amp_i = np.max(
        np.abs(
            V2_i
        )
    )

    amplitude_ratio.append(
        amp_i / amp0
    )


    # --------------------------------------------------------
    # 中心斜率
    # --------------------------------------------------------

    slope_i = real_center_slope(
        y_two,
        V2_i
    )

    slope_ratio.append(
        np.abs(slope_i)
        / np.abs(slope0)
    )


    # --------------------------------------------------------
    # 特征峰间距
    # --------------------------------------------------------

    peak_separation.append(
        peak_separation_near_zero(
            y_two,
            V2_i
        )
    )


    # --------------------------------------------------------
    # 零点
    # --------------------------------------------------------

    zero_shift.append(
        zero_crossing_near_zero(
            y_two,
            V2_i
        )
    )


    # --------------------------------------------------------
    # 远翼背景
    # --------------------------------------------------------

    edge_mask = (
        np.abs(
            y_two
        )
        > 4.5
    )


    farwing_i = (

        np.mean(
            np.abs(
                V2_i[
                    edge_mask
                ]
            )
        )

        /

        amp_i
    )


    farwing_ratio.append(
        farwing_i
    )


    # --------------------------------------------------------
    # 如果每个 ku 都重新优化相位，
    # 最佳相位如何漂移？
    # --------------------------------------------------------

    phi_opt_i, _ = (
        optimum_phase(
            y_two,
            Z2_i
        )
    )


    optimum_phase_values.append(
        phi_opt_i
    )


    # wrap 到 [-pi, pi]
    dphi = np.angle(
        np.exp(
            1j
            * (
                phi_opt_i
                - phi_S2_reference
            )
        )
    )


    phase_shift_from_reference.append(
        np.degrees(
            dphi
        )
    )


    # --------------------------------------------------------
    # 普通单光子吸收：
    #
    # 双光子共振位置 Delta_ab=20
    # 相对于该 Doppler 包络中心 Delta_ab=0
    # 的背景比例
    # --------------------------------------------------------

    absorption_test = (
        linear_absorption_doppler(
            np.array([
                0.0,
                level_offset
            ]),
            ku_ratio=ku_ratio,
            Nq=NQ_SCAN
        )
    )


    background_fraction = (

        absorption_test[1]
        / absorption_test[0]
    )


    conventional_background_fraction.append(
        background_fraction
    )


# ============================================================
# 22. 转 numpy array
# ============================================================

amplitude_ratio = np.array(
    amplitude_ratio
)

slope_ratio = np.array(
    slope_ratio
)

peak_separation = np.array(
    peak_separation
)

zero_shift = np.array(
    zero_shift
)

farwing_ratio = np.array(
    farwing_ratio
)

phase_shift_from_reference = np.array(
    phase_shift_from_reference
)

conventional_background_fraction = np.array(
    conventional_background_fraction
)


# ============================================================
# 23. Figure 5
#
# 不同 Doppler 宽度下 S2 线型
#
# 全部使用同一个无 Doppler 最大值归一化
# ============================================================

plt.figure(
    figsize=(8, 5)
)

for ku_ratio, V in zip(
    KU_SCAN,
    scan_signals
):

    plt.plot(
        y_two,
        V / S2_COMMON_SCALE,
        label=(
            rf"$ku/\gamma_{{ac}}="
            rf"{ku_ratio:g}$"
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
    r"$\bar{\Delta}/\gamma_{ac}$"
)

plt.ylabel(
    "MTS signal / no-Doppler peak"
)

plt.title(
    r"$S_2$ versus Doppler width "
    "(common normalization)"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 24. Figure 6
#
# Doppler 对：
#
# 峰值
# 中心斜率
#
# 的影响
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    KU_SCAN,
    amplitude_ratio,
    marker="o",
    label="Peak amplitude"
)

plt.plot(
    KU_SCAN,
    slope_ratio,
    marker="s",
    label="Center slope"
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=0.8
)

plt.xlabel(
    r"$ku/\gamma_{ac}$"
)

plt.ylabel(
    "Ratio to no-Doppler value"
)

plt.title(
    "Doppler-induced enhancement of "
    "MTS amplitude and slope"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 25. Figure 7
#
# 特征峰间距 + 零点偏移
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(8, 5)
)


line1 = ax1.plot(
    KU_SCAN,
    peak_separation,
    marker="o",
    label="Nearest-extrema separation"
)

ax1.set_xlabel(
    r"$ku/\gamma_{ac}$"
)

ax1.set_ylabel(
    r"Peak separation "
    r"$/\gamma_{ac}$"
)


ax2 = ax1.twinx()


line2 = ax2.plot(
    KU_SCAN,
    zero_shift,
    marker="s",
    label="Numerical zero crossing"
)

ax2.set_ylabel(
    r"Zero shift "
    r"$\bar{\Delta}_0/\gamma_{ac}$"
)


ax2.axhline(
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
    "MTS width and lock-point shift"
)

ax1.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 26. Figure 8
#
# MTS 基线抑制
# vs
# 普通 Doppler 背景
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    KU_SCAN,
    conventional_background_fraction,
    marker="o",
    label=(
        "Ordinary absorption at "
        "two-photon resonance"
    )
)

plt.plot(
    KU_SCAN,
    farwing_ratio,
    marker="s",
    label="MTS far-wing / peak"
)

plt.xlabel(
    r"$ku/\gamma_{ac}$"
)

plt.ylabel(
    "Dimensionless ratio"
)

plt.title(
    "Doppler background versus "
    "MTS baseline"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 27. Figure 9
#
# 最佳解调相位随 Doppler 宽度变化
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    KU_SCAN,
    phase_shift_from_reference,
    marker="o"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=0.8
)

plt.xlabel(
    r"$ku/\gamma_{ac}$"
)

plt.ylabel(
    "Optimum phase shift "
    "relative to ku=50 (deg)"
)

plt.title(
    "Doppler dependence of "
    "optimum demodulation phase"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 28. 积分收敛检查
#
# 保留原脚本功能
# ============================================================

_, Z2_2001 = (
    doppler_average_mts(
        y_two,
        ku_ratio=KU_REFERENCE,
        Nq=2001
    )
)

_, Z2_4001 = (
    doppler_average_mts(
        y_two,
        ku_ratio=KU_REFERENCE,
        Nq=4001
    )
)

_, Z2_8001 = (
    doppler_average_mts(
        y_two,
        ku_ratio=KU_REFERENCE,
        Nq=8001
    )
)


error_2001 = (

    np.linalg.norm(
        Z2_2001
        - Z2_8001
    )

    /

    np.linalg.norm(
        Z2_8001
    )
)


error_4001 = (

    np.linalg.norm(
        Z2_4001
        - Z2_8001
    )

    /

    np.linalg.norm(
        Z2_8001
    )
)


# ============================================================
# 29. 原参考工作点诊断
# ============================================================

edge_mask_ref = (
    np.abs(
        y_two
    )
    > 4.5
)


mts_background_ratio_ref = (

    np.mean(
        np.abs(
            V2_ref[
                edge_mask_ref
            ]
        )
    )

    /

    np.max(
        np.abs(
            V2_ref
        )
    )
)


zero_S2_ref = (
    zero_crossing_near_zero(
        y_two,
        V2_ref
    )
)


amp_ratio_ref = (

    np.max(
        np.abs(
            V2_ref
        )
    )

    /

    np.max(
        np.abs(
            V2_0
        )
    )
)


slope_ref = real_center_slope(
    y_two,
    V2_ref
)

slope_no = real_center_slope(
    y_two,
    V2_0
)

slope_ratio_ref = (
    np.abs(
        slope_ref
    )
    /
    np.abs(
        slope_no
    )
)


# ============================================================
# 30. 控制台输出
# ============================================================

print()
print(
    "============================================================"
)
print(
    "Stage C+: Doppler scan with common normalization"
)
print(
    "============================================================"
)


print()
print(
    "Paper/reference parameters:"
)

print(
    "delta/gamma_ac =",
    delta / gamma_ac
)

print(
    "(omega_ac/2 - omega_ab)/gamma_ac =",
    level_offset / gamma_ac
)


print()
print(
    "Reference Doppler point:"
)

print(
    "ku/gamma_ac =",
    KU_REFERENCE
)

print(
    "Fixed S2 demodulation phase = "
    f"{np.degrees(phi_S2_reference):.3f} deg"
)

print(
    "Fixed S1 demodulation phase = "
    f"{np.degrees(phi_S1_reference):.3f} deg"
)


print()
print(
    "Common-normalization diagnostics at ku=50:"
)

print(
    "S2 amplitude / no-Doppler amplitude =",
    amp_ratio_ref
)

print(
    "S2 center slope / no-Doppler slope =",
    slope_ratio_ref
)

print(
    "S2 far-wing / peak =",
    mts_background_ratio_ref
)

print(
    "S2 zero crossing =",
    zero_S2_ref
)


print()
print(
    "Doppler integration convergence at ku=50:"
)

print(
    "Nq=2001 vs 8001 relative L2 error =",
    error_2001
)

print(
    "Nq=4001 vs 8001 relative L2 error =",
    error_4001
)


print()
print(
    "============================================================"
)

print(
    "Doppler scan results"
)

print(
    "============================================================"
)

print(
    f"{'ku/gamma':>10} "
    f"{'Amp/A0':>10} "
    f"{'Slope/K0':>10} "
    f"{'PeakSep':>10} "
    f"{'Zero':>10} "
    f"{'Wing/Peak':>12} "
    f"{'AbsBG':>10} "
    f"{'dPhi(deg)':>12}"
)


for (
    ku,
    amp,
    slope,
    width,
    zero,
    wing,
    bg,
    dphi
) in zip(

    KU_SCAN,
    amplitude_ratio,
    slope_ratio,
    peak_separation,
    zero_shift,
    farwing_ratio,
    conventional_background_fraction,
    phase_shift_from_reference
):

    print(
        f"{ku:10.2f} "
        f"{amp:10.5f} "
        f"{slope:10.5f} "
        f"{width:10.5f} "
        f"{zero:10.5f} "
        f"{wing:12.5e} "
        f"{bg:10.5f} "
        f"{dphi:12.3f}"
    )


print()
print(
    "All scan values finite =",
    (
        np.all(
            np.isfinite(
                amplitude_ratio
            )
        )
        and
        np.all(
            np.isfinite(
                slope_ratio
            )
        )
    )
)

print(
    "============================================================"
)


plt.show()