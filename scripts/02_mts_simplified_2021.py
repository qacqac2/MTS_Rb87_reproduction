import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv
from scipy.signal import find_peaks


# ============================================================
# 0. 基本配置
# ============================================================

# 87Rb D2 自然线宽
# 注意：本简化模型直接使用 MHz 数值，
# 不在这里额外乘 2*pi。
Gamma = 6.065            # MHz

# EOM 相位调制度
beta = 0.28

# 整体比例常数 C
# 只影响绝对幅度，不影响线型、峰位和最优调制频率
C = 1.0

# 论文给出的典型调制频率
fm_list = [
    2.0,
    6.0,
    9.0,
    12.5,
    24.3
]

# 失谐范围使用论文量级：
# Delta/Gamma = -10 ... +10
detuning_norm = np.linspace(
    -10.0,
    10.0,
    10001
)

detuning = detuning_norm * Gamma


# ============================================================
# 1. Lorentzian 吸收和色散函数
# ============================================================

def L(det, n, gamma, fm):
    """
    Lorentzian 吸收型分量

    L_n =
        gamma^2 /
        [gamma^2 + (det - n*fm)^2]

    参数单位必须一致，例如全部使用 MHz。
    """

    x = det - n * fm

    return gamma**2 / (
        gamma**2 + x**2
    )


def D(det, n, gamma, fm):
    """
    Lorentzian 色散型分量

    D_n =
        gamma*(det - n*fm) /
        [gamma^2 + (det - n*fm)^2]
    """

    x = det - n * fm

    return gamma * x / (
        gamma**2 + x**2
    )


# ============================================================
# 2. 2021 文献式(4)
# ============================================================

def mts_components(
    det,
    gamma,
    fm,
    beta,
    C=1.0
):
    """
    根据 2021 文献式(4)计算 MTS 的：

    absorption  吸收型正交分量
    dispersion  色散型正交分量

    特别注意：
    原式中有频率相关前因子

        C / sqrt(gamma^2 + fm^2)
        * J0(beta) * J1(beta)

    上一版脚本漏掉了该因子。
    """

    J0 = jv(0, beta)
    J1 = jv(1, beta)

    prefactor = (
        C
        * J0
        * J1
        / np.sqrt(
            gamma**2 + fm**2
        )
    )

    # ----------------------------
    # 吸收型分量
    # ----------------------------

    absorption = prefactor * (
        L(det, -1.0, gamma, fm)
        - L(det, -0.5, gamma, fm)
        + L(det, +0.5, gamma, fm)
        - L(det, +1.0, gamma, fm)
    )

    # ----------------------------
    # 色散型分量
    # ----------------------------

    dispersion_raw = prefactor * (
        D(det, +1.0, gamma, fm)
        - D(det, +0.5, gamma, fm)
        - D(det, -0.5, gamma, fm)
        + D(det, -1.0, gamma, fm)
    )

    # 整体负号仅用于匹配论文图中的斜率方向。
    # 实际实验中该符号取决于 Mixer / LO 的相位约定。
    dispersion = -dispersion_raw

    return absorption, dispersion


# ============================================================
# 3. 任意解调相位下的 MTS 信号
# ============================================================

def mts_demodulated_signal(
    det,
    gamma,
    fm,
    beta,
    phi,
    C=1.0
):
    """
    解调后的混合 MTS 信号：

        S = A cos(phi) + D sin(phi)

    phi 单位 rad。
    """

    A, Dsig = mts_components(
        det,
        gamma,
        fm,
        beta,
        C
    )

    return (
        A * np.cos(phi)
        + Dsig * np.sin(phi)
    )


# ============================================================
# 4. 全局峰峰值
#    对应论文图1(c)
# ============================================================

def global_peak_to_peak(signal):
    """
    整条曲线的全局峰峰值：

        max(signal) - min(signal)
    """

    return (
        np.max(signal)
        - np.min(signal)
    )


# ============================================================
# 5. 找最靠近零点的两个极值
#    对应论文图2
# ============================================================

def local_peak_to_peak_near_zero(
    signal,
    det
):
    """
    找到零失谐两侧距离零点最近的两个局部极值。

    返回：
        local_vpp
        left_detuning
        right_detuning
    """

    center_idx = np.argmin(
        np.abs(det)
    )

    maxima, _ = find_peaks(
        signal
    )

    minima, _ = find_peaks(
        -signal
    )

    extrema = np.sort(
        np.concatenate(
            [maxima, minima]
        )
    )

    left = extrema[
        extrema < center_idx
    ]

    right = extrema[
        extrema > center_idx
    ]

    if (
        len(left) == 0
        or len(right) == 0
    ):
        return (
            np.nan,
            np.nan,
            np.nan
        )

    left_idx = left[-1]
    right_idx = right[0]

    vpp = abs(
        signal[right_idx]
        - signal[left_idx]
    )

    return (
        vpp,
        det[left_idx],
        det[right_idx]
    )


# ============================================================
# 6. 中心过零斜率
# ============================================================

def center_slope(
    signal,
    det,
    gamma,
    window=0.05
):
    """
    在零点附近使用三次多项式：

        S = a0 + a1*Delta
              + a2*Delta^2
              + a3*Delta^3

    拟合中心线型。

    返回：
        a1 = dS/dDelta | Delta=0

    window = 0.05
    表示拟合 |Delta| <= 0.05*Gamma。
    """

    mask = (
        np.abs(det)
        <= window * gamma
    )

    coeff = np.polyfit(
        det[mask],
        signal[mask],
        deg=3
    )

    # np.polyfit 返回：
    # [a3, a2, a1, a0]
    slope = coeff[-2]

    return slope


# ============================================================
# 7. 对称性检查
# ============================================================

def odd_symmetry_error(signal):
    """
    对奇函数应满足：

        S(Delta) = -S(-Delta)

    返回归一化最大误差。
    """

    denominator = np.max(
        np.abs(signal)
    )

    if denominator == 0:
        return 0.0

    error = np.max(
        np.abs(
            signal
            + signal[::-1]
        )
    )

    return error / denominator


# ============================================================
# 8. Bessel 系数检查
# ============================================================

J0 = jv(0, beta)
J1 = jv(1, beta)

print(
    "======================================"
)
print(
    "Bessel coefficients"
)
print(
    "======================================"
)

print(
    "beta =",
    beta
)

print(
    "J0(beta) =",
    J0
)

print(
    "J1(beta) =",
    J1
)

print(
    "J0*J1 =",
    J0 * J1
)


# ============================================================
# 9. 图1(a)：不同调制频率下吸收型信号
# ============================================================

abs_curves = []

for fm in fm_list:

    A, Dsig = mts_components(
        detuning,
        Gamma,
        fm,
        beta,
        C
    )

    abs_curves.append(A)


# 统一归一化
# 不能逐条曲线各自归一化
abs_common_scale = max(
    np.max(np.abs(s))
    for s in abs_curves
)


plt.figure(
    figsize=(8, 5)
)

for fm, signal in zip(
    fm_list,
    abs_curves
):

    plt.plot(
        detuning / Gamma,
        signal / abs_common_scale,
        label=(
            f"{fm:.1f} MHz "
            f"({fm/Gamma:.2f}Γ)"
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
    r"Detuning $\Delta/\Gamma$"
)

plt.ylabel(
    "Absorption signal "
    "(common normalization)"
)

plt.title(
    "MTS absorption component"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 10. 图1(b)：不同调制频率下色散型信号
# ============================================================

disp_curves = []

for fm in fm_list:

    A, Dsig = mts_components(
        detuning,
        Gamma,
        fm,
        beta,
        C
    )

    disp_curves.append(
        Dsig
    )


disp_common_scale = max(
    np.max(np.abs(s))
    for s in disp_curves
)


plt.figure(
    figsize=(8, 5)
)

for fm, signal in zip(
    fm_list,
    disp_curves
):

    plt.plot(
        detuning / Gamma,
        signal / disp_common_scale,
        label=(
            f"{fm:.1f} MHz "
            f"({fm/Gamma:.2f}Γ)"
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
    r"Detuning $\Delta/\Gamma$"
)

plt.ylabel(
    "Dispersion signal "
    "(common normalization)"
)

plt.title(
    "MTS dispersion component"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 11. 参数扫描
# ============================================================

# 论文包含到 24.3 MHz ≈ 4.01 Gamma，
# 因此扫描到 5 Gamma。
ratio_scan = np.linspace(
    0.05,
    5.0,
    1201
)

global_vpp_abs = []
global_vpp_disp = []

local_vpp_abs = []
local_vpp_disp = []

slope_abs = []
slope_disp = []

odd_error_abs = []
odd_error_disp = []


for ratio in ratio_scan:

    fm = ratio * Gamma

    A, Dsig = mts_components(
        detuning,
        Gamma,
        fm,
        beta,
        C
    )

    # ----------------------------
    # 图1(c)定义：
    # 全局峰峰值
    # ----------------------------

    global_vpp_abs.append(
        global_peak_to_peak(A)
    )

    global_vpp_disp.append(
        global_peak_to_peak(Dsig)
    )

    # ----------------------------
    # 图2定义：
    # 零点附近两极值峰峰值
    # ----------------------------

    local_A, _, _ = (
        local_peak_to_peak_near_zero(
            A,
            detuning
        )
    )

    local_D, _, _ = (
        local_peak_to_peak_near_zero(
            Dsig,
            detuning
        )
    )

    local_vpp_abs.append(
        local_A
    )

    local_vpp_disp.append(
        local_D
    )

    # ----------------------------
    # 中心过零斜率
    # ----------------------------

    slope_abs.append(
        center_slope(
            A,
            detuning,
            Gamma
        )
    )

    slope_disp.append(
        center_slope(
            Dsig,
            detuning,
            Gamma
        )
    )

    # ----------------------------
    # 奇函数检查
    # ----------------------------

    odd_error_abs.append(
        odd_symmetry_error(A)
    )

    odd_error_disp.append(
        odd_symmetry_error(Dsig)
    )


global_vpp_abs = np.array(
    global_vpp_abs
)

global_vpp_disp = np.array(
    global_vpp_disp
)

local_vpp_abs = np.array(
    local_vpp_abs
)

local_vpp_disp = np.array(
    local_vpp_disp
)

slope_abs = np.array(
    slope_abs
)

slope_disp = np.array(
    slope_disp
)

odd_error_abs = np.array(
    odd_error_abs
)

odd_error_disp = np.array(
    odd_error_disp
)


# ============================================================
# 12. 图1(c)：全局峰峰值
# ============================================================

# 以色散信号最大 Vpp 作为统一归一化基准，
# 这样能直接比较论文中的 0.67。
vpp_reference = np.max(
    global_vpp_disp
)

global_vpp_abs_norm = (
    global_vpp_abs
    / vpp_reference
)

global_vpp_disp_norm = (
    global_vpp_disp
    / vpp_reference
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    ratio_scan,
    global_vpp_abs_norm,
    label="Absorption"
)

plt.plot(
    ratio_scan,
    global_vpp_disp_norm,
    linestyle="--",
    label="Dispersion"
)

plt.xlabel(
    r"$f_m/\Gamma$"
)

plt.ylabel(
    "Peak-to-peak amplitude "
    "(common normalization)"
)

plt.title(
    "Global MTS peak-to-peak amplitude"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 13. 图1(d)：中心过零斜率
# ============================================================

# 用色散正斜率最大值统一归一化
slope_reference = np.max(
    slope_disp
)

slope_abs_norm = (
    slope_abs
    / slope_reference
)

slope_disp_norm = (
    slope_disp
    / slope_reference
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    ratio_scan,
    slope_abs_norm,
    label="Absorption"
)

plt.plot(
    ratio_scan,
    slope_disp_norm,
    linestyle="--",
    label="Dispersion"
)

plt.axhline(
    0,
    linewidth=0.8
)

plt.xlabel(
    r"$f_m/\Gamma$"
)

plt.ylabel(
    "Zero-crossing slope "
    "(common normalization)"
)

plt.title(
    "MTS zero-crossing slope"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 14. 图2：
#     零点附近两个最近极值的峰峰值
# ============================================================

local_reference = np.nanmax(
    local_vpp_disp
)

local_vpp_abs_norm = (
    local_vpp_abs
    / local_reference
)

local_vpp_disp_norm = (
    local_vpp_disp
    / local_reference
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    ratio_scan,
    local_vpp_abs_norm,
    label="Absorption"
)

plt.plot(
    ratio_scan,
    local_vpp_disp_norm,
    linestyle="--",
    label="Dispersion"
)

plt.xlabel(
    r"$f_m/\Gamma$"
)

plt.ylabel(
    "Near-zero peak-to-peak "
    "(common normalization)"
)

plt.title(
    "Peak-to-peak amplitude "
    "between extrema nearest zero"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 15. 解调相位演示
#     修正逐条归一化问题
# ============================================================

fm_phase = 12.5

phase_deg_list = [
    0,
    30,
    60,
    90
]

phase_signals = []

for phi_deg in phase_deg_list:

    phi = np.deg2rad(
        phi_deg
    )

    S = mts_demodulated_signal(
        detuning,
        Gamma,
        fm_phase,
        beta,
        phi,
        C
    )

    phase_signals.append(
        S
    )


# 所有相位曲线使用同一个尺度
phase_common_scale = max(
    np.max(np.abs(s))
    for s in phase_signals
)


plt.figure(
    figsize=(8, 5)
)

for phi_deg, signal in zip(
    phase_deg_list,
    phase_signals
):

    plt.plot(
        detuning / Gamma,
        signal / phase_common_scale,
        label=f"{phi_deg}°"
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
    r"Detuning $\Delta/\Gamma$"
)

plt.ylabel(
    "Demodulated signal "
    "(common normalization)"
)

plt.title(
    "Effect of demodulation phase"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 16. 提取关键指标
# ============================================================

idx_abs_vpp = np.argmax(
    global_vpp_abs
)

idx_disp_vpp = np.argmax(
    global_vpp_disp
)

idx_abs_slope_max = np.argmax(
    slope_abs
)

idx_abs_slope_min = np.argmin(
    slope_abs
)

idx_disp_slope_max = np.argmax(
    slope_disp
)


ratio_abs_vpp = (
    ratio_scan[idx_abs_vpp]
)

ratio_disp_vpp = (
    ratio_scan[idx_disp_vpp]
)

ratio_abs_slope = (
    ratio_scan[idx_abs_slope_max]
)

ratio_abs_slope_negative = (
    ratio_scan[idx_abs_slope_min]
)

ratio_disp_slope = (
    ratio_scan[idx_disp_slope_max]
)


vpp_ratio = (
    np.max(global_vpp_abs)
    / np.max(global_vpp_disp)
)

slope_ratio = (
    np.max(slope_abs)
    / np.max(slope_disp)
)


# ============================================================
# 17. 找过零斜率变号位置
# ============================================================

def find_sign_change_positions(
    x,
    y
):
    """
    用线性插值估计 y(x)=0 的位置。
    """

    idx = np.where(
        y[:-1] * y[1:] < 0
    )[0]

    roots = []

    for i in idx:

        x1 = x[i]
        x2 = x[i + 1]

        y1 = y[i]
        y2 = y[i + 1]

        root = (
            x1
            - y1
            * (x2 - x1)
            / (y2 - y1)
        )

        roots.append(root)

    return roots


abs_slope_zero = (
    find_sign_change_positions(
        ratio_scan,
        slope_abs
    )
)

disp_slope_zero = (
    find_sign_change_positions(
        ratio_scan,
        slope_disp
    )
)


# ============================================================
# 18. 打印验证结果
# ============================================================

print()
print(
    "======================================"
)
print(
    "2021 MTS validation results"
)
print(
    "======================================"
)

print()
print(
    "Global peak-to-peak:"
)

print(
    "Absorption maximum at fm/Gamma =",
    ratio_abs_vpp
)

print(
    "Dispersion maximum at fm/Gamma =",
    ratio_disp_vpp
)

print(
    "Max absorption Vpp / "
    "max dispersion Vpp =",
    vpp_ratio
)


print()
print(
    "Zero-crossing slope:"
)

print(
    "Absorption positive maximum at "
    "fm/Gamma =",
    ratio_abs_slope
)

print(
    "Absorption negative minimum at "
    "fm/Gamma =",
    ratio_abs_slope_negative
)

print(
    "Dispersion maximum at "
    "fm/Gamma =",
    ratio_disp_slope
)

print(
    "Max absorption slope / "
    "max dispersion slope =",
    slope_ratio
)


print()
print(
    "Slope sign-change positions:"
)

print(
    "Absorption:",
    abs_slope_zero
)

print(
    "Dispersion:",
    disp_slope_zero
)


print()
print(
    "Odd-symmetry error:"
)

print(
    "Maximum absorption odd error =",
    np.max(
        odd_error_abs
    )
)

print(
    "Maximum dispersion odd error =",
    np.max(
        odd_error_disp
    )
)


# ============================================================
# 19. 对照论文图2的几个关键点
# ============================================================

print()
print(
    "======================================"
)
print(
    "Near-zero Vpp at paper reference points"
)
print(
    "======================================"
)

reference_ratios = [
    0.74,
    1.48,
    2.55,
    2.72
]

for r in reference_ratios:

    abs_value = np.interp(
        r,
        ratio_scan,
        local_vpp_abs_norm
    )

    disp_value = np.interp(
        r,
        ratio_scan,
        local_vpp_disp_norm
    )

    print(
        f"fm/Gamma = {r:.2f}: "
        f"Abs = {abs_value:.3f}, "
        f"Disp = {disp_value:.3f}"
    )


# ============================================================
# 20. 显示全部图
# ============================================================

plt.show()