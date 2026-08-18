import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv


# ============================================================
# Stage D-1
# 87Rb D2 hyperfine-resolved MTS bridge model
# ============================================================
#
# 目标：
#
# 1. 从一般单跃迁 MTS 进入真实 87Rb D2 超精细结构
# 2. 加入 F=2 -> F'=1,2,3 的真实频率位置
# 3. 加入相对超精细跃迁强度
# 4. 保留调制频率、调制度、解调相位
# 5. 扫描 modulation frequency
# 6. 评价 F=2 -> F'=3 锁点的：
#
#       amplitude
#       center slope
#       zero crossing
#       optimum demodulation phase
#
# 注意：
# 本阶段尚未包含 mF Zeeman 子能级和 optical pumping。
# ============================================================


# ============================================================
# 0. 87Rb D2 参数
# ============================================================

# ------------------------------------------------------------
# 频率单位全部使用 MHz
#
# 把 F=2 -> F'=3 定义为 0 MHz
# ------------------------------------------------------------

TRANSITIONS = {
    "F=2 -> F'=1": {
        "offset": -423.597,
        "strength": 1.0 / 20.0
    },

    "F=2 -> F'=2": {
        "offset": -266.650,
        "strength": 1.0 / 4.0
    },

    "F=2 -> F'=3": {
        "offset": 0.0,
        "strength": 7.0 / 10.0
    }
}


# ============================================================
# 1. 有效均匀线宽
# ============================================================
#
# 这里延续前面的论文记号，
# 使用 Γ_eff = 6.065 MHz 作为第一版基准。
#
# 注意：
# 它在当前代码中是 Lorentzian 复分母的宽度参数。
#
# 实验中应使用实际测得的 sub-Doppler effective linewidth，
# 后续 Stage D-2 再加入 power broadening。
# ============================================================

GAMMA_EFF = 6.065


# ============================================================
# 2. EOM / MTS 参数
# ============================================================

# 先采用之前文献使用的标准点
FM = 12.5          # MHz

# phase modulation depth
BETA = 0.28

# 一阶边带近似的 Bessel 因子
J0 = jv(0, BETA)
J1 = jv(1, BETA)

BESSEL_FACTOR = J0 * J1


# ============================================================
# 3. 扫描范围
# ============================================================

detuning = np.linspace(
    -500.0,
    +100.0,
    12001
)


# ============================================================
# 4. 单跃迁 MTS 复线型
#
# 延续已经通过 Stage B 验证的 S1(delta) 结构
# ============================================================

def mts_single_complex(
    Delta,
    fm,
    gamma
):
    """
    单个近似两能级跃迁的复 MTS 响应。

    对应之前已经验证过的 S1(delta) 闭式结构。

    Delta:
        激光相对于该跃迁中心的失谐 MHz

    fm:
        调制频率 MHz

    gamma:
        有效均匀线宽参数 MHz

    返回：
        complex MTS response
    """

    delta = fm

    gamma_a = gamma
    gamma_b = gamma
    gamma_ab = gamma


    # --------------------------------------------------------
    # 布居响应
    # --------------------------------------------------------

    population_factor = (

        1.0 / (
            gamma_a
            + 1j * delta
        )

        +

        1.0 / (
            gamma_b
            + 1j * delta
        )
    )


    # --------------------------------------------------------
    # 四个 modulation-transfer 共振分母
    # --------------------------------------------------------

    bracket = (

        1.0 / (
            gamma_ab
            + 1j * (
                Delta
                + delta / 2.0
            )
        )

        -

        1.0 / (
            gamma_ab
            + 1j * (
                Delta
                + delta
            )
        )

        +

        1.0 / (
            gamma_ab
            - 1j * (
                Delta
                - delta
            )
        )

        -

        1.0 / (
            gamma_ab
            - 1j * (
                Delta
                - delta / 2.0
            )
        )
    )


    return (
        -population_factor
        * bracket
    )


# ============================================================
# 5. 一个超精细跃迁的贡献
# ============================================================

def hyperfine_component(
    laser_detuning,
    offset,
    strength,
    fm,
    gamma
):
    """
    laser_detuning：
        相对于 F=2 -> F'=3 的频率

    offset：
        当前超精细跃迁相对于 F'=3 的频移
    """

    local_detuning = (
        laser_detuning
        - offset
    )


    return (

        strength
        * BESSEL_FACTOR
        * mts_single_complex(
            local_detuning,
            fm,
            gamma
        )
    )


# ============================================================
# 6. 总 87Rb MTS 信号
# ============================================================

def rb87_mts_complex(
    laser_detuning,
    fm=FM,
    gamma=GAMMA_EFF
):

    total = np.zeros_like(
        laser_detuning,
        dtype=np.complex128
    )

    components = {}


    for name, p in TRANSITIONS.items():

        component = hyperfine_component(

            laser_detuning,

            p["offset"],

            p["strength"],

            fm,

            gamma
        )


        components[name] = component

        total += component


    return (
        total,
        components
    )


# ============================================================
# 7. 解调
# ============================================================

def demodulate(
    S,
    phi
):

    return np.real(
        S
        * np.exp(
            1j * phi
        )
    )


# ============================================================
# 8. F'=3 中心复斜率
# ============================================================

def complex_center_slope(
    x,
    S,
    x0=0.0,
    window=1.0
):

    mask = (
        np.abs(
            x - x0
        )
        <= window
    )


    xx = (
        x[mask]
        - x0
    )


    cr = np.polyfit(
        xx,
        np.real(
            S[mask]
        ),
        3
    )

    ci = np.polyfit(
        xx,
        np.imag(
            S[mask]
        ),
        3
    )


    return (
        cr[-2]
        + 1j * ci[-2]
    )


# ============================================================
# 9. 最佳解调相位
# ============================================================

def optimum_phase(
    x,
    S,
    x0=0.0
):

    slope = complex_center_slope(
        x,
        S,
        x0
    )

    phi = (
        -np.angle(
            slope
        )
    )


    return (
        phi,
        np.abs(
            slope
        )
    )


# ============================================================
# 10. 零点
# ============================================================

def zero_crossing_near(
    x,
    y,
    target=0.0,
    window=10.0
):

    mask = (
        np.abs(
            x - target
        )
        < window
    )

    xx = x[mask]
    yy = y[mask]


    ids = np.where(
        yy[:-1] * yy[1:] <= 0
    )[0]


    if len(ids) == 0:

        return np.nan


    candidates = (

        xx[ids]
        + xx[ids + 1]

    ) / 2.0


    k = ids[
        np.argmin(
            np.abs(
                candidates - target
            )
        )
    ]


    x1 = xx[k]
    x2 = xx[k + 1]

    y1 = yy[k]
    y2 = yy[k + 1]


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
# 11. 基准 MTS
# ============================================================

S_total, S_components = (
    rb87_mts_complex(
        detuning
    )
)


phi_opt, slope_opt = (
    optimum_phase(
        detuning,
        S_total,
        x0=0.0
    )
)


V_total = demodulate(
    S_total,
    phi_opt
)


# ============================================================
# 12. 共同归一化
# ============================================================

COMMON_SCALE = np.max(
    np.abs(
        V_total
    )
)


# ============================================================
# Figure 1
#
# 三个真实 hyperfine MTS 分量
# ============================================================

plt.figure(
    figsize=(9, 5)
)


for name, component in S_components.items():

    V = demodulate(
        component,
        phi_opt
    )

    plt.plot(
        detuning,
        V / COMMON_SCALE,
        label=name
    )


plt.axhline(
    0,
    linewidth=0.8
)


for name, p in TRANSITIONS.items():

    plt.axvline(
        p["offset"],
        linestyle="--",
        linewidth=0.6
    )


plt.xlabel(
    "Laser detuning from "
    r"$F=2\rightarrow F'=3$ (MHz)"
)

plt.ylabel(
    "MTS signal "
    "(common normalization)"
)

plt.title(
    r"$^{87}$Rb D2 hyperfine MTS components"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 2
#
# 总 Rb87 MTS
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.plot(
    detuning,
    V_total / COMMON_SCALE,
    label="Total MTS"
)


plt.axhline(
    0,
    linewidth=0.8
)


for name, p in TRANSITIONS.items():

    plt.axvline(
        p["offset"],
        linestyle="--",
        linewidth=0.7,
        label=name
    )


plt.xlabel(
    "Laser detuning from "
    r"$F=2\rightarrow F'=3$ (MHz)"
)

plt.ylabel(
    "Normalized MTS signal"
)

plt.title(
    rf"$^{{87}}$Rb D2 MTS, "
    rf"$f_m={FM}$ MHz"
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
# 放大 F=2 -> F'=3 锁频区域
# ============================================================

zoom = (
    np.abs(
        detuning
    )
    <= 40.0
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    detuning[zoom],
    (
        V_total[zoom]
        / COMMON_SCALE
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
    r"Detuning from $F=2\rightarrow F'=3$ (MHz)"
)

plt.ylabel(
    "Normalized MTS signal"
)

plt.title(
    "Cycling-transition locking region"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 4
#
# 解调相位影响
# ============================================================

phases_deg = [
    0,
    30,
    60,
    90
]


plt.figure(
    figsize=(8, 5)
)


for pdeg in phases_deg:

    phi = np.radians(
        pdeg
    )


    V = demodulate(
        S_total,
        phi
    )


    # 共同归一化，不能各自归一化
    plt.plot(
        detuning[zoom],
        V[zoom] / COMMON_SCALE,
        label=f"{pdeg}°"
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
    "MTS signal "
    "(common normalization)"
)

plt.title(
    "Demodulation-phase dependence"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 13. 调制频率扫描
# ============================================================

fm_scan = np.linspace(
    1.0,
    30.0,
    300
)


slope_scan = []
amplitude_scan = []

phase_scan = []
zero_scan = []


# ------------------------------------------------------------
# 注意：
# 每个 fm 允许重新选择最佳相位。
#
# 这是寻找理论最佳 operating point。
# ------------------------------------------------------------

for fm_i in fm_scan:

    S_i, _ = rb87_mts_complex(
        detuning,
        fm=fm_i
    )


    phi_i, slope_i = (
        optimum_phase(
            detuning,
            S_i,
            x0=0.0
        )
    )


    V_i = demodulate(
        S_i,
        phi_i
    )


    # F'=3 附近 ±40 MHz
    local = (
        np.abs(
            detuning
        )
        < 40.0
    )


    amp_i = np.max(
        np.abs(
            V_i[local]
        )
    )


    zero_i = zero_crossing_near(
        detuning,
        V_i,
        target=0.0,
        window=10.0
    )


    slope_scan.append(
        slope_i
    )

    amplitude_scan.append(
        amp_i
    )

    phase_scan.append(
        np.degrees(
            phi_i
        )
    )

    zero_scan.append(
        zero_i
    )


slope_scan = np.array(
    slope_scan
)

amplitude_scan = np.array(
    amplitude_scan
)

phase_scan = np.unwrap(
    np.radians(
        phase_scan
    )
)

phase_scan = np.degrees(
    phase_scan
)

zero_scan = np.array(
    zero_scan
)


# ============================================================
# 14. modulation-frequency maxima
# ============================================================

idx_slope = np.argmax(
    slope_scan
)

idx_amp = np.argmax(
    amplitude_scan
)


fm_slope_opt = (
    fm_scan[idx_slope]
)

fm_amp_opt = (
    fm_scan[idx_amp]
)


# ============================================================
# Figure 5
#
# modulation frequency:
# slope and amplitude
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    fm_scan,
    slope_scan
    / np.max(
        slope_scan
    ),
    label="Center slope"
)


plt.plot(
    fm_scan,
    amplitude_scan
    / np.max(
        amplitude_scan
    ),
    label="Local peak amplitude"
)


plt.axvline(
    fm_slope_opt,
    linestyle="--",
    linewidth=0.8,
    label=(
        "Slope optimum "
        f"{fm_slope_opt:.2f} MHz"
    )
)


plt.axvline(
    fm_amp_opt,
    linestyle=":",
    linewidth=0.8,
    label=(
        "Amplitude optimum "
        f"{fm_amp_opt:.2f} MHz"
    )
)


plt.xlabel(
    "Modulation frequency (MHz)"
)

plt.ylabel(
    "Normalized metric"
)

plt.title(
    r"$^{87}$Rb $F=2\rightarrow F'=3$ "
    "MTS optimization"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# Figure 6
#
# 最佳解调相位和锁点
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(8, 5)
)


line1 = ax1.plot(
    fm_scan,
    phase_scan,
    label="Optimum demod phase"
)

ax1.set_xlabel(
    "Modulation frequency (MHz)"
)

ax1.set_ylabel(
    "Optimum phase (deg)"
)


ax2 = ax1.twinx()


line2 = ax2.plot(
    fm_scan,
    zero_scan,
    label="Lock zero"
)

ax2.set_ylabel(
    "Zero crossing shift (MHz)"
)


ax2.axhline(
    0,
    linestyle="--",
    linewidth=0.8
)


lines = line1 + line2

labels = [
    x.get_label()
    for x in lines
]


ax1.legend(
    lines,
    labels,
    loc="best"
)


plt.title(
    "Demodulation phase and lock point"
)

ax1.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 15. Bessel information
# ============================================================

print()
print(
    "===================================================="
)

print(
    "Stage D-1: 87Rb D2 hyperfine MTS"
)

print(
    "===================================================="
)


print()
print(
    "Hyperfine transitions:"
)


for name, p in TRANSITIONS.items():

    print(
        f"{name:18s} "
        f"offset = {p['offset']:9.3f} MHz, "
        f"strength = {p['strength']:.5f}"
    )


print()
print(
    "EOM:"
)

print(
    "beta =",
    BETA
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
    BESSEL_FACTOR
)


print()
print(
    "Reference operating point:"
)

print(
    "fm =",
    FM,
    "MHz"
)

print(
    "Gamma_eff =",
    GAMMA_EFF,
    "MHz"
)

print(
    "fm/Gamma_eff =",
    FM / GAMMA_EFF
)

print(
    "Optimum demodulation phase = "
    f"{np.degrees(phi_opt):.3f} deg"
)

print(
    "F'=3 center slope =",
    slope_opt
)


zero_ref = zero_crossing_near(
    detuning,
    V_total,
    target=0.0,
    window=10.0
)


print(
    "F'=3 zero crossing =",
    zero_ref,
    "MHz"
)


print()
print(
    "Modulation-frequency scan:"
)

print(
    "Maximum center slope at fm = "
    f"{fm_slope_opt:.3f} MHz"
)

print(
    "Maximum local amplitude at fm = "
    f"{fm_amp_opt:.3f} MHz"
)


print()
print(
    "All finite =",
    (
        np.all(
            np.isfinite(
                V_total
            )
        )
        and
        np.all(
            np.isfinite(
                slope_scan
            )
        )
    )
)

print(
    "===================================================="
)


plt.show()