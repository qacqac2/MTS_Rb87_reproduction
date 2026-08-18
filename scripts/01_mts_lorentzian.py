import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 基本参数
# ============================================================

# 87Rb D2 线宽尺度，单位 MHz
Gamma = 6.065

# EOM 调制频率，单位 MHz
fm = 12.5

# 激光相对于原子共振中心的失谐
# 从 -40 MHz 扫描到 +40 MHz
detuning = np.linspace(-40, 40, 4001)


# ============================================================
# 2. 定义 Lorentzian 吸收型函数 L_n
# ============================================================

def L(detuning, n, fm, Gamma):
    """
    Lorentzian 吸收型响应。

    detuning : 激光失谐 Δ，单位 MHz
    n        : 第 n 个频率分量的位置编号
    fm       : 调制频率，单位 MHz
    Gamma    : 线宽参数，单位 MHz
    """

    x = detuning - n * fm

    return Gamma**2 / (Gamma**2 + x**2)


# ============================================================
# 3. 定义 Lorentzian 色散型函数 D_n
# ============================================================

def D(detuning, n, fm, Gamma):
    """
    Lorentzian 色散型响应。
    """

    x = detuning - n * fm

    return Gamma * x / (Gamma**2 + x**2)


# ============================================================
# 4. 计算未平移的原子响应
# ============================================================

L0 = L(detuning, 0, fm, Gamma)
D0 = D(detuning, 0, fm, Gamma)


# ============================================================
# 5. 计算上下边带对应的响应
# ============================================================

L_plus = L(detuning, +1, fm, Gamma)
L_minus = L(detuning, -1, fm, Gamma)

D_plus = D(detuning, +1, fm, Gamma)
D_minus = D(detuning, -1, fm, Gamma)


# ============================================================
# 6. 画图：原始吸收和色散响应
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(detuning, L0, label="Absorption $L_0$")
plt.plot(detuning, D0, label="Dispersion $D_0$")

plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.xlabel("Detuning Δ (MHz)")
plt.ylabel("Normalized response")
plt.title("Lorentzian absorption and dispersion")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 7. 画图：调制产生的正、负边带响应
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(detuning, L_minus, label="$L_{-1}$")
plt.plot(detuning, L0, label="$L_0$")
plt.plot(detuning, L_plus, label="$L_{+1}$")

plt.axvline(-fm, linestyle="--", linewidth=0.8)
plt.axvline(0, linestyle="--", linewidth=0.8)
plt.axvline(+fm, linestyle="--", linewidth=0.8)

plt.xlabel("Carrier detuning Δ (MHz)")
plt.ylabel("Normalized absorption response")
plt.title(f"Shifted Lorentzian components, fm = {fm} MHz")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 8. 画图：上下边带色散响应
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(detuning, D_minus, label="$D_{-1}$")
plt.plot(detuning, D0, label="$D_0$")
plt.plot(detuning, D_plus, label="$D_{+1}$")

plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.xlabel("Carrier detuning Δ (MHz)")
plt.ylabel("Normalized dispersion response")
plt.title(f"Shifted dispersive components, fm = {fm} MHz")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 9. 一个最简单的“差分鉴频”示意
# ============================================================

simple_error = L_minus - L_plus

# 中心附近数值斜率
dV_dnu = np.gradient(simple_error, detuning)

center_index = np.argmin(np.abs(detuning))

print("Gamma =", Gamma, "MHz")
print("fm =", fm, "MHz")
print("Error signal at resonance =", simple_error[center_index])
print(
    "Central slope =",
    dV_dnu[center_index],
    "per MHz"
)


plt.figure(figsize=(8, 5))

plt.plot(detuning, simple_error)

plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.xlabel("Detuning Δ (MHz)")
plt.ylabel("Difference signal")
plt.title("Simple sideband-difference discriminator")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()