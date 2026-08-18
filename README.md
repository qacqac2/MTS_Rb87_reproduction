# Rb87 调制转移光谱（MTS）三阶密度矩阵仿真复现

## 1. 项目简介

本项目用于逐步复现并扩展调制转移光谱（MTS, Modulation Transfer Spectroscopy，调制转移光谱）的理论模型。

整体路线从最简单的 Lorentzian（洛伦兹）吸收/色散线型出发，逐步加入：EOM（Electro-Optic Modulator，电光调制器）边带与解调、1990/1993 文献解析模型、Doppler（多普勒）效应、`87Rb` D2 线超精细结构、Zeeman（塞曼）子能级、OBE（Optical Bloch Equations，光学布洛赫方程）、显式三阶密度矩阵 `rho^(3)`、QuTiP 完整主方程交叉验证、Maxwell–Boltzmann（麦克斯韦–玻尔兹曼）热速度平均，以及最终热原子 MTS 鉴频信号与锁点数值收敛检查。

项目最终物理主线为：

```text
rho^(0) -> rho^(1) -> rho^(2) -> rho^(3)
        -> P^(3) -> E_± -> Z -> V_MTS
```

其中：

- **13**：最终原子物理核心；
- **14c**：最终完整热原子物理模型；
- **14d**：最终锁点的局部频率网格数值检查。

## 2. 目录结构

```text
MTS_Rb87_reproduction/
│
├─ README.md
├─ requirements.txt
│
├─ scripts/
│   ├─ 01_mts_lorentzian.py
│   ├─ 02_mts_simplified_2021.py
│   ├─ 03_mts_1993_s1_s2.py
│   ├─ 04_mts_1990_reference.py
│   ├─ 05_mts_doppler.py
│   ├─ 06_rb87_hyperfine_mts.py
│   ├─ 07_rb87_zeeman_obe_power.py
│   ├─ 08_rb87_mts_time_domain.py
│   ├─ 09_mts_rho3_chain.py
│   ├─ 10_mts_rho3_vs_qutip.py
│   ├─ 11_mts_doppler_average.py
│   ├─ 12_rb87_zeeman_rho3.py
│   ├─ 13_rb87_hyperfine_zeeman_mts.py
│   ├─ 13b_rb87_hyperfine_refine.py
│   ├─ 14c_rb87_full_thermal_mts_composite.py
│   └─ 14d_rb87_thermal_lock_spotcheck.py
│
├─ figures/
│   ├─ key_figures/
│   └─ all_figures/
│
├─ results/
│   ├─ final_results.md
│   └─ validation_summary.md
│
├─ data/
│
├─ references/
│
└─ archive/
    ├─ 14_rb87_full_thermal_mts.py
    └─ 14b_rb87_full_thermal_mts_converged.py
```

## 3. 脚本说明

| 脚本 | 主要作用 |
|---|---|
| `01_mts_lorentzian.py` | 建立最基本的 Lorentzian 吸收/色散线型、边带平移与差分鉴频概念 |
| `02_mts_simplified_2021.py` | 加入 Bessel（贝塞尔）边带、调制频率、调制度和解调相位，研究 MTS 线型与斜率 |
| `03_mts_1993_s1_s2.py` | 复现 1993 文献中的 `S1`、`S2` 单光子/双光子 MTS 解析结构 |
| `04_mts_1990_reference.py` | 复现 1990 文献中的相关双光子 MTS 参考模型 |
| `05_mts_doppler.py` | 在早期解析模型中研究 Doppler 展宽与热速度效应 |
| `06_rb87_hyperfine_mts.py` | 引入 `87Rb` D2 线 `F=2 -> F'=1,2,3` 超精细频率与相对线强 |
| `07_rb87_zeeman_obe_power.py` | 建立 Zeeman 分辨 OBE、光抽运、饱和和功率展宽的原子动力学模型 |
| `08_rb87_mts_time_domain.py` | 用时域 OBE 计算相位调制泵浦、反向弱探测和 MTS 外差信号 |
| `09_mts_rho3_chain.py` | 显式构造 `rho^(0) -> rho^(3) -> P^(3) -> 生成边带 -> 外差` 的三阶主链 |
| `10_mts_rho3_vs_qutip.py` | 用 QuTiP（Python 量子工具箱）完整主方程的弱场极限交叉验证显式 `rho^(3)` |
| `11_mts_doppler_average.py` | 对经过验证的三阶响应进行带传播方向信息的热速度平均 |
| `12_rb87_zeeman_rho3.py` | 将通用三能级三阶模型替换为真实 `87Rb` Zeeman 分辨三阶模型 |
| `13_rb87_hyperfine_zeeman_mts.py` | 建立完整超精细 + Zeeman 分辨的 `87Rb` 三阶原子物理核心 |
| `13b_rb87_hyperfine_refine.py` | 检查局部频率网格收敛及渡越弛豫参数敏感性 |
| `14c_rb87_full_thermal_mts_composite.py` | 用固定复合速度网格完成最终热原子 MTS 计算 |
| `14d_rb87_thermal_lock_spotcheck.py` | 固定物理模型与速度网格，仅细化激光频率网格，对最终锁点做局部数值检查 |

## 4. 项目主线

### 4.1 基础与文献模型：01–05

建立吸收与色散响应、EOM 边带、差分鉴频、解调相位、1990/1993 文献解析 MTS，以及早期 Doppler 模型。

### 4.2 真实 Rb 与完整 OBE 路线：06–08

逐步加入超精细结构、Zeeman 子能级、Clebsch–Gordan（克莱布什–戈尔丹）系数、光抽运、饱和、功率展宽和时域 MTS。

### 4.3 显式三阶理论主线：09–13

核心关系为：

```text
rho = rho^(0) + rho^(1) + rho^(2) + rho^(3) + ...
```

三阶原子相干产生三阶极化 `P^(3)`，并通过近简并四波混频在 probe（探测光）方向生成新边带。新边带与 probe carrier（探测载波）发生拍频，再经相敏解调得到 MTS 鉴频信号。

### 4.4 最终数值验证：13b–14d

主要检查原子稳态残差、三阶场幅标度、QuTiP 交叉验证、速度积分边界、速度网格收敛、外差线性、`beta=0` 零信号以及局部激光频率网格收敛。

## 5. 最终物理模型

最终原子核心 `13_rb87_hyperfine_zeeman_mts.py` 包含：

```text
Ground（基态）:
5S1/2, Fg = 1, 2

Excited（激发态）:
5P3/2, Fe = 1, 2, 3
```

`14c` 在此基础上进一步进行 Maxwell–Boltzmann 热速度平均。

## 6. 关键计算条件与结果

最终热模型代表性条件：

```text
Atom: 87Rb D2
Temperature: 300 K
Modulation frequency: 12.5 MHz
Phase modulation index beta: 0.28
Gamma_transit / Gamma: 0.02
```

### 14c 热原子结果

```text
Thermal optimum demodulation phase:
phi ≈ 39.18698 deg

Refined thermal lock point:
Delta_nu_lock ≈ -1.61768 MHz

Velocity-grid convergence:
dq_local = 0.05 -> 0.025
relative vector error ≈ 0.493 %

Boundary convergence:
±4u -> ±5u ≈ 6.2e-12

Heterodyne linearity:
~1e-14

beta = 0 null test:
PASS
```

### 14d 局部频率网格检查

```text
Frequency step:
0.250 -> 0.125 -> 0.0625 MHz

Lock point:
-1.6173244
-> -1.6173309
-> -1.6173306 MHz

Last refinement change:
≈ 0.296 Hz
```

## 7. 验证摘要

当前项目已完成的主要一致性检查：

1. `beta = 0` 时 MTS 信号消失；
2. 显式三阶响应满足 `s^3` 场幅标度；
3. 09 的显式 `rho^(3)` 与 10 的完整 QuTiP 弱场三阶结果一致；
4. 自发辐射通道满足衰减一致性；
5. `rho^(0)` 稳态残差足够小；
6. Doppler 积分边界收敛；
7. 热速度网格收敛；
8. “先热平均复边带、后外差”与直接平均外差信号满足线性一致性；
9. 最终锁点局部激光频率网格高度收敛。

这些检查说明：**当前数值程序能够稳定求解所建立的模型。**

## 8. 运行环境

建议使用 Python 3.10+。

主要依赖：

```text
numpy
scipy
matplotlib
sympy
qutip
```

安装：

```bash
pip install -r requirements.txt
```


## 9. Archive（归档）说明

以下脚本保留在 `archive/`：

```text
14_rb87_full_thermal_mts.py
14b_rb87_full_thermal_mts_converged.py
```

它们记录了热速度积分算法从失败到定位问题的过程：

- `14`：共振相关速度网格随失谐改变，导致强相干抵消下积分不稳定；
- `14b`：改为固定全局均匀速度网格，边界与外差线性通过，但速度分辨率仍未收敛；
- `14c`：最终采用固定复合速度网格，对窄共振速度类局部加密，完成收敛。


## 10. 模型限制

主要限制包括：

- 最终主模型采用 weak-field third-order perturbation（弱场三阶微扰）；
- transit relaxation（渡越弛豫）为现象学参数，且对锁点、斜率和幅度有明显影响；
- 当前信号主要为归一化模型单位，未映射到实验中的 V/Hz；
- 未完整纳入实际实验中的功率饱和、光束腰、磁场残差、偏振杂质、RAM（Residual Amplitude Modulation，残余幅度调制）、探测器和电子学增益等效应；
- 数值收敛证明当前模型已经被稳定求解，不等同于真实实验锁点具有相同数量级的物理精度。

## 11. 参考资料

- `01_1993_MTS_line_shape_theory.pdf`：整个 03 → 05 → 09 → 14c 理论路线最核心的文献：级联三能级、密度矩阵三阶微扰、$S_1/S_2$、Doppler 平均、四波混频和外差检测都从这里发展出来。
- `02_1990_Doppler_Free_Two_Photon_MTS.pdf`：项目的物理主链：相位调制泵浦产生边带，反向未调制探测光进入非线性介质，近简并四波混频在探测方向生成新边带，再与探测载波发生外差拍频。
- `03_2021_Rb87_MTS_multi_parameter_locking.pdf`：它和我们的实际 $^{87}\mathrm{Rb}$ D2 稳频条件最接近，直接研究了 $F=2\rightarrow F'=3$、调制频率、解调相位、偏振、磁场等参数；文中实验采用 $12.5$ MHz 调制频率，这也是我们项目采用这一参数的重要现实依据。