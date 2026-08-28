# MEC-010 Mass-Spring（质量—弹簧简谐振子）

标准质量—弹簧简谐振子：质量 m 在弹性恢复力 F = -k·x 作用下沿一维方向运动。
这是 010 号段振动系统的基础模型，为后续 MEC-011（阻尼振子）、MEC-012（受迫振子）、
MEC-014（耦合振子）等建立标准规范，包括相图（phase portrait）约定。

## 物理背景

质量—弹簧系统是最基本的振动模型。弹簧的恢复力 F = -k·x（胡克定律）
驱动质量做简谐振动（simple harmonic motion, SHM）。运动方程

$$
\frac{d^2 x}{dt^2} + \omega_0^2 x = 0, \quad \omega_0 = \sqrt{\frac{k}{m}}
$$

是所有振动问题的起点：阻尼振子（MEC-011）在此基础上加 -bv 项，
受迫振子（MEC-012）在此基础上加驱动力，耦合振子（MEC-014）将其扩展到多自由度。

**假设**：
- 弹性恢复力 `F = -k·x`（k > 0，线性弹簧）
- 质量 `m > 0`
- 无阻尼、无外力
- 一维运动
- 平衡位置在 x = 0

## 数学模型

- 状态：位移 `x`、速度 `v`
- 参数：弹性系数 `k`、质量 `m`

一阶常微分方程：

```
dx/dt = v
dv/dt = -(k/m)·x
```

## 状态空间表示

```
state = [x, v]
```

其中：

- `x` —— 位移（m，相对于平衡位置）
- `v` —— 速度（m/s）

## 微分方程推导

弹簧恢复力：`F = -k·x`（胡克定律）

牛顿第二定律：`m·a = F`，因此 `a = -(k/m)·x`

$$
\frac{d^2 x}{dt^2} = -\frac{k}{m} x
$$

令 `ω₀ = √(k/m)`（固有角频率），得到简谐振动标准形式：

$$
\frac{d^2 x}{dt^2} + \omega_0^2 x = 0
$$

周期：

$$
T = \frac{2\pi}{\omega_0} = 2\pi\sqrt{\frac{m}{k}}
$$

## 解析解

$$
\begin{aligned}
x(t) &= x_0 \cos(\omega_0 t) + \frac{v_0}{\omega_0} \sin(\omega_0 t) \\
v(t) &= -x_0 \omega_0 \sin(\omega_0 t) + v_0 \cos(\omega_0 t)
\end{aligned}
$$

等价地，用振幅 A 和初相位 φ 表示：

$$
x(t) = A \cos(\omega_0 t + \varphi)
$$

其中：

$$
A = \sqrt{x_0^2 + \frac{v_0^2}{\omega_0^2}}, \quad
\varphi = \arctan2\!\left(\frac{-v_0}{\omega_0},\; x_0\right)
$$

## 守恒量

**机械能**：

$$
E = \frac{1}{2} m v^2 + \frac{1}{2} k x^2 = \frac{1}{2} k A^2
$$

机械能守恒是简谐振动的核心特征：动能和势能交替转换，总和恒定。

## 相图规范

本模型建立后续振动系统将沿用的相图（phase portrait）规范：

- **横轴**：位移 x（m）
- **纵轴**：速度 v（m/s）
- **轨迹**：等能量椭圆，方程为 `x²/A² + v²/(Aω₀)² = 1`
- **半轴**：x 方向半轴 = A，v 方向半轴 = A·ω₀
- **面积与能量的关系**：`S = π·A·Aω₀ = 2πE/(mω₀)`

无阻尼简谐振动的相图为闭合椭圆，表示周期运动。后续 MEC-011（阻尼振子）
的相图将退化为内旋螺线，MEC-012（受迫振子）的相图将趋向极限环。

## 初始状态约束

任意 `(x0, v0)` 都是合法初始状态。

仅要求：
- `k > 0`
- `m > 0`

**退化情形**：当 `x0=0, v0=0` 时 `A=0`，质点静止在平衡位置（零振幅）。

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `k` | 弹性系数 (N/m) | `1.0` |
| `m` | 质量 (kg) | `1.0` |
| `x0` | 初始位移 (m) | `1.0` |
| `v0` | 初始速度 (m/s) | `0.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：动力学方程 `dynamics` + 解析解 `analytical` + 角频率/周期/振幅/机械能工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC010_consistency.py` | 数值解 vs 解析解一致性测试 + 物理检验（含相图椭圆验证、周期标度律） |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC010_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, angular_frequency, period

k, m = 1.0, 1.0
x0, v0 = 1.0, 0.0
T = period(k, m)
t_eval = np.linspace(0, 2 * T, 401)
sol = solve_ivp(dynamics, (0, 2 * T), [x0, v0],
                t_eval=t_eval, args=(k, m), rtol=1e-9, atol=1e-12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 时域图
ax1.plot(sol.t, sol.y[0], label="x(t)")
ax1.plot(sol.t, sol.y[1], label="v(t)")
ax1.set_title("MEC-010: Mass-Spring (time domain)")
ax1.set_xlabel("t (s)")
ax1.legend()
ax1.grid(True)

# 相图
ax2.plot(sol.y[0], sol.y[1])
ax2.plot(0, 0, "ro", label="equilibrium")
ax2.set_title("MEC-010: Phase Portrait")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("v (m/s)")
ax2.legend()
ax2.grid(True)
ax2.axis("equal")

plt.tight_layout()
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
