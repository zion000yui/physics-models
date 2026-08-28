# MEC-011 Damped Oscillator（阻尼振子）

阻尼振子：质量 m 在弹性力 F = -k·x 和线性阻尼力 F_d = -b·v 共同作用下
沿一维方向运动。覆盖欠阻尼、临界阻尼和过阻尼三种状态，是 MEC-010
简谐振子的自然推广。

## 物理背景

真实物理系统中的振动总是伴随阻尼。最简单的阻尼模型是线性阻尼（粘性阻尼）
F_d = -b·v，与速度成正比、方向相反。引入阻尼后，系统的机械能不再守恒，
而是随时间单调耗散。根据阻尼比 ζ 的大小，系统表现出三种截然不同的行为。

**假设**：
- 弹性恢复力 `F = -k·x`（k > 0，线性弹簧）
- 线性阻尼力 `F_d = -b·v`（b ≥ 0，粘性阻尼）
- 质量 `m > 0`
- 一维运动
- 平衡位置在 x = 0

## 与 MEC-010 的关系

| 特性 | MEC-010（简谐振子） | MEC-011（阻尼振子） |
|------|---------------------|---------------------|
| 阻尼 | 无（b=0） | 线性阻尼 b·v |
| 阻尼比 | ζ=0 | 0 ≤ ζ < ∞ |
| 机械能 | 守恒 | 单调递减 |
| 相图 | 闭合椭圆 | 内旋螺线 / 单调趋向原点 |
| 周期 | T = 2π/ω₀ | 欠阻尼 T_d = 2π/ω_d |

当 b=0（ζ=0）时，MEC-011 精确退化为 MEC-010。

## 数学模型

- 状态：位移 `x`、速度 `v`
- 参数：弹性系数 `k`、质量 `m`、阻尼系数 `b`

一阶常微分方程：

```
dx/dt = v
dv/dt = -(k/m)·x - (b/m)·v
```

## 状态空间表示

```
state = [x, v]
```

其中：

- `x` —— 位移（m，相对于平衡位置）
- `v` —— 速度（m/s）

## 微分方程推导

合力 = 弹性力 + 阻尼力：

$$
\mathbf{F} = -k x - b v
$$

牛顿第二定律 `m·a = F`：

$$
\frac{d^2 x}{dt^2} + \frac{b}{m} \frac{dx}{dt} + \frac{k}{m} x = 0
$$

令 `ω₀ = √(k/m)`（固有角频率，同 MEC-010），`γ = b/(2m)`（衰减率）：

$$
\frac{d^2 x}{dt^2} + 2\gamma \frac{dx}{dt} + \omega_0^2 x = 0
$$

## 阻尼比与三种状态

**阻尼比**（damping ratio）：

$$
\zeta = \frac{b}{2\sqrt{mk}} = \frac{\gamma}{\omega_0}
$$

| 状态 | 条件 | 行为 |
|------|------|------|
| 欠阻尼（underdamped） | 0 ≤ ζ < 1 | 衰减振荡，角频率 ω_d = ω₀√(1-ζ²) |
| 临界阻尼（critically damped） | ζ = 1 | 以最快速度回到平衡，无振荡 |
| 过阻尼（overdamped） | ζ > 1 | 缓慢回到平衡，无振荡 |

## 解析解

**1. 欠阻尼（0 ≤ ζ < 1）**：

$$
x(t) = e^{-\gamma t} \left[ x_0 \cos(\omega_d t) + \frac{v_0 + \gamma x_0}{\omega_d} \sin(\omega_d t) \right]
$$

其中 `ω_d = ω₀√(1-ζ²)` 为阻尼角频率。

**2. 临界阻尼（ζ = 1）**：

$$
x(t) = e^{-\gamma t} \left[ x_0 + (v_0 + \gamma x_0) t \right]
$$

**3. 过阻尼（ζ > 1）**：

$$
x(t) = e^{-\gamma t} \left[ x_0 \cosh(\alpha t) + \frac{v_0 + \gamma x_0}{\alpha} \sinh(\alpha t) \right]
$$

其中 `α = √(γ² - ω₀²)`。

## 守恒量与耗散

**机械能**：

$$
E = \frac{1}{2} m v^2 + \frac{1}{2} k x^2
$$

**能量变化率**：

$$
\frac{dE}{dt} = -b v^2 \leq 0
$$

当 b > 0 时机械能单调递减；当 b = 0 时机械能守恒（退化为 MEC-010）。

## 相图规范（延续 MEC-010）

- **横轴**：位移 x（m）
- **纵轴**：速度 v（m/s）
- **欠阻尼**：内旋螺线（逐渐收缩的椭圆螺旋，趋向原点）
- **临界阻尼**：以最快速度趋向原点（无螺旋）
- **过阻尼**：缓慢单调趋向原点（无螺旋）
- **无阻尼（b=0）**：退化为 MEC-010 的闭合椭圆

## 初始状态约束

任意 `(x0, v0)` 都是合法初始状态。

仅要求：
- `k > 0`
- `m > 0`
- `b ≥ 0`

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `k` | 弹性系数 (N/m) | `1.0` |
| `m` | 质量 (kg) | `1.0` |
| `b` | 阻尼系数 (N·s/m) | `0.4` |
| `x0` | 初始位移 (m) | `1.0` |
| `v0` | 初始速度 (m/s) | `0.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：动力学方程 `dynamics` + 解析解 `analytical`（三种状态自动切换）+ 阻尼比/固有频率/阻尼频率/机械能工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC011_consistency.py` | 三种阻尼状态数值解 vs 解析解 + 无阻尼退化 + 能量耗散 + 相图轨迹验证 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC011_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, damping_ratio

k, m = 1.0, 1.0
x0, v0 = 1.0, 0.0
t_eval = np.linspace(0, 15, 401)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 时域：三种阻尼状态
for b, label in [(0.0, "no damping (MEC-010)"),
                  (0.4, "underdamped"),
                  (2.0, "critical"),
                  (4.0, "overdamped")]:
    sol = solve_ivp(dynamics, (0, 15), [x0, v0],
                    t_eval=t_eval, args=(k, m, b), rtol=1e-9, atol=1e-12)
    zeta = damping_ratio(k, m, b)
    ax1.plot(sol.t, sol.y[0], label=f"b={b}, ζ={zeta:.2f}")

ax1.set_title("MEC-011: Damped Oscillator (time domain)")
ax1.set_xlabel("t (s)")
ax1.set_ylabel("x (m)")
ax1.legend()
ax1.grid(True)

# 相图：三种阻尼状态
for b, label in [(0.4, "underdamped"), (2.0, "critical"), (4.0, "overdamped")]:
    sol = solve_ivp(dynamics, (0, 15), [x0, v0],
                    t_eval=t_eval, args=(k, m, b), rtol=1e-9, atol=1e-12)
    ax2.plot(sol.y[0], sol.y[1], label=label)

ax2.set_title("MEC-011: Phase Portrait")
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
