# MEC-024 Rolling Without Slipping（纯滚动）

纯滚动：圆形刚体（球、圆柱等）在斜面上做纯滚动，通过运动学约束 v_cm = R·ω
耦合平动（MEC-020）和转动（MEC-021）。是 MEC-020 和 MEC-021 的自然综合模型。

## 物理背景

纯滚动是刚体平动与转动耦合的经典案例。核心约束 v_cm = R·ω 是几何约束，
表示接触点瞬时速度为零。维持此约束需要静摩擦力，但理想静摩擦不做功
（接触点不动），因此机械能守恒。

**纯滚动 ≠ 无摩擦**：无摩擦时刚体只滑动不旋转，不满足 v_cm = R·ω。
静摩擦力是维持约束的约束力，而非耗散力。

**假设**：
- 圆形刚体（球、圆柱、圆筒等），质量 `m > 0`，半径 `R > 0`
- 转动惯量 `I > 0`（绕质心轴）
- 重力加速度 `g > 0`
- 斜面倾角 `alpha ≥ 0`
- 纯滚动约束严格成立（无滑动）
- 静摩擦不做功（理想条件）

## 与 MEC-020/021 的关系

| 特性 | MEC-020（平动） | MEC-021（转动） | MEC-024（纯滚动） |
|------|----------------|----------------|-------------------|
| 自由度 | 2D 平动 | 1D 转动 | 平动+转动耦合 |
| 约束 | 无 | 固定轴 | v_cm = R·ω |
| 有效质量 | m | I | m + I/R² |
| 退化 | — | — | 去约束+去转动 → MEC-020 |

**退化关系**：取消滚动约束、去除转动自由度后，平动部分退化为 MEC-020
（a = g·sin(α)，纯滑动无转动）。

**注意**：不将 I→0 或 I→∞ 作为退化。I→0 时转动动能→0 但约束仍在；
I→∞ 时 a→0 且 ω→0，物体静止。这些不是严格退化。

## 数学模型

- 状态：质心位移 `x_cm`、转角 `theta`、质心速度 `v_cm`、角速度 `omega`
- 参数：`m, I, R, g, alpha`

纯滚动约束代入后的运动方程：

```
dx_cm/dt = v_cm
dtheta/dt = omega
dv_cm/dt = a = g·sin(α) / (1 + I/(mR²))
domega/dt = a / R
```

## 状态空间表示

```
state = [x_cm, theta, v_cm, omega]
```

- `x_cm` —— 质心沿斜面位移（m）
- `theta` —— 转角（rad）
- `v_cm` —— 质心速度（m/s）
- `omega` —— 角速度（rad/s）

## 微分方程推导

**运动方程**：
- 平动（沿斜面）：`m·a = m·g·sin(α) - f`（f 为静摩擦力）
- 转动（绕质心）：`I·α_rot = f·R`
- 约束：`a = R·α_rot`

联立求解（消去 f 和 α_rot）：

$$
a = \frac{g \sin\alpha}{1 + I/(mR^2)}
$$

**有效质量**：`m_eff = m + I/R²`

## 解析解

恒加速度：

$$
\begin{aligned}
x_{\text{cm}}(t) &= x_0 + v_0 t + \frac{1}{2} a t^2 \\
\theta(t) &= \theta_0 + \frac{v_0}{R} t + \frac{1}{2}\frac{a}{R} t^2
\end{aligned}
$$

## 守恒量

**机械能**：

$$
E = \frac{1}{2} m v_{\text{cm}}^2 + \frac{1}{2} I \omega^2 + mg h
$$

其中 h = -x_cm·sin(α)。纯滚动时静摩擦不做功，机械能守恒。

## 典型转动惯量

| 物体 | I/(mR²) | a/g·sin(α) |
|------|---------|------------|
| 实心球 | 2/5 | 5/7 ≈ 0.714 |
| 实心圆柱 | 1/2 | 2/3 ≈ 0.667 |
| 薄壁圆筒 | 1 | 1/2 = 0.500 |

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `m` | 质量 (kg) | `1.0` |
| `I` | 转动惯量 (kg·m²) | `0.1` |
| `R` | 半径 (m) | `0.5` |
| `g` | 重力加速度 (m/s²) | `9.81` |
| `alpha` | 斜面倾角 (rad) | `π/6` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：纯滚动动力学 `dynamics`（约束已代入）+ 解析解 `analytical` + 加速度/有效质量/机械能工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC024_consistency.py` | 数值解 vs 解析解 + 约束验证 + 能量守恒 + 加速度公式 + 典型 I/(mR²) + 非法参数 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC024_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, acceleration

g, alpha = 9.81, np.radians(30)
m, R = 1.0, 0.5
t_eval = np.linspace(0, 3, 100)

fig, ax = plt.subplots(figsize=(8, 5))
for name, I_ratio in [("solid sphere", 0.4), ("solid cylinder", 0.5), ("thin tube", 1.0)]:
    I = I_ratio * m * R ** 2
    a = acceleration(g, m, I, R, alpha)
    sol = solve_ivp(dynamics, (0, 3), [0, 0, 0, 0],
                    t_eval=t_eval, args=(m, I, R, g, alpha),
                    rtol=1e-9, atol=1e-12)
    ax.plot(sol.t, sol.y[0], label=f"{name} (a={a:.2f})")

ax.set_title("MEC-024: Rolling Without Slipping")
ax.set_xlabel("t (s)")
ax.set_ylabel("x_cm (m)")
ax.legend()
ax.grid(True)
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
