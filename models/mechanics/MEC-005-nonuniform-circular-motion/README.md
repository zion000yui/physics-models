# MEC-005 Non-Uniform Circular Motion（非匀速圆周运动）

非匀速圆周运动：质点沿固定半径 R 的圆周运动，角速度 ω 不再恒定，
而是随时间线性变化。这是 MEC-004 匀速圆周运动的自然推广，
用来验证「模型定义 → 引擎求解 → 解析解对照」闭环在变角速度下仍然成立。

## 物理背景

在匀速圆周运动中，速率恒定，向心加速度完全由向心力提供。
在非匀速圆周运动中，质点除了向心加速度外，还存在切向加速度，
使速率随时间变化。本模型仅引入**常数角加速度 α**，即匀变速圆周运动。

**假设**：
- 半径 R 恒定
- 圆心 (xc, yc) 恒定
- 角加速度 α 为常数（不随时间变化）
- 无空气阻力、无其他外力
- 运动限于二维平面

## 数学模型

- 状态：位置 `(x, y)`、速度 `(vx, vy)`
- 参数：半径 `R`、初始角速度 `omega0`、角加速度 `alpha`、圆心 `(xc, yc)`

一阶常微分方程：

```
dx/dt = vx
dy/dt = vy
dvx/dt = -ω(t)²·(x-xc) - α·(y-yc)
dvy/dt = -ω(t)²·(y-yc) + α·(x-xc)
```

其中 `ω(t) = ω₀ + α·t`。

## 状态空间表示

```
state = [x, y, vx, vy]
```

其中：

- `x`  —— 水平位置（m）
- `y`  —— 垂直位置（m）
- `vx` —— 水平速度（m/s）
- `vy` —— 垂直速度（m/s）

## 微分方程推导

$$
\frac{d}{dt}\begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}
=
\begin{bmatrix}
v_x \\
v_y \\
-\omega(t)^2 (x - x_c) - \alpha (y - y_c) \\
-\omega(t)^2 (y - y_c) + \alpha (x - x_c)
\end{bmatrix}
$$

其中：

- 法向加速度：`a_n = R·ω(t)²`，指向圆心
- 切向加速度：`a_t = R·α`，沿运动方向（θ 增加方向）
- 合成加速度：将法向和切向分量合成后，代入 `x-xc = R·cosθ, y-yc = R·sinθ` 得上述方程

## 解析解

$$
\begin{aligned}
\theta(t) &= \theta_0 + \omega_0 t + \tfrac{1}{2} \alpha t^2 \\
\omega(t) &= \omega_0 + \alpha t \\
x(t) &= x_c + R \cos\theta(t) \\
y(t) &= y_c + R \sin\theta(t) \\
v_x(t) &= -R \omega(t) \sin\theta(t) \\
v_y(t) &= R \omega(t) \cos\theta(t)
\end{aligned}
$$

其中 `θ₀ = arctan2(y₀ - yc, x₀ - xc)`。

## 初始状态约束

非匀速圆周运动要求初始状态满足：

$$
\begin{aligned}
(x_0 - x_c)^2 + (y_0 - y_c)^2 &= R^2 \\
v_{x0}(x_0 - x_c) + v_{y0}(y_0 - y_c) &= 0 \\
|\mathbf{v}_0| &= R|\omega_0|
\end{aligned}
$$

模型入口 `validate_initial_state()` 会在求解前验证上述条件，
不满足时抛出 `AssertionError`。

**退化情形**：`α = 0` 时，模型退化为 MEC-004 匀速圆周运动。

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `R` | 圆周半径 (m) | `1.0` |
| `omega0` | 初始角速度 (rad/s) | `1.0` |
| `alpha` | 角加速度 (rad/s²) | `0.0` |
| `xc` | 圆心水平坐标 (m) | `0.0` |
| `yc` | 圆心垂直坐标 (m) | `0.0` |
| `x0` | 初始水平位置 (m) | `1.0` |
| `y0` | 初始垂直位置 (m) | `0.0` |
| `vx0` | 初始水平速度 (m/s) | `0.0` |
| `vy0` | 初始垂直速度 (m/s) | `1.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：动力学方程 `dynamics` + 解析解 `analytical` + 初始状态验证 `validate_initial_state` |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC005_consistency.py` | 数值解 vs 解析解一致性测试 + 物理检验 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC005_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, validate_initial_state

x0, y0, vx0, vy0 = 1.0, 0.0, 0.0, 1.0
R, omega0, alpha, xc, yc = 1.0, 1.0, 0.0, 0.0, 0.0
t_eval = np.linspace(0, 2 * np.pi, 401)
initial_state = np.array([x0, y0, vx0, vy0], dtype=float)
validate_initial_state(initial_state, R, omega0, xc, yc)
sol = solve_ivp(dynamics, (0, 2 * np.pi), initial_state,
                t_eval=t_eval, args=(R, omega0, alpha, xc, yc))

x, y = sol.y[0], sol.y[1]

plt.figure(figsize=(6, 6))
plt.plot(x, y, label="trajectory")
plt.plot(xc, yc, "ro", label="center")
plt.plot(x0, y0, "go", label="start")
plt.title("MEC-005: Non-Uniform Circular Motion")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
