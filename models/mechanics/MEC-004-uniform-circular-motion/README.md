# MEC-004 Uniform Circular Motion（匀速圆周运动）

匀速圆周运动：质点在二维平面内以恒定速率沿圆周运动，
半径 R 和角速度 ω 均保持不变。这是 MEC-003 抛体运动中恒力方向
持续转向的自然结果，也是第一阶段中第一次引入曲线运动。

## 物理背景

匀速圆周运动中，质点速率恒定但速度方向持续变化，因此存在指向
圆心的向心加速度。向心加速度由向心力提供，大小为 `a_c = Rω² = v²/R`。

**假设**：
- 无切向力（无空气阻力、无驱动力矩）
- 半径 R 恒定
- 角速度 ω 恒定
- 质量不影响运动学（动力学中向心力 F_c = mRω²，但运动方程与 m 无关）

## 数学模型

- 状态：位置 `(x, y)`、速度 `(vx, vy)`
- 参数：半径 `R`、角速度 `omega`、圆心 `(xc, yc)`

一阶常微分方程：

```
dx/dt = vx
dy/dt = vy
dvx/dt = -ω² (x - xc)
dvy/dt = -ω² (y - yc)
```

## 状态空间表示

```
state = [x, y, vx, vy]
```

其中：

- `x`  —— 水平位置（m）
- `y`  —— 垂直位置（m）
- `vx` —— 水平速度（m/s）
- `vy` —— 垂直速度（m/s）

## 微分方程

$$
\frac{d}{dt}\begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}
=
\begin{bmatrix}
v_x \\
v_y \\
-\omega^2 (x - x_c) \\
-\omega^2 (y - y_c)
\end{bmatrix}
$$

## 初始状态约束

匀速圆周运动要求初始状态满足：

$$
\begin{aligned}
(x_0 - x_c)^2 + (y_0 - y_c)^2 &= R^2 \\
v_{x0}(x_0 - x_c) + v_{y0}(y_0 - y_c) &= 0 \\
|\\mathbf{v}_0| &= R|\\omega|
\end{aligned}
$$

模型入口 `validate_initial_state()` 会在求解前验证上述条件，
不满足时抛出 `AssertionError`。

## 解析解

$$
\begin{aligned}
\\theta(t) &= \\theta_0 + \\omega t \\\\
x(t) &= x_c + R \\cos\\theta(t) \\\\
y(t) &= y_c + R \\sin\\theta(t) \\\\
v_x(t) &= -R \\omega \\sin\\theta(t) \\\\
v_y(t) &= R \\omega \\cos\\theta(t)
\end{aligned}
$$

其中 `θ₀ = arctan2(y₀ - yc, x₀ - xc)`。

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `R` | 圆周半径 (m) | `1.0` |
| `omega` | 角速度 (rad/s，可正可负) | `1.0` |
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
| `test_MEC004_consistency.py` | 数值解 vs 解析解一致性测试 + 物理检验 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC004_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, validate_initial_state

x0, y0, vx0, vy0 = 1.0, 0.0, 0.0, 1.0
R, omega, xc, yc = 1.0, 1.0, 0.0, 0.0
t_eval = np.linspace(0, 2 * np.pi, 401)
initial_state = np.array([x0, y0, vx0, vy0], dtype=float)
validate_initial_state(initial_state, R, omega, xc, yc)
sol = solve_ivp(dynamics, (0, 2 * np.pi), initial_state,
                t_eval=t_eval, args=(R, omega, xc, yc))

x, y = sol.y[0], sol.y[1]

plt.figure(figsize=(6, 6))
plt.plot(x, y, label="trajectory")
plt.plot(xc, yc, "ro", label="center")
plt.plot(x0, y0, "go", label="start")
plt.title("MEC-004: Uniform Circular Motion")
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

## 数值验证

以 `R=1, ω=1, θ₀=0` 为例，`scipy_solve.py` 输出：

```
时间点数      : 101
终止时间      : 6.283185 s
末点数值 [x,y]: [1.000000, 0.000000]
末点解析 [x,y]: [1.000000, 0.000000]
最大 x 误差   : 1.421e-14
最大 y 误差   : 4.974e-14
速率均值      : 1.000000 m/s，标准差 0.000e+00
向心加速度均值: 1.000000 m/s²，理论值 1.000000
轨道半径均值  : 1.000000 m，标准差 3.553e-15
```

验证结果：

`OK: MEC-004 数值解与解析解一致`
