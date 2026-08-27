# MEC-006 Central Force Hooke（胡克型中心力场运动）

胡克型中心力场：质点在指向原点的线性 restoring force F = -k·r 作用下运动。
这是系列第一次引入真实的力，质量 m 成为显式参数，需要验证角动量守恒和机械能守恒。

## 物理背景

在胡克型中心力场中，质点受到的力与位置成正比、方向指向原点：
`F = -k·r`。这等价于各向同性的简谐振子，轨迹通常为椭圆，特殊情况下退化为圆。

**假设**：
- 力心固定在原点
- 胡克型恢复力 `F = -k·r`（k > 0）
- 质量 m > 0
- 无阻尼、无其他外力
- 运动限于二维平面

## 与 MEC-004/005 的关键区别

| 特性 | MEC-004/005 | MEC-006 |
|------|-------------|---------|
| 驱动方式 | 运动学约束（ω 给定） | 动力学（力 F=-kr） |
| 半径 | 恒定 | 可变 |
| 质量 m | 无关 | 显式参数 |
| 守恒量 | 无 | 角动量 + 机械能守恒 |
| 初始状态 | 必须在圆上 | 任意 (x0,y0,vx0,vy0) |

## 数学模型

- 状态：位置 `(x, y)`、速度 `(vx, vy)`
- 参数：弹性系数 `k`、质量 `m`

一阶常微分方程：

```
dx/dt = vx
dy/dt = vy
dvx/dt = -(k/m)·x
dvy/dt = -(k/m)·y
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

## 微分方程推导

胡克型向心力：`F = -k·r = -k·(x, y)`

牛顿第二定律：`m·a = F`

因此：

$$
\begin{bmatrix} \dot{v}_x \\ \dot{v}_y \end{bmatrix}
= -\frac{k}{m} \begin{bmatrix} x \\ y \end{bmatrix}
$$

令 `ω₀ = √(k/m)`，则 x、y 方向各自独立满足简谐振动方程：

$$
\frac{d^2 x}{dt^2} + \omega_0^2 x = 0, \quad
\frac{d^2 y}{dt^2} + \omega_0^2 y = 0
$$

## 解析解

$$
\begin{aligned}
x(t) &= x_0 \cos(\omega_0 t) + \frac{v_{x0}}{\omega_0} \sin(\omega_0 t) \\
y(t) &= y_0 \cos(\omega_0 t) + \frac{v_{y0}}{\omega_0} \sin(\omega_0 t) \\
v_x(t) &= -x_0 \omega_0 \sin(\omega_0 t) + v_{x0} \cos(\omega_0 t) \\
v_y(t) &= -y_0 \omega_0 \sin(\omega_0 t) + v_{y0} \cos(\omega_0 t)
\end{aligned}
$$

其中 `ω₀ = √(k/m)`。

## 守恒量

**角动量（绕原点）**：

$$L = m \cdot (x \cdot v_y - y \cdot v_x)$$

**机械能**：

$$E = \frac{1}{2} m (v_x^2 + v_y^2) + \frac{1}{2} k (x^2 + y^2)$$

## 初始状态约束

任意 `(x0, y0, vx0, vy0)` 都是合法初始状态。

仅要求：
- `k > 0`
- `m > 0`

**退化情形**：当初始条件恰好满足 `|r0|` 恒定、`v0⊥r0`、`|v0| = ω₀·|r0|` 时，轨迹退化为圆形。

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `k` | 弹性系数 (N/m) | `1.0` |
| `m` | 质量 (kg) | `1.0` |
| `x0` | 初始水平位置 (m) | `1.0` |
| `y0` | 初始垂直位置 (m) | `0.0` |
| `vx0` | 初始水平速度 (m/s) | `0.0` |
| `vy0` | 初始垂直速度 (m/s) | `1.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：动力学方程 `dynamics` + 解析解 `analytical` + 角动量/能量工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC006_consistency.py` | 数值解 vs 解析解一致性测试 + 物理检验 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC006_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics

x0, y0, vx0, vy0, k, m = 1.0, 0.0, 0.0, 1.0, 1.0, 1.0
omega0 = np.sqrt(k / m)
t_eval = np.linspace(0, 2 * np.pi / omega0, 401)
sol = solve_ivp(dynamics, (0, 2 * np.pi / omega0), [x0, y0, vx0, vy0],
                t_eval=t_eval, args=(k, m))

x, y = sol.y[0], sol.y[1]

plt.figure(figsize=(6, 6))
plt.plot(x, y)
plt.plot(0, 0, "ro", label="center")
plt.plot(x0, y0, "go", label="start")
plt.title("MEC-006: Central Force Hooke")
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
