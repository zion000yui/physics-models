# MEC-003 Projectile Motion（抛体运动）

抛体运动：质点仅在重力作用下在二维平面内运动。
这是 MEC-002 受恒力质点在二维空间和重力方向上的自然推广，
用来验证「模型定义 → 引擎求解 → 解析解对照」闭环在二维运动下仍然成立。

## 物理背景

抛体运动是最经典的力学问题之一。当忽略空气阻力，且重力加速度 `g`
视为常数时，质点在水平方向不受力、在竖直方向受恒定的重力加速度作用。

其轨迹为抛物线；水平方向速度守恒，竖直方向做匀变速直线运动。

## 数学模型

- 状态：位置 `(x, y)`、速度 `(vx, vy)`
- 参数：重力加速度 `g`

一阶常微分方程：

```
dx/dt = vx
dy/dt = vy
dvx/dt = 0
dvy/dt = -g
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
0 \\
-g
\end{bmatrix}
$$

## 解析解

$$
\begin{aligned}
x(t) &= x_0 + v_{x0}\, t \\
y(t) &= y_0 + v_{y0}\, t - \tfrac{1}{2} g\, t^2 \\
v_x(t) &= v_{x0} \\
v_y(t) &= v_{y0} - g\, t
\end{aligned}
$$

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `x0` | 初始水平位置 (m) | `0.0` |
| `y0` | 初始垂直位置 (m) | `10.0` |
| `vx0` | 初始水平速度 (m/s) | `10.0` |
| `vy0` | 初始垂直速度 (m/s) | `15.0` |
| `g` | 重力加速度 (m/s²) | `9.81` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：动力学方程 `dynamics` + 解析解 `analytical` |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC003_consistency.py` | 数值解 vs 解析解一致性测试 + 物理检验 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC003_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics

x0, y0, vx0, vy0, g = 0.0, 10.0, 10.0, 15.0, 9.81
t_eval = np.linspace(0, 5, 401)
sol = solve_ivp(dynamics, (0, 5), [x0, y0, vx0, vy0],
                t_eval=t_eval, args=(g,))

x, y = sol.y[0], sol.y[1]

plt.figure(figsize=(6, 4))
plt.plot(x, y)
plt.title("MEC-003: Projectile Motion Trajectory")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
