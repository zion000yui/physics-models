# MEC-020 Rigid Translation（刚体平动）

刚体平动：刚体在合外力作用下做纯平动（无旋转），所有质点具有相同速度和加速度。
根据质心运动定理，刚体平动等价于将全部质量集中于质心的质点运动。
是 020 号段（刚体动力学）的第一个模型。

## 物理背景

刚体平动是最简单的刚体运动形式。质心运动定理 `M·a_cm = F_ext` 表明，
刚体在合外力作用下的质心运动与一个质量为 M 的质点在相同外力下的运动完全等价。
旋转运动（MEC-021）需要额外的力矩和转动惯量描述，但在纯平动中不出现。

**假设**：
- 刚体不受力矩作用（无旋转，或初始角速度为零且无力矩）
- 合外力 `Fx, Fy` 可为任意实数（含零、恒力、重力等）
- 质量 `m > 0`
- 二维平面运动
- 无阻尼

## 与 MEC-001/002/003 的关系

| 条件 | 退化为 | 物理依据 |
|------|--------|---------|
| F = 0 | MEC-001（自由质点） | 质心匀速直线运动 |
| F = const | MEC-002（受力质点，2D） | 质心匀加速运动 |
| F = (0, -mg) | MEC-003（抛体运动） | 质心抛体轨迹 |

MEC-020 在数学上与 MEC-001/002/003 等价，但引入了"刚体"和"质心运动定理"
的概念框架，为后续 MEC-021（转动）和 MEC-022（平动+转动耦合）做铺垫。

## 数学模型

- 状态：质心位置 `(x_cm, y_cm)`、质心速度 `(vx_cm, vy_cm)`
- 参数：质量 `m`、合外力 `Fx, Fy`

一阶常微分方程：

```
dx_cm/dt = vx_cm
dy_cm/dt = vy_cm
dvx_cm/dt = Fx / m
dvy_cm/dt = Fy / m
```

## 状态空间表示

```
state = [x_cm, y_cm, vx_cm, vy_cm]
```

- `x_cm, y_cm` —— 质心位置（m）
- `vx_cm, vy_cm` —— 质心速度（m/s）

## 微分方程推导

质心运动定理：

$$
M \mathbf{a}_{\text{cm}} = \mathbf{F}_{\text{ext}}
$$

分量形式：

$$
\begin{aligned}
m \ddot{x}_{\text{cm}} &= F_x \\
m \ddot{y}_{\text{cm}} &= F_y
\end{aligned}
$$

## 解析解

恒力下：

$$
\begin{aligned}
x_{\text{cm}}(t) &= x_0 + v_{x0} t + \frac{1}{2}\frac{F_x}{m} t^2 \\
y_{\text{cm}}(t) &= y_0 + v_{y0} t + \frac{1}{2}\frac{F_y}{m} t^2 \\
v_{x,\text{cm}}(t) &= v_{x0} + \frac{F_x}{m} t \\
v_{y,\text{cm}}(t) &= v_{y0} + \frac{F_y}{m} t
\end{aligned}
$$

## 守恒量

无外力时（Fx=Fy=0），动量守恒：

$$
\mathbf{P} = m \mathbf{v}_{\text{cm}} = \text{const}
$$

## 初始状态约束

任意 `(x0, y0, vx0, vy0)` 都是合法初始状态。

仅要求：`m > 0`

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `m` | 刚体质量 (kg) | `1.0` |
| `Fx` | x 方向合外力 (N) | `0.0` |
| `Fy` | y 方向合外力 (N) | `0.0` |
| `x0` | 初始质心 x (m) | `0.0` |
| `y0` | 初始质心 y (m) | `0.0` |
| `vx0` | 初始质心 vx (m/s) | `10.0` |
| `vy0` | 初始质心 vy (m/s) | `15.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：质心运动定理 `dynamics` + 恒力解析解 `analytical` + 动量工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC020_consistency.py` | 恒力一致性 + 三种退化验证 + 动量守恒 + 非法参数 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC020_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics

# 重力场下平动（抛体运动）
m, g = 2.0, 9.81
sol = solve_ivp(dynamics, (0, 3), [0, 0, 10, 15],
                t_eval=np.linspace(0, 3, 100),
                args=(m, 0, -m*g), rtol=1e-9, atol=1e-12)

plt.figure(figsize=(8, 5))
plt.plot(sol.y[0], sol.y[1])
plt.title("MEC-020: Rigid Translation (projectile)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.axis("equal")
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
