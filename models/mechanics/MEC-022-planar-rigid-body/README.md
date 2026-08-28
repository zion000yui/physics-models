# MEC-022 Planar Rigid Body（平面刚体运动）

平面刚体运动：刚体在 2D 平面内做自由运动，具有 3 个自由度——
2D 平动（质心运动）+ 1D 绕质心轴（z 轴）转动。
当外力不通过质心时，力矩 τ = r × F 将平动与转动耦合。
这是 2D 框架下的最大自由度刚体模型，**不是 3D/6DOF 自由刚体**。

## 物理背景

MEC-022 的核心新增概念不是简单地同时有"力"和"力矩"两个独立参数，
而是：同一个外力 F 作用于偏离质心的位置 r 时，同时产生
- 质心加速度 a_cm = F/m（平动效应）
- 角加速度 α = (r×F)/I（转动效应）

力矩不是独立的黑盒参数，而是由力臂 r 和力 F 共同决定的。

**假设**：
- 刚体在 2D 平面内运动（3 自由度）
- 质量 `m > 0`，转动惯量 `I > 0`（绕质心轴）
- 外力 `(Fx, Fy)` 作用于相对于质心的位置 `(rx, ry)`
- 无阻尼
- 完整 3D/6DOF 欧拉方程和惯性张量不在本模型范围

## 与 MEC-020/021 的关系

| 条件 | 退化为 | 物理依据 | 严格性 |
|------|--------|---------|--------|
| 力通过质心（rx=ry=0） | MEC-020（纯平动） | τ=r×F=0→无旋转 | ✓ 严格 |
| 无外力+质心初速为零 | MEC-021（纯转动） | F=0→质心不动，初始ω≠0→绕质心转 | ✓ 严格 |
| 无外力无初角速度 | MEC-020（匀速平动） | F=0→v=const, ω=0 | ✓ 严格 |

## 数学模型

- 状态：质心位置 `(x, y)`、质心速度 `(vx, vy)`、转角 `theta`、角速度 `omega`
- 参数：质量 `m`、转动惯量 `I`、外力 `Fx, Fy`、力臂 `rx, ry`

一阶常微分方程：

```
dx/dt = vx
dy/dt = vy
dtheta/dt = omega
dvx/dt = Fx / m
dvy/dt = Fy / m
domega/dt = (rx·Fy - ry·Fx) / I
```

## 状态空间表示

```
state = [x_cm, y_cm, vx_cm, vy_cm, theta, omega]
```

- `x_cm, y_cm` —— 质心位置（m）
- `vx_cm, vy_cm` —— 质心速度（m/s）
- `theta` —— 绕质心转角（rad）
- `omega` —— 角速度（rad/s）

## 微分方程推导

**平动**（质心运动定理）：

$$
m \mathbf{a}_{\text{cm}} = \mathbf{F}_{\text{ext}}
$$

**转动**（绕质心轴）：

$$
I \alpha = \tau_{\text{cm}}
$$

**力矩的来源**（2D 叉积 z 分量）：

$$
\tau = r_x F_y - r_y F_x = (\mathbf{r} \times \mathbf{F})_z
$$

## 解析解

恒力 + 恒力臂（即恒力和恒力矩）：

$$
\begin{aligned}
x(t) &= x_0 + v_{x0} t + \frac{1}{2}\frac{F_x}{m} t^2 \\
y(t) &= y_0 + v_{y0} t + \frac{1}{2}\frac{F_y}{m} t^2 \\
\theta(t) &= \theta_0 + \omega_0 t + \frac{1}{2}\frac{\tau}{I} t^2
\end{aligned}
$$

其中 `τ = rx·Fy - ry·Fx`。

## 守恒量

无外力时：动量守恒 `P = m·v_cm = const`
无外力矩时：角动量守恒 `L = I·omega = const`
保守力时：机械能守恒

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `m` | 质量 (kg) | `1.0` |
| `I` | 转动惯量 (kg·m²) | `1.0` |
| `Fx, Fy` | 外力分量 (N) | `0.0, 4.0` |
| `rx, ry` | 力臂 (m) | `0.5, 0.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：3DOF 动力学 `dynamics` + 解析解 `analytical` + 力矩计算 `torque_from_force` + 动量/角动量/机械能工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC022_consistency.py` | 解析解一致性 + 力矩公式 + 两种退化 + 同力产生平转 + 动量/角动量守恒 + 机械能 + 非法参数 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC022_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, torque_from_force

# 偏心力：同一力同时驱动平动和转动
m, I = 1.0, 2.0
Fx, Fy, rx, ry = 0.0, 4.0, 0.5, 0.0
sol = solve_ivp(dynamics, (0, 3), [0, 0, 0, 0, 0, 0],
                t_eval=np.linspace(0, 3, 200),
                args=(m, I, Fx, Fy, rx, ry), rtol=1e-9, atol=1e-12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 质心轨迹
ax1.plot(sol.y[0], sol.y[1])
ax1.set_title("MEC-022: CM trajectory")
ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
ax1.grid(True); ax1.set_aspect("equal")

# 角度演化
ax2.plot(sol.t, sol.y[4], label="θ(t)")
ax2.plot(sol.t, sol.y[5], "--", label="ω(t)")
ax2.set_title("MEC-022: Rotation")
ax2.set_xlabel("t (s)")
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
