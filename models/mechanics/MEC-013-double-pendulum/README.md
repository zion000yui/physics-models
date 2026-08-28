# MEC-013 Double Pendulum（双摆）

双摆：两个单摆串联连接，具有强非线性耦合和混沌行为。
是经典力学中混沌动力学的标准范例，也是 010 号段振动系统的最终模型。

## 物理背景

第一个摆悬挂于固定支点，第二个摆悬挂于第一个摆的末端。双摆系统只有两个
自由度 (θ₁, θ₂)，但表现出丰富的动力学行为：

- **小角度**：线性化为耦合振子（MEC-014 类型），运动为准周期
- **大角度**：强非线性，表现出混沌行为——对初始条件极度敏感
- **能量守恒**：无阻尼时总机械能守恒，但运动不可预测

**假设**：
- 刚性无质量杆，长度 `L1 > 0`、`L2 > 0`
- 质点质量 `m1 > 0`、`m2 > 0`
- 重力加速度 `g > 0`
- 无阻尼、无外力
- 平衡位置在 θ₁=0, θ₂=0（竖直向下）

## 与 MEC-010~015 的关系

| 特性 | MEC-010 | MEC-014 | MEC-015 | MEC-013 |
|------|---------|---------|---------|---------|
| 自由度 | 1 | 2 | 1 | 2 |
| 线性/非线性 | 线性 | 线性 | 非线性 | 非线性 |
| 混沌 | 否 | 否 | 否 | 是 |
| 解析解 | 闭式 | 简正模态 | 椭圆积分周期 | 无（数值积分） |
| 小角度退化 | — | — | MEC-010 | MEC-014 |

小角度极限：双摆线性化为两个耦合的线性振子，与 MEC-014 耦合振子形式一致。

## 数学模型

- 状态：角度 `(theta1, theta2)`、角速度 `(omega1, omega2)`
- 参数：`m1, m2, L1, L2, g`

状态空间形式：

```
dθ1/dt = ω1
dθ2/dt = ω2
dω1/dt = α1（由耦合方程组求解）
dω2/dt = α2（由耦合方程组求解）
```

## 状态空间表示

```
state = [theta1, theta2, omega1, omega2]
```

- `theta1, theta2` —— 上摆、下摆角度（rad）
- `omega1, omega2` —— 上摆、下摆角速度（rad/s）

## 微分方程推导

双摆的拉格朗日方程为：

$$(m_1+m_2)L_1\ddot{\theta}_1 + m_2 L_2 \ddot{\theta}_2 \cos\Delta + m_2 L_2 \dot{\theta}_2^2 \sin\Delta + (m_1+m_2)g\sin\theta_1 = 0$$

$$m_2 L_2 \ddot{\theta}_2 + m_2 L_1 \ddot{\theta}_1 \cos\Delta - m_2 L_1 \dot{\theta}_1^2 \sin\Delta + m_2 g\sin\theta_2 = 0$$

其中 Δ = θ₁ - θ₂。通过克莱默法则求解 2×2 线性方程组得到 α₁, α₂ 的显式表达式。

## 守恒量

**总机械能**：

$$
E = \frac{1}{2}(m_1+m_2)L_1^2\dot{\theta}_1^2 + \frac{1}{2}m_2 L_2^2\dot{\theta}_2^2 + m_2 L_1 L_2 \dot{\theta}_1\dot{\theta}_2\cos\Delta + (m_1+m_2)gL_1(1-\cos\theta_1) + m_2 g L_2(1-\cos\theta_2)
$$

无阻尼时机械能守恒。

## 解析解

**无闭式时间解析解**。双摆是非线性混沌系统，一般运动不存在初等函数解析解。

**小角度线性化**：退化为线性耦合振子（MEC-014 类型），可通过简正模态分析求解。

**验证方法**：由于无解析解，交叉验证通过以下方式完成：
1. 机械能守恒（物理定律）
2. 小角度行为与线性理论对照
3. 对初始条件敏感性（混沌特征）
4. dynamics 方程的手动验证

## 混沌特征

- **对初始条件敏感**：初始角度差 1e-4 rad 的两条轨迹在有限时间后显著发散
- **Lyapunov 指数为正**： Nearby trajectories diverge exponentially
- **不可预测性**：长期行为无法通过解析方法预测
- **Poincaré 截面**：显示混沌区域与规则区域共存

## 相空间

4 维相空间 (θ₁, θ₂, ω₁, ω₂)。可视化的常用方式：
- **二维投影**：(θ₁, ω₁) 或 (θ₂, ω₂)
- **Poincaré 截面**：在 θ₂=0 时记录 (θ₁, ω₁)，揭示混沌结构
- **轨迹图**：(x₂, y₂) 平面（下摆末端笛卡尔坐标），展示混沌轨迹

## 初始状态约束

任意 `(theta1_0, theta2_0, omega1_0, omega2_0)` 都是合法初始状态。

仅要求：`m1 > 0`、`m2 > 0`、`L1 > 0`、`L2 > 0`、`g > 0`

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `m1` | 上摆质量 (kg) | `1.0` |
| `m2` | 下摆质量 (kg) | `1.0` |
| `L1` | 上摆摆长 (m) | `1.0` |
| `L2` | 下摆摆长 (m) | `1.0` |
| `g` | 重力加速度 (m/s²) | `9.81` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：非线性动力学方程 `dynamics` + 机械能工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分（rtol=1e-10）并打印能量统计 |
| `test_MEC013_consistency.py` | 能量守恒 + 小角度行为 + 混沌敏感性 + 平衡点 + dynamics 手动验证 + 非法参数 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC013_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics

m1, m2, L1, L2, g = 1.0, 1.0, 1.0, 1.0, 9.81
theta1_0, theta2_0 = np.pi / 2, 0.0
t_eval = np.linspace(0, 10, 2000)
sol = solve_ivp(dynamics, (0, 10), [theta1_0, theta2_0, 0, 0],
                t_eval=t_eval, args=(m1, m2, L1, L2, g),
                rtol=1e-10, atol=1e-12)

# 下摆末端笛卡尔坐标
x2 = L1 * np.sin(sol.y[0]) + L2 * np.sin(sol.y[1])
y2 = -L1 * np.cos(sol.y[0]) - L2 * np.cos(sol.y[1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 轨迹图
ax1.plot(x2, y2, linewidth=0.5)
ax1.set_title("MEC-013: Double Pendulum (tip trajectory)")
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.set_aspect("equal"); ax1.grid(True)

# 相空间投影 (θ1, ω1)
ax2.plot(sol.y[0], sol.y[2], linewidth=0.5)
ax2.set_title("MEC-013: Phase Space (θ1, ω1)")
ax2.set_xlabel("θ1 (rad)"); ax2.set_ylabel("ω1 (rad/s)")
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
