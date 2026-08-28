# MEC-015 Nonlinear Pendulum（非线性单摆）

非线性单摆：质量 m 在长度 L 的刚性杆末端，在重力 g 作用下做有限振幅摆动。
完整保留 sin(θ) 非线性，不使用小角度近似作为主模型。是从线性振动系统
（MEC-010~014）进入非线性动力学的关键过渡模型。

## 物理背景

标准单摆运动方程 θ̈ + (g/L)·sin(θ) = 0 是最简单的非线性振子。
小角度近似 sin(θ) ≈ θ 将其简化为线性简谐振子（MEC-010），但有限振幅下
非线性效应导致：周期增长、相图偏离椭圆、出现分离轨道和旋转运动。

**假设**：
- 刚性无质量杆，长度 `L > 0`
- 质点质量 `m > 0`（不参与运动方程，仅出现在机械能中）
- 重力加速度 `g > 0`，方向竖直向下
- 无阻尼、无外力
- 平衡位置在 θ=0（最低点）

## 与 MEC-010 的关系

| 特性 | MEC-010（简谐振子） | MEC-015（非线性单摆） |
|------|---------------------|----------------------|
| 方程 | θ̈ + ω₀²θ = 0 | θ̈ + (g/L)·sin(θ) = 0 |
| 线性/非线性 | 线性 | 非线性 |
| 周期 | 恒定 T₀ = 2π/ω₀ | 随振幅增长，T > T₀ |
| 解析解 | 闭式 cos/sin | 仅小角度有闭式；一般用椭圆积分 |
| 相图 | 精确椭圆 | 小振幅接近椭圆，大振幅偏离 |
| 有效 k | k | k_eff = mg/L（线性化后） |

小角度极限：sin(θ)→θ，ω₀ = √(g/L) = √(k_eff/m)，退化为 MEC-010。

## 数学模型

- 状态：角度 `theta`、角速度 `omega`
- 参数：重力 `g`、摆长 `L`、质量 `m`

非线性运动方程：

```
dθ/dt = ω
dω/dt = -(g/L)·sin(θ)
```

## 状态空间表示

```
state = [theta, omega]
```

- `theta` —— 摆角（rad，相对于平衡位置）
- `omega` —— 角速度（rad/s）

## 微分方程推导

重力恢复力矩：`τ = -mgL·sin(θ)`

转动定律：`I·θ̈ = τ`，其中 `I = mL²`

$$
\ddot{\theta} + \frac{g}{L} \sin\theta = 0
$$

小角度线性近似（sin θ ≈ θ）：

$$
\ddot{\theta} + \omega_0^2 \theta = 0, \quad \omega_0 = \sqrt{\frac{g}{L}}
$$

## 解析解

**1. 小角度线性近似**（|θ| ≪ 1）：

$$
\theta(t) = \theta_0 \cos(\omega_0 t) + \frac{\omega_0^{\text{init}}}{\omega_0} \sin(\omega_0 t)
$$

周期 `T₀ = 2π·√(L/g)`，与振幅无关。

**2. 有限振幅周期**（椭圆积分理论结果）：

$$
T = 4\sqrt{\frac{L}{g}} \, K(k), \quad k = \sin\frac{\theta_{\max}}{2}
$$

其中 `K(k)` 为第一类完全椭圆积分。振幅增大时周期增长：
- θ_max = 0: T = T₀（小角度极限）
- θ_max → π: T → ∞（分离轨道）

**3. 一般有限振幅运动**：无初等函数时间解析解，通过数值积分获得。

## 非线性与线性近似的差异

| 振幅 | T/T₀ | 偏差 |
|------|------|------|
| 0.1 rad (5.7°) | 1.0006 | 0.06% |
| 0.5 rad (28.6°) | 1.0159 | 1.6% |
| 1.0 rad (57.3°) | 1.0663 | 6.6% |
| 1.5 rad (85.9°) | 1.1620 | 16.2% |
| 2.0 rad (114.6°) | 1.3289 | 32.9% |
| 3.0 rad (171.9°) | 2.5712 | 157% |

## 守恒量

**机械能**：

$$
E = \frac{1}{2} m L^2 \omega^2 + mgL(1 - \cos\theta)
$$

无阻尼时机械能守恒。

## 相空间（θ - ω 平面）

- **小振幅**（E ≪ 2mgL）：轨迹接近椭圆，接近线性简谐振子
- **有限振幅**（E < 2mgL）：轨迹偏离椭圆，周期变长
- **分离轨道**（E = 2mgL）：连接不稳定平衡点 (±π, 0) 的分界线
- **摆动**（E < 2mgL）：θ 在 (-π, π) 间周期振荡
- **旋转**（E > 2mgL）：θ 单调递增/递减（连续旋转）

## 初始状态约束

任意 `(theta0, omega0)` 都是合法初始状态。

仅要求：`g > 0`、`L > 0`、`m > 0`

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `g` | 重力加速度 (m/s²) | `9.81` |
| `L` | 摆长 (m) | `1.0` |
| `m` | 质量 (kg) | `1.0` |
| `theta0` | 初始角度 (rad) | `1.0` |
| `omega0` | 初始角速度 (rad/s) | `0.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：非线性动力学方程 `dynamics` + 小角度解析解 `analytical` + 非线性周期（椭圆积分）`nonlinear_period` + 机械能/固有频率工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC015_consistency.py` | 小角度一致性 + 退化 + 周期递增 + 椭圆积分周期 + 能量守恒 + 平衡点 + 分离轨道 + 非法参数 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC015_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, small_angle_period, nonlinear_period

g, L, m = 9.81, 1.0, 1.0

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 时域：不同振幅
for theta0 in [0.1, 1.0, 2.0]:
    T = nonlinear_period(g, L, theta0)
    t_eval = np.linspace(0, T, 401)
    sol = solve_ivp(dynamics, (0, T), [theta0, 0],
                    t_eval=t_eval, args=(g, L, m), rtol=1e-9, atol=1e-12)
    axes[0].plot(sol.t, sol.y[0], label=f"θ₀={theta0} rad")
axes[0].set_title("MEC-015: Nonlinear Pendulum (time domain)")
axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("θ (rad)")
axes[0].legend(); axes[0].grid(True)

# 相空间：不同能量
for theta0 in [0.2, 1.0, 2.0, np.pi - 0.01]:
    T = nonlinear_period(g, L, theta0) if theta0 < np.pi - 0.1 else 10
    t_eval = np.linspace(0, T, 1001)
    sol = solve_ivp(dynamics, (0, T), [theta0, 0],
                    t_eval=t_eval, args=(g, L, m), rtol=1e-9, atol=1e-12)
    axes[1].plot(sol.y[0], sol.y[1], label=f"θ₀={theta0:.2f}")
# 旋转运动
sol_rot = solve_ivp(dynamics, (0, 5), [0, 8],
                    t_eval=np.linspace(0, 5, 1001), args=(g, L, m),
                    rtol=1e-9, atol=1e-12)
axes[1].plot(sol_rot.y[0], sol_rot.y[1], "--", label="rotation")
axes[1].set_title("MEC-015: Phase Portrait (θ - ω)")
axes[1].set_xlabel("θ (rad)"); axes[1].set_ylabel("ω (rad/s)")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
