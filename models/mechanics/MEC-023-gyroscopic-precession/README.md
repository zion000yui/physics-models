# MEC-023 Gyroscopic Precession（陀螺慢进动近似）

高速自旋对称陀螺的慢进动近似模型（slow precession approximation for a
fast-spinning symmetric top）。陀螺绕自身对称轴高速自旋，在重力力矩作用下
做慢进动和微小章动。

**本模型不是完整 3D Euler top / Euler equations。** 完整模型留到后续独立模型。

## 物理背景

对称陀螺（尖端固定）绕自身对称轴高速自旋时，重力力矩不使陀螺倒下，
而是使自旋轴绕竖直方向做缓慢进动（precession）。这是角动量守恒和
力矩引起角动量方向改变的经典演示。

**假设**：
1. **对称陀螺**：I₁ = I₂ ≠ I₃
2. **高速自旋**：ω_s >> |θ̇|, |φ̇|（自旋远快于进动和章动）
3. **慢进动近似**：稳态分析中忽略 Ω_p² 项（Ω_p << ω_s）
4. **定点运动**：尖端固定，只有转动
5. **自旋角速度视为常数**（p_ψ = I₃ω_s 守恒）

## 与 MEC-021/015 的关系

| 关系 | 说明 | 严格性 |
|------|------|--------|
| ω_s=0 时方程 | Routhian 退化为复摆 L_eff（2DOF），方程仍成立 | ✓ 但近似失效 |
| ω_s=0 + φ̇=0 + I₁=mL² | 方程退化为 MEC-015 单摆形式 | ✓ 需额外约束 |
| ω_s=0 时稳态公式 | Ω_p = mgl/(I₃ω_s) → ∞，不适用 | ✗ 近似失效 |

**注意**：ω_s=0 时不能简单宣称"模型退化为 MEC-015"。
- Routhian 方程本身仍成立（退化为 2DOF 复摆）
- 但高速自旋假设和慢进动近似全部失效
- 稳态公式 Ω_p = mgl/(I₃ω_s) 发散
- 需额外约束 φ̇=0 且 I₁=mL² 才与 MEC-015 一致

## 坐标系：欧拉角

- **θ**（倾斜角）：自旋轴与竖直向上方向的夹角。θ=0 直立，θ=π 倒挂
- **φ**（进动角）：自旋轴绕竖直轴的方位角
- **ψ**（自旋角）：陀螺绕自身对称轴的转角（循环坐标，被消去）

**三种角速度的区分**：
- **自旋角速度** ω_s = ψ̇ + φ̇cos θ：陀螺绕对称轴的总角速度（视为常数）
- **进动角速度** Ω_p = φ̇：自旋轴绕竖直轴的旋转速度
- **章动角速度** θ̇：倾斜角的变化率

## Routhian 降维

完整拉格朗日量（3DOF）消去循环坐标 ψ 后得到有效拉格朗日量：

$$
L_{\text{eff}} = \frac{1}{2}I_1(\dot{\theta}^2 + \dot{\varphi}^2\sin^2\theta) + I_3\omega_s\dot{\varphi}\cos\theta - mgl\cos\theta
$$

- **被消去的自由度**：ψ（自旋角）
- **由此产生的守恒量**：p_ψ = I₃ω_s（轴向角动量）
- **另一个循环坐标**：φ → p_φ = I₁φ̇sin²θ + I₃ω_s cos θ = const（竖直角动量）

## 状态空间表示

```
state = [theta, theta_dot, phi, phi_dot]
```

- `theta` —— 倾斜角（rad）
- `theta_dot` —— 章动角速度（rad/s）
- `phi` —— 进动角（rad）
- `phi_dot` —— 进动角速度（rad/s）

## 核心方程

**精确方程**（dynamics 实现，非近似）：

$$
I_1\ddot{\theta} = I_1\dot{\varphi}^2\sin\theta\cos\theta - I_3\omega_s\dot{\varphi}\sin\theta + mgl\sin\theta
$$

$$
\frac{d}{dt}(I_1\dot{\varphi}\sin^2\theta + I_3\omega_s\cos\theta) = 0
$$

**量纲检查**：所有项量纲均为 [kg·m²/s²] = [N·m]（力矩）✓

## 稳态进动解

稳态条件 θ=θ₀（常数）、φ̇=Ω_p（常数），代入 θ 方程：

$$
\sin\theta_0 (I_1\cos\theta_0 \cdot \Omega_p^2 - I_3\omega_s\Omega_p + mgl) = 0
$$

**sin θ₀ ≠ 0 时**（消去 sin θ₀）：

$$
I_1\cos\theta_0 \cdot \Omega_p^2 - I_3\omega_s\Omega_p + mgl = 0
$$

**慢进动近似**（忽略 Ω_p² 项）：

$$
\Omega_p \approx \frac{mgl}{I_3\omega_s}
$$

**sin θ₀ 因子的说明**：
重力力矩 |τ| = mgl·sin θ₀，角动量变化率 |dL/dt| = Ω_p·I₃ω_s·sin θ₀。
sin θ₀ 在分子和分母中同时出现并消去，因此 Ω_p 不含 sin θ₀。
但此消去仅在 sin θ₀ ≠ 0 时合法。

**sin θ₀ = 0 时**（θ₀=0 或 π）：原方程恒等于 0=0，任意 Ω_p 都是稳态
（重力力矩为零，无重力驱动进动）。

## 守恒量

| 守恒量 | 公式 | 来源 |
|--------|------|------|
| p_ψ（轴向角动量） | I₃ω_s = const | ψ 循环坐标 |
| p_φ（竖直角动量） | I₁φ̇sin²θ + I₃ω_s cos θ = const | φ 循环坐标 |
| E_eff（有效能量） | ½I₁(θ̇²+φ̇²sin²θ) + mgl cos θ = const | L_eff 不含 t |

注意：E_eff 不是总机械能。总机械能 = E_eff + ½I₃ω_s²。

## 极限分析

| 极限 | 物理含义 |
|------|---------|
| ω_s → ∞ | 进动极慢（Ω_p→0），章动振幅→0，陀螺几乎不倒 |
| ω_s = 0 | 高速自旋假设和慢进动近似全部失效；Routhian 方程退化为复摆但稳态公式发散 |

## 适用范围和局限性

**适用**：对称陀螺、高速自旋、定点、稳态或近稳态进动

**不适用**：非对称陀螺、低速/零自旋、自由陀螺、大幅度章动

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `m` | 陀螺质量 (kg) | `1.0` |
| `l` | 支点到质心 (m) | `0.5` |
| `I1` | 横向转动惯量 (kg·m²) | `0.2` |
| `I3` | 轴向转动惯量 (kg·m²) | `0.1` |
| `omega_s` | 自旋角速度 (rad/s) | `50.0` |
| `g` | 重力加速度 (m/s²) | `9.81` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 精确 Routhian 方程 `dynamics` + 稳态解析解 `analytical`（近似）+ 精确/近似稳态进动 + 守恒量工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 数值积分并打印守恒量统计 |
| `test_MEC023_consistency.py` | 精确稳态 θ 恒定 + p_φ 守恒 + E_eff 守恒 + 稳态方程 + 1/ω_s 关系 + 近似改善 + sin θ₀ + 非法参数 |
| `README.md` | 物理定义、公式、适用范围、局限性 |

## 运行

```bash
python scipy_solve.py
python test_MEC023_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, steady_state_precession, exact_steady_state_precession

m, l, I1, I3, omega_s, g = 1.0, 0.5, 0.2, 0.1, 50.0, 9.81
theta_0 = np.pi / 4
omega_slow, _ = exact_steady_state_precession(m, l, I1, I3, omega_s, g, theta_0)

t_eval = np.linspace(0, 10, 500)
sol = solve_ivp(dynamics, (0, 10), [theta_0, 0, 0, omega_slow],
                t_eval=t_eval, args=(m, l, I1, I3, omega_s, g),
                rtol=1e-10, atol=1e-12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# θ(t) — 应保持恒定（精确稳态）
ax1.plot(sol.t, np.degrees(sol.y[0]))
ax1.set_title("MEC-023: Tilt angle θ(t) (exact steady state)")
ax1.set_xlabel("t (s)"); ax1.set_ylabel("θ (°)")
ax1.grid(True)

# φ(t) — 线性增长（匀速进动）
ax2.plot(sol.t, sol.y[2])
ax2.set_title("MEC-023: Precession angle φ(t)")
ax2.set_xlabel("t (s)"); ax2.set_ylabel("φ (rad)")
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
