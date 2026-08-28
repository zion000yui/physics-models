# MEC-012 Forced Oscillator（受迫振子）

受迫阻尼振子：质量 m 在弹性力 F = -k·x、线性阻尼力 F_d = -b·v 和
周期性外力 F_ext = F₀·cos(ωt) 共同作用下沿一维方向运动。
是 MEC-010 和 MEC-011 的进一步推广，引入瞬态/稳态分解和幅频响应。

## 物理背景

当振动系统受到周期性外力驱动时，其响应可分为两个阶段：
- **瞬态响应**（transient）：由初始条件决定，随时间衰减（同 MEC-011）
- **稳态响应**（steady-state）：由驱动力决定，以驱动频率持续振荡

稳态振幅随驱动频率变化的关系称为幅频响应。当驱动频率接近系统固有频率时
发生共振（resonance），振幅达到最大。

**假设**：
- 弹性恢复力 `F = -k·x`（k > 0）
- 线性阻尼力 `F_d = -b·v`（b ≥ 0）
- 周期性驱动力 `F_ext = F₀·cos(ωt)`（F0 ≥ 0，omega ≥ 0）
- 质量 `m > 0`
- 一维运动

## 与 MEC-010/011 的关系

| 特性 | MEC-010 | MEC-011 | MEC-012 |
|------|---------|---------|---------|
| 弹性力 | ✓ | ✓ | ✓ |
| 阻尼 | 无 | ✓ | ✓ |
| 驱动力 | 无 | 无 | ✓ |
| 机械能 | 守恒 | 耗散 | 注入+耗散 |
| 相图 | 闭合椭圆 | 内旋螺线 | 瞬态螺线 → 稳态极限环 |
| 解析解 | 闭式 | 闭式（3种） | 瞬态+稳态分解 |

当 F0=0 时退化为 MEC-011；当 F0=0 且 b=0 时退化为 MEC-010。

## 数学模型

- 状态：位移 `x`、速度 `v`
- 参数：`k`、`m`、`b`、`F0`、`omega`

一阶常微分方程：

```
dx/dt = v
dv/dt = -(k/m)·x - (b/m)·v + (F0/m)·cos(ωt)
```

## 状态空间表示

```
state = [x, v]
```

- `x` —— 位移（m）
- `v` —— 速度（m/s）

## 微分方程推导

合力 = 弹性力 + 阻尼力 + 驱动力：

$$
m\frac{d^2 x}{dt^2} = -kx - bv + F_0 \cos(\omega t)
$$

令 `ω₀ = √(k/m)`，`γ = b/(2m)`：

$$
\frac{d^2 x}{dt^2} + 2\gamma \frac{dx}{dt} + \omega_0^2 x = \frac{F_0}{m} \cos(\omega t)
$$

## 解析解：瞬态 + 稳态分解

**通解 = 瞬态解（齐次） + 稳态解（特解）**

**瞬态解**：与 MEC-011 完全相同（欠阻尼/临界/过阻尼三种），随时间衰减至零。

**稳态解**：

$$
x_{ss}(t) = A_{ss} \cos(\omega t - \delta)
$$

其中：

$$
A_{ss} = \frac{F_0/m}{\sqrt{(\omega_0^2 - \omega^2)^2 + (2\gamma\omega)^2}}, \quad
\tan\delta = \frac{2\gamma\omega}{\omega_0^2 - \omega^2}
$$

瞬态解的初始条件为 `x_tr(0) = x0 - x_ss(0)`，`v_tr(0) = v0 - v_ss(0)`。

## 幅频响应与共振

**共振频率**（最大响应频率）：

$$
\omega_{max} = \omega_0 \sqrt{1 - 2\zeta^2}
$$

仅在 `ζ < 1/√2` 时存在。当 `ζ ≥ 1/√2` 时，振幅随 ω 单调递减，无共振峰。

**无阻尼理想共振**：当 b=0 且 ω→ω₀ 时，稳态振幅发散（线性增长）。

## 能量关系

**稳态能量平衡**：驱动力注入功率 = 阻尼耗散功率

$$
\langle P_{in} \rangle = \langle F_0 \cos(\omega t) \cdot v \rangle = \langle P_{diss} \rangle = \langle b v^2 \rangle
$$

## 相图规范（延续 MEC-010/011）

- **横轴**：位移 x（m）
- **纵轴**：速度 v（m/s）
- **瞬态阶段**：与 MEC-011 相同的衰减轨迹
- **稳态阶段**：相图收敛为极限环（limit cycle）——椭圆
  - x 半轴 = A_ss，v 半轴 = A_ss·ω
- 无阻尼时无极限环

## 初始状态约束

任意 `(x0, v0)` 都是合法初始状态。

仅要求：`k > 0`、`m > 0`、`b ≥ 0`、`F0 ≥ 0`、`omega ≥ 0`

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `k` | 弹性系数 (N/m) | `1.0` |
| `m` | 质量 (kg) | `1.0` |
| `b` | 阻尼系数 (N·s/m) | `0.4` |
| `F0` | 驱动力幅值 (N) | `1.0` |
| `omega` | 驱动角频率 (rad/s) | `0.8` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：动力学方程 `dynamics` + 解析解 `analytical`（瞬态+稳态分解）+ 稳态振幅/相位/共振频率/阻尼比工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC012_consistency.py` | 数值解 vs 解析解 + 退化验证 + 稳态振幅 + 共振频率 + 能量平衡 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC012_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, damping_ratio, steady_state_amplitude

k, m, b = 1.0, 1.0, 0.4
F0, omega = 1.0, 0.8
x0, v0 = 0.0, 0.0
t_eval = np.linspace(0, 30, 1001)
sol = solve_ivp(dynamics, (0, 30), [x0, v0], t_eval=t_eval,
                args=(k, m, b, F0, omega), rtol=1e-9, atol=1e-12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 时域：瞬态 → 稳态
ax1.plot(sol.t, sol.y[0])
ax1.axvline(x=10/0.2, color='r', ls='--', label='transient decay')
ax1.set_title("MEC-012: Forced Oscillator (time domain)")
ax1.set_xlabel("t (s)")
ax1.set_ylabel("x (m)")
ax1.legend()
ax1.grid(True)

# 相图：螺线 → 极限环
ax2.plot(sol.y[0], sol.y[1])
ax2.set_title("MEC-012: Phase Portrait (transient → limit cycle)")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("v (m/s)")
ax2.grid(True)
ax2.axis("equal")

plt.tight_layout()
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
