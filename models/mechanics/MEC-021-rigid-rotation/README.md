# MEC-021 Rigid Rotation（定轴转动）

定轴转动：刚体绕固定轴转动，核心方程 I·θ̈ = τ。
是 020 号段刚体动力学的第二个模型，引入转动惯量和角动量概念。

## 物理背景

刚体定轴转动是最基本的转动模型。转动惯量 I 度量刚体对转动的惯性，
外力矩 τ 驱动角加速度 α = τ/I。这与质点力学中 m·a = F 形成精确类比：
I ↔ m，α ↔ a，τ ↔ F。

**假设**：
- 刚体绕固定轴（z 轴）转动
- 转动惯量 `I > 0`
- 外力矩 `tau`（恒力矩或一般力矩函数）
- 无阻尼
- 一维转动（仅描述绕一个轴的运动）

## 与 MEC-001/002/010/015 的关系

| 条件 | 退化/对应 | 物理依据 | 严格性 |
|------|---------|---------|--------|
| τ = 0 | MEC-001（转动版） | ω = const，匀速转动 | ✓ I·α=0 → α=0 |
| τ = const | MEC-002（转动版） | α = τ/I = const，匀角加速 | ✓ 对应 F/m=const |
| τ = -κθ | MEC-010（角向） | θ̈+(κ/I)θ=0，ω₀=√(κ/I) | ✓ κ/I ↔ k/m |
| τ = -mgL sin(θ), I=mL² | MEC-015 | θ̈+(g/L)sin(θ)=0 | ✓ mgL/I=g/L |

MEC-021 是**一般**定轴转动模型。MEC-015 是其在特定力矩 τ=-mgL sin(θ)
下的特例，不是反过来。

## 数学模型

- 状态：角位移 `theta`、角速度 `omega`
- 参数：转动惯量 `I`、外力矩 `tau`

一阶常微分方程：

```
dθ/dt = ω
dω/dt = τ / I
```

## 状态空间表示

```
state = [theta, omega]
```

- `theta` —— 角位移（rad）
- `omega` —— 角速度（rad/s）

## 微分方程推导

刚体定轴转动方程：

$$
I \ddot{\theta} = \tau
$$

其中 I 为转动惯量（kg·m²），τ 为外力矩（N·m），θ̈ 为角加速度（rad/s²）。

## 解析解

恒力矩（τ = const）：

$$
\begin{aligned}
\theta(t) &= \theta_0 + \omega_0 t + \frac{1}{2}\frac{\tau}{I} t^2 \\
\omega(t) &= \omega_0 + \frac{\tau}{I} t
\end{aligned}
$$

## 守恒量

**角动量**：`L = I·ω`

无力矩时（τ=0）角动量守恒：L = const

## 初始状态约束

任意 `(theta0, omega0)` 都是合法初始状态。

仅要求：`I > 0`

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `I` | 转动惯量 (kg·m²) | `1.0` |
| `tau` | 外力矩 (N·m) | `0.0` |
| `theta0` | 初始角位移 (rad) | `1.0` |
| `omega0` | 初始角速度 (rad/s) | `0.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：定轴转动方程 `dynamics` + 恒力矩解析解 `analytical` + 角动量工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC021_consistency.py` | 恒力矩一致性 + 四种退化验证 + 角动量守恒 + 转动惯量标度 + 非法参数 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC021_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics

# 恒力矩转动
I, tau = 2.0, 3.0
t_eval = np.linspace(0, 5, 100)
sol = solve_ivp(dynamics, (0, 5), [1.0, 0.5],
                t_eval=t_eval, args=(I, tau), rtol=1e-9, atol=1e-12)

plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0], label="θ(t)")
plt.plot(sol.t, sol.y[1], "--", label="ω(t)")
plt.title("MEC-021: Rigid Rotation (constant torque)")
plt.xlabel("t (s)")
plt.legend()
plt.grid(True)
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
