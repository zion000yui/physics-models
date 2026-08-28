# MEC-009 Particle with Drag（速度相关阻力下的质点运动）

速度相关阻力下的质点运动：质点在重力场中运动，同时受线性阻力
（Stokes 阻力，F = -b·v）和/或二次阻力（Newton 阻力，F = -c·|v|·v）。
当 b=0、c=0 时精确退化为 MEC-003 抛体运动。

## 物理背景

真实物理环境中的质点运动通常伴随阻力。最常见的两种阻力模型：
- **线性阻力（Stokes 阻力）**：F = -b·v，适用于低 Reynolds 数（层流）
- **二次阻力（Newton 阻力）**：F = -c·|v|·v，适用于高 Reynolds 数（湍流）

两者可以同时存在（混合阻力），也可以单独使用。

**假设**：
- 质点在均匀重力场中运动
- 重力加速度 `g ≥ 0`（向下）
- 线性阻力系数 `b ≥ 0`
- 二次阻力系数 `c ≥ 0`
- 质量 `m > 0`
- 运动限于二维平面
- 阻力始终与速度方向相反

## 与 MEC-003 的关系

| 特性 | MEC-003（抛体） | MEC-009（阻力） |
|------|----------------|-----------------|
| 阻力 | 无 | 线性 + 二次 |
| 水平速度 | 恒定 | 衰减（有线速度时） |
| 机械能 | 守恒 | 耗散 |
| 终态速度 | 无 | 有（v_t = m·g/b 或 √(m·g/c)） |
| 解析解 | 闭式 | 线性有闭式；二次仅 1D 垂直有闭式 |

当 b=0、c=0 时，MEC-009 精确退化为 MEC-003：dvx/dt=0, dvy/dt=-g。

## 数学模型

- 状态：位置 `(x, y)`、速度 `(vx, vy)`
- 参数：重力 `g`、线性阻力系数 `b`、二次阻力系数 `c`、质量 `m`

一阶常微分方程：

```
dx/dt = vx
dy/dt = vy
dvx/dt = -(b/m)·vx - (c/m)·|v|·vx
dvy/dt = -g - (b/m)·vy - (c/m)·|v|·vy
```

其中 `|v| = √(vx² + vy²)`。

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

合力 = 重力 + 线性阻力 + 二次阻力：

$$
\mathbf{F} = -mg\,\hat{\mathbf{y}} - b\,\mathbf{v} - c\,|\mathbf{v}|\,\mathbf{v}
$$

牛顿第二定律 `m·a = F`，因此：

$$
\begin{aligned}
a_x &= -\frac{b}{m} v_x - \frac{c}{m} |\mathbf{v}| v_x \\
a_y &= -g - \frac{b}{m} v_y - \frac{c}{m} |\mathbf{v}| v_y
\end{aligned}
$$

## 解析解

**1. 无阻力（b=0, c=0）**：标准抛体运动（MEC-003 极限）

$$
\begin{aligned}
x(t) &= x_0 + v_{x0} t \\
y(t) &= y_0 + v_{y0} t - \tfrac{1}{2} g t^2
\end{aligned}
$$

**2. 纯线性阻力（c=0, b>0）**：闭式指数解

令 `γ = b/m`：

$$
\begin{aligned}
v_x(t) &= v_{x0} e^{-\gamma t} \\
v_y(t) &= \left(v_{y0} + \frac{g}{\gamma}\right) e^{-\gamma t} - \frac{g}{\gamma} \\
x(t) &= x_0 + \frac{v_{x0}}{\gamma}\left(1 - e^{-\gamma t}\right) \\
y(t) &= y_0 + \frac{v_{y0} + g/\gamma}{\gamma}\left(1 - e^{-\gamma t}\right) - \frac{g}{\gamma} t
\end{aligned}
$$

**3. 纯二次阻力 + 一维垂直（b=0, c>0, vx0=0）**：分段解析解

终端速度 `v_t = √(m·g/c)`

上升阶段（vy > 0），令 `θ(t) = θ₀ - (g/v_t)·t`，`θ₀ = arctan(vy0/v_t)`：

$$
v_y(t) = v_t \tan\theta(t), \quad
y(t) = y_0 + \frac{v_t^2}{g} \ln\frac{\cos\theta(t)}{\cos\theta_0}
$$

下降阶段（vy < 0），令 `φ(t) = (g/v_t)·(t - t_{\text{apex}})`：

$$
v_y(t) = -v_t \tanh\phi(t), \quad
y(t) = y(t_{\text{apex}}) - \frac{v_t^2}{g} \ln\cosh\phi(t)
$$

**4. 一般二维二次阻力或混合阻力**：无闭式解析解。

## 守恒量与终态速度

**无阻力时**：机械能 `E = ½m(vx²+vy²) + mgy` 守恒（退化为 MEC-003）。

**有阻力时**：机械能单调递减（阻力做负功），无守恒量。

**终态速度**（terminal velocity）：当 `dv/dt = 0` 时：

| 阻力类型 | 终态速度 |
|---------|---------|
| 纯线性 | `v_t = m·g/b` |
| 纯二次 | `v_t = √(m·g/c)` |
| 混合 | `v_t = (-b + √(b² + 4cmg)) / (2c)` |

## 初始状态约束

任意 `(x0, y0, vx0, vy0)` 都是合法初始状态。

仅要求：
- `g ≥ 0`
- `b ≥ 0`、`c ≥ 0`
- `m > 0`

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `g` | 重力加速度 (m/s²) | `9.81` |
| `b` | 线性阻力系数 (N·s/m) | `0.5` |
| `c` | 二次阻力系数 (N·s²/m²) | `0.0` |
| `m` | 质量 (kg) | `1.0` |
| `x0` | 初始水平位置 (m) | `0.0` |
| `y0` | 初始垂直位置 (m) | `0.0` |
| `vx0` | 初始水平速度 (m/s) | `10.0` |
| `vy0` | 初始垂直速度 (m/s) | `15.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：动力学方程 `dynamics` + 解析解 `analytical`（线性阻力闭式解 + 二次阻力 1D 分段解）+ 终态速度 `terminal_velocity` + 参数验证 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC009_consistency.py` | 数值解 vs 解析解一致性测试 + 物理检验（含无阻力退化 MEC-003、终态速度、能量耗散） |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC009_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics

g, b, c, m = 9.81, 0.5, 0.0, 1.0
x0, y0, vx0, vy0 = 0.0, 0.0, 10.0, 15.0
t_eval = np.linspace(0, 5, 401)
sol = solve_ivp(dynamics, (0, 5), [x0, y0, vx0, vy0],
                t_eval=t_eval, args=(g, b, c, m), rtol=1e-9, atol=1e-12)

# 同时画无阻力轨迹做对照
sol_nd = solve_ivp(dynamics, (0, 5), [x0, y0, vx0, vy0],
                   t_eval=t_eval, args=(9.81, 0, 0, 1), rtol=1e-9, atol=1e-12)

plt.figure(figsize=(8, 5))
plt.plot(sol.y[0], sol.y[1], label="with drag (b=0.5)")
plt.plot(sol_nd.y[0], sol_nd.y[1], "--", label="no drag (MEC-003)")
plt.plot(0, 0, "go", label="start")
plt.title("MEC-009: Particle with Drag")
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
