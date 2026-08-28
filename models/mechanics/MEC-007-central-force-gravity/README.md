# MEC-007 Central Force Gravity（万有引力中心力场 / 开普勒问题）

平方反比中心力场：质点在指向原点的引力 F = -μm/r²·r̂ 作用下运动。
对于束缚态（E < 0），轨迹为以力心为焦点的椭圆——这是开普勒行星运动定律
的动力学根源。本模型与 MEC-006（胡克型中心力 F ∝ -r）共同验证 Bertrand 定理。

## 物理背景

在万有引力中心力场中，质点受到的力与距离平方成反比、方向指向力心：
`F = -μm/r²·r̂`，其中 μ = GM 为引力参数。加速度 `a = -μr/r³` 与质点
质量无关（等效原理），但质量 m 出现在守恒量（角动量、机械能）中。

对于束缚态（比能量 ε < 0），轨道为椭圆，力心位于椭圆的一个焦点
（开普勒第一定律）。特殊情况下退化为圆。

**假设**：
- 力心固定在原点
- 万有引力 `F = -μm/r²·r̂`（μ > 0）
- 质量 m > 0（不影响加速度，但出现在守恒量中）
- 无阻尼、无其他外力
- 运动限于二维平面

## 与 MEC-006 的关键区别

| 特性 | MEC-006（胡克型） | MEC-007（万有引力） |
|------|-------------------|---------------------|
| 力律 | F ∝ -r | F ∝ -1/r² |
| 力心位置 | 椭圆中心 | 椭圆焦点 |
| 轨道周期 | T = 2π/ω₀，与振幅无关 | T = 2π√(a³/μ)，与 a^(3/2) 成正比 |
| 解析解 | x、y 独立简谐振动 | 开普勒方程（半解析） |
| 额外守恒量 | 无 | 偏心率向量（Laplace-Runge-Lenz） |
| 闭合性 | 所有轨道闭合 | 所有束缚轨道闭合 |

**Bertrand 定理**：只有 F ∝ r¹ 和 F ∝ r⁻² 两种幂律中心力能产生对所有
初始条件都闭合的轨道。MEC-006 和 MEC-007 分别验证了这两种情形。

## 数学模型

- 状态：位置 `(x, y)`、速度 `(vx, vy)`
- 参数：引力参数 `mu`、质量 `m`

一阶常微分方程：

```
dx/dt = vx
dy/dt = vy
dvx/dt = -μ·x / r³
dvy/dt = -μ·y / r³
```

其中 `r = √(x² + y²)`。

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

万有引力：`F = -μm/r²·r̂ = -μm·r / r³`

牛顿第二定律：`m·a = F`，因此 `a = -μ·r / r³`

$$
\begin{bmatrix} \dot{v}_x \\ \dot{v}_y \end{bmatrix}
= -\frac{\mu}{r^3} \begin{bmatrix} x \\ y \end{bmatrix}
$$

加速度与质点质量 m 无关（等效原理）。

## 解析解

与 MEC-006 不同，万有引力问题中 x、y 方向耦合，不存在简单的闭式解。
解析解通过开普勒方程半解析地获得：

**轨道根数**（从初始条件提取）：

- 半长轴：`a = -μ/(2ε)`，其中 `ε = v²/2 - μ/r` 为比能量
- 偏心率：`e = |e_vec|`，其中偏心率向量 `e_vec = (v × h)/μ - r̂`
- 近心点幅角：`ω = atan2(e_y, e_x)`
- 平均运动：`n = √(μ/a³)`
- 半通径：`p = h²/μ`

**椭圆轨道**（ε < 0，e < 1）：

平近点角 `M = M₀ + sign(h)·n·t`，通过开普勒方程

$$
M = E - e \sin E
$$

求解偏近点角 E（牛顿迭代），然后

$$
\begin{aligned}
x_{\text{orb}} &= a(\cos E - e) \\
y_{\text{orb}} &= a\sqrt{1-e^2} \sin E
\end{aligned}
$$

旋转近心点幅角 ω 回到惯性系。

**双曲轨道**（ε > 0，e > 1）：

双曲开普勒方程 `M = e·sinh(H) - H`，位置 `x_orb = |a|(e - cosh H)`，
`y_orb = |a|√(e²-1)·sinh H`。

## 守恒量

**角动量（绕原点）**：

$$L = m \cdot (x \cdot v_y - y \cdot v_x)$$

**机械能**：

$$E = \frac{1}{2} m (v_x^2 + v_y^2) - \frac{\mu m}{r}$$

**偏心率向量（Laplace-Runge-Lenz 向量，比单位质量）**：

$$\mathbf{e} = \frac{\mathbf{v} \times \mathbf{h}}{\mu} - \hat{\mathbf{r}}$$

其中 `h = x·vy - y·vx` 为比角动量。该向量的模等于偏心率 e，
方向指向近心点。其守恒性是 r⁻² 幂律中心力的独特性质（隐藏 SO(4)
对称性），在其他幂律中心力中不存在。

## 初始状态约束

任意 `(x0, y0, vx0, vy0)` 都是合法初始状态。

仅要求：
- `mu > 0`
- `m > 0`

**轨道分类**（由初始条件决定）：
- `ε < 0`（`v < v_esc = √(2μ/r)`）：束缚态，椭圆轨道
- `ε = 0`（`v = v_esc`）：抛物线轨道
- `ε > 0`（`v > v_esc`）：非束缚态，双曲轨道

**退化情形**：当初始条件满足 `|v| = √(μ/r)` 且 `v ⊥ r` 时，
e = 0，轨道退化为以力心为圆心的圆。

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `mu` | 引力参数 (m³/s²) | `1.0` |
| `m` | 质量 (kg) | `1.0` |
| `x0` | 初始水平位置 (m) | `1.0` |
| `y0` | 初始垂直位置 (m) | `0.0` |
| `vx0` | 初始水平速度 (m/s) | `0.0` |
| `vy0` | 初始垂直速度 (m/s) | `0.8` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：动力学方程 `dynamics` + 半解析解 `analytical`（开普勒方程） + 角动量/能量/偏心率向量工具函数 + 轨道根数计算 `orbital_elements` |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC007_consistency.py` | 数值解 vs 解析解一致性测试 + 物理检验（含 Bertrand 定理闭合性验证） |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC007_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, orbital_elements

x0, y0, vx0, vy0, mu, m = 1.0, 0.0, 0.0, 0.8, 1.0, 1.0
elem = orbital_elements([x0, y0, vx0, vy0], mu=mu)
T = 2 * np.pi / elem['n']
t_eval = np.linspace(0, T, 401)
sol = solve_ivp(dynamics, (0, T), [x0, y0, vx0, vy0],
                t_eval=t_eval, args=(mu, m), rtol=1e-9, atol=1e-12)

x, y = sol.y[0], sol.y[1]

plt.figure(figsize=(6, 6))
plt.plot(x, y)
plt.plot(0, 0, "ro", label="focus (force center)")
plt.plot(x0, y0, "go", label="start")
plt.title("MEC-007: Central Force Gravity (Kepler)")
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
