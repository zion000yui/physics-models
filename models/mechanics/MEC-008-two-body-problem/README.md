# MEC-008 Two-Body Problem（二体问题）

二体问题：两个质点在互相万有引力作用下运动。通过约化质量（reduced mass）
和相对坐标（relative coordinates）化简为等效的单体开普勒问题，
是 MEC-007 的直接推广。当 m2 >> m1 时退化为 MEC-007 的单体问题。

## 物理背景

两个质点（质量 m1、m2）在互相万有引力作用下运动。通过引入约化质量
μ_red = m1·m2/(m1+m2) 和相对坐标 r_rel = r1 - r2，二体问题可化简为
等效单体问题：相对运动满足与 MEC-007 完全相同的开普勒方程，
引力参数 μ = G·(m1+m2)。质心在无外力下做匀速直线运动。

**假设**：
- 两质点间仅有万有引力，无其他外力
- 引力常数 `G > 0`
- 质量 `m1 > 0`、`m2 > 0`
- 两质点初始位置不重合（r_rel ≠ 0）
- 运动限于二维平面
- 无阻尼、无碰撞

## 与 MEC-007 的关系

| 特性 | MEC-007（单体） | MEC-008（二体） |
|------|----------------|------------------|
| 力心 | 固定在原点 | 质点 2 可动 |
| 引力参数 | mu = GM（给定） | mu = G·(m1+m2) |
| 状态维度 | 4D [x,y,vx,vy] | 8D [x1,y1,vx1,vy1,x2,y2,vx2,vy2] |
| 约化质量 | 无 | μ_red = m1·m2/(m1+m2) |
| 质心运动 | 无 | 匀速直线运动 |
| 解析解 | 开普勒方程 | 质心运动 + 相对开普勒方程 |

当 m2 >> m1 时，质点 2 近似不动（固定力心），质点 1 的运动退化为
MEC-007 的单体开普勒问题，引力参数 μ ≈ G·m2。

## 数学模型

- 状态：质点 1 位置 `(x1, y1)`、速度 `(vx1, vy1)`，质点 2 位置 `(x2, y2)`、速度 `(vx2, vy2)`
- 参数：引力常数 `G`、质量 `m1`、`m2`

一阶常微分方程：

```
dx1/dt = vx1                    dx2/dt = vx2
dy1/dt = vy1                    dy2/dt = vy2
dvx1/dt = -G·m2·(x1-x2) / r³   dvx2/dt = -G·m1·(x2-x1) / r³
dvy1/dt = -G·m2·(y1-y2) / r³   dvy2/dt = -G·m1·(y2-y1) / r³
```

其中 `r = √((x1-x2)² + (y1-y2)²)`。

## 状态空间表示

```
state = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
```

其中：

- `x1, y1` —— 质点 1 位置（m）
- `vx1, vy1` —— 质点 1 速度（m/s）
- `x2, y2` —— 质点 2 位置（m）
- `vx2, vy2` —— 质点 2 速度（m/s）

## 微分方程推导

万有引力：`F_12 = -G·m1·m2/r²·r̂`（质点 1 受质点 2 的引力）

牛顿第二定律：
- 质点 1：`m1·a1 = F_12`，因此 `a1 = -G·m2·(r1-r2)/r³`
- 质点 2：`m2·a2 = -F_12`，因此 `a2 = -G·m1·(r2-r1)/r³`

约化描述：
- 相对运动：`d²r_rel/dt² = -μ·r_rel/r³`，其中 `μ = G·(m1+m2)`
- 质心运动：`R_cm(t) = R_cm(0) + V_cm·t`（匀速直线运动）

## 解析解

通过约化质量和相对坐标，二体问题分解为：

**1. 质心运动**（匀速直线运动）：

$$
\mathbf{R}_{\text{cm}}(t) = \mathbf{R}_{\text{cm}}(0) + \mathbf{V}_{\text{cm}} \cdot t
$$

**2. 相对运动**（等效开普勒问题，与 MEC-007 相同）：

引力参数 `μ = G·(m1+m2)`，通过开普勒方程求解：

$$
M = E - e \sin E \quad (\text{椭圆})
$$

$$
\mathbf{r}_{\text{orb}} = a(\cos E - e,\; \sqrt{1-e^2} \sin E)
$$

旋转近心点幅角 ω 回到惯性系。

**3. 重建**：

$$
\mathbf{r}_1 = \mathbf{R}_{\text{cm}} + \frac{m_2}{m_1+m_2} \mathbf{r}_{\text{rel}}
$$

$$
\mathbf{r}_2 = \mathbf{R}_{\text{cm}} - \frac{m_1}{m_1+m_2} \mathbf{r}_{\text{rel}}
$$

## 守恒量

**总动量**：

$$\mathbf{P} = m_1 \mathbf{v}_1 + m_2 \mathbf{v}_2$$

**总角动量**：

$$L = m_1(x_1 v_{y1} - y_1 v_{x1}) + m_2(x_2 v_{y2} - y_2 v_{x2})$$

**总机械能**：

$$E = \frac{1}{2} m_1 |\mathbf{v}_1|^2 + \frac{1}{2} m_2 |\mathbf{v}_2|^2 - \frac{G m_1 m_2}{r}$$

**质心速度**（恒定）：

$$\mathbf{V}_{\text{cm}} = \frac{m_1 \mathbf{v}_1 + m_2 \mathbf{v}_2}{m_1 + m_2}$$

## 初始状态约束

任意 `(x1, y1, vx1, vy1, x2, y2, vx2, vy2)` 都是合法初始状态。

仅要求：
- `G > 0`
- `m1 > 0`、`m2 > 0`
- 两质点初始位置不重合

**退化情形**：当相对速度满足 `|v_rel| = √(μ/r_rel)` 且 `v_rel ⊥ r_rel` 时，
偏心率 e = 0，两个质点绕共同质心做匀速圆周运动。

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `G` | 引力常数 | `1.0` |
| `m1` | 质点 1 质量 (kg) | `1.0` |
| `m2` | 质点 2 质量 (kg) | `1.0` |
| `x1, y1` | 质点 1 初始位置 (m) | `1.0, 0.0` |
| `vx1, vy1` | 质点 1 初始速度 (m/s) | `0.0, 0.3` |
| `x2, y2` | 质点 2 初始位置 (m) | `-1.0, 0.0` |
| `vx2, vy2` | 质点 2 初始速度 (m/s) | `0.0, -0.3` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：二体动力学 `dynamics` + 半解析解 `analytical`（质心+相对开普勒）+ 守恒量/约化质量/质心/相对坐标工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC008_consistency.py` | 数值解 vs 解析解一致性测试 + 物理检验（含大质量极限逼近 MEC-007 验证） |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC008_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, relative_orbital_elements

G, m1, m2 = 1.0, 1.0, 1.0
x1, y1, vx1, vy1 = 1.0, 0.0, 0.0, 0.3
x2, y2, vx2, vy2 = -1.0, 0.0, 0.0, -0.3
state0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
elem = relative_orbital_elements(state0, G, m1, m2)
T = 2 * np.pi / elem['n']
t_eval = np.linspace(0, T, 401)
sol = solve_ivp(dynamics, (0, T), state0, t_eval=t_eval,
                args=(G, m1, m2), rtol=1e-9, atol=1e-12)

plt.figure(figsize=(6, 6))
plt.plot(sol.y[0], sol.y[1], label="body 1")
plt.plot(sol.y[4], sol.y[5], label="body 2")
plt.plot(0, 0, "k+", label="center of mass")
plt.title("MEC-008: Two-Body Problem")
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
