# MEC-014 Coupled Oscillators（耦合振子）

耦合振子：两个质量块通过弹簧串联并耦合振动，引入简正模态（normal modes）概念。
是多自由度振动系统的标准入门模型，为 MEC-013 双摆、MEC-060 分析力学和
MEC-080 多体动力学的重要过渡。

## 物理背景

物理模型为三个弹簧串联两个质量块：
```
墙壁 —[k1]— m1 —[kc]— m2 —[k2]— 墙壁
```
质点 1 受左侧弹簧 k1 和耦合弹簧 kc 作用，质点 2 受右侧弹簧 k2 和耦合弹簧 kc
作用。系统有两个自由度 (x1, x2)，运动可分解为两个独立的简正模态。

**假设**：
- 弹性恢复力服从胡克定律（线性弹簧）
- 质量 `m1 > 0`、`m2 > 0`
- 弹簧系数 `k1 > 0`、`k2 > 0`、`kc ≥ 0`
- 无阻尼、无外力
- 一维运动
- 平衡位置在 x1=0, x2=0

## 与 MEC-010 的关系

| 特性 | MEC-010 | MEC-014 |
|------|---------|---------|
| 自由度 | 1 (x) | 2 (x1, x2) |
| 状态维度 | 2D [x,v] | 4D [x1,x2,v1,v2] |
| 频率 | 单一 ω₀ | 两个简正频率 ω₁, ω₂ |
| 模态 | 单一 | 两个正交简正模态 |
| 相空间 | x-v 椭圆 | 4D 相空间，可投影到模态空间 |

当 kc=0 时退化为两个独立的 MEC-010 简谐振子。

## 数学模型

- 状态：位移 `(x1, x2)`、速度 `(v1, v2)`
- 参数：`m1, m2, k1, k2, kc`

一阶常微分方程：

```
dx1/dt = v1
dx2/dt = v2
dv1/dt = -(k1+kc)/m1 · x1 + kc/m1 · x2
dv2/dt = kc/m2 · x1 - (k2+kc)/m2 · x2
```

## 状态空间表示

```
state = [x1, x2, v1, v2]
```

- `x1, x2` —— 质点 1、2 的位移（m）
- `v1, v2` —— 质点 1、2 的速度（m/s）

## 微分方程推导

合力：
- 质点 1：`F1 = -k1·x1 + kc·(x2-x1) = -(k1+kc)·x1 + kc·x2`
- 质点 2：`F2 = -k2·x2 + kc·(x1-x2) = kc·x1 - (k2+kc)·x2`

矩阵形式：

$$
\mathbf{M} \ddot{\mathbf{x}} + \mathbf{K} \mathbf{x} = 0
$$

其中 $\mathbf{M} = \text{diag}(m_1, m_2)$，$\mathbf{K} = \begin{pmatrix} k_1+k_c & -k_c \\ -k_c & k_2+k_c \end{pmatrix}$

## 简正模态

通过广义特征值问题 $\mathbf{K}\boldsymbol{\phi} = \omega^2 \mathbf{M}\boldsymbol{\phi}$ 求解：

**对称系统**（m1=m2=m, k1=k2=k）：

| 模态 | 模态形状 | 频率 |
|------|---------|------|
| 同相（in-phase） | [1, 1] | ω₁ = √(k/m) |
| 反相（anti-phase） | [1, -1] | ω₂ = √((k+2kc)/m) |

通解 = 两个简正模态的线性叠加：

$$
\mathbf{x}(t) = A_1 \boldsymbol{\phi}_1 \cos(\omega_1 t + \varphi_1) + A_2 \boldsymbol{\phi}_2 \cos(\omega_2 t + \varphi_2)
$$

## 解析解

通过简正模态分解：
1. 将初始条件投影到模态空间：$\mathbf{q}_0 = \mathbf{P}^{-1} \mathbf{x}_0$
2. 在每个模态空间中独立求解简谐振动
3. 叠加回物理坐标：$\mathbf{x} = \mathbf{P} \cdot \mathbf{q}$

## 守恒量

**总机械能**：

$$
E = \frac{1}{2} m_1 v_1^2 + \frac{1}{2} m_2 v_2^2 + \frac{1}{2} k_1 x_1^2 + \frac{1}{2} k_2 x_2^2 + \frac{1}{2} k_c (x_2 - x_1)^2
$$

无阻尼时机械能守恒。

## 相空间与模态空间表示

本模型有 4 维相空间 (x1, x2, v1, v2)。与 MEC-010 的 2D x-v 相图不同，
多自由度系统的相空间维度更高。通过简正模态变换可将 4D 相空间投影到
两个独立的 2D 模态子空间 (q₁, q̇₁) 和 (q₂, q̇₂)，每个子空间内的行为
等价于一个独立的 MEC-010 简谐振子——椭圆轨迹。

## 初始状态约束

任意 `(x1_0, x2_0, v1_0, v2_0)` 都是合法初始状态。

仅要求：`m1 > 0`、`m2 > 0`、`k1 > 0`、`k2 > 0`、`kc ≥ 0`

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `m1` | 质点 1 质量 (kg) | `1.0` |
| `m2` | 质点 2 质量 (kg) | `1.0` |
| `k1` | 左侧弹簧 (N/m) | `1.0` |
| `k2` | 右侧弹簧 (N/m) | `1.0` |
| `kc` | 耦合弹簧 (N/m) | `0.5` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：动力学方程 `dynamics` + 简正模态分解解析解 `analytical` + 简正模态/刚度矩阵/质量矩阵/机械能工具函数 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 做数值积分并打印误差 |
| `test_MEC014_consistency.py` | 数值解 vs 解析解 + 简正频率 + 模态形状 + 退化 + 能量守恒 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC014_consistency.py
```

## 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from model import dynamics, normal_modes

m, k, kc = 1.0, 1.0, 0.5
modes = normal_modes(m, m, k, k, kc)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 同相模态
sol1 = solve_ivp(dynamics, (0, 10), [1, 1, 0, 0],
                 t_eval=np.linspace(0, 10, 401), args=(m, m, k, k, kc),
                 rtol=1e-9, atol=1e-12)
axes[0,0].plot(sol1.t, sol1.y[0], label="x1")
axes[0,0].plot(sol1.t, sol1.y[1], "--", label="x2")
axes[0,0].set_title(f"In-phase mode (ω={modes[0]['omega']:.3f})")
axes[0,0].legend(); axes[0,0].grid(True)

# 反相模态
sol2 = solve_ivp(dynamics, (0, 10), [1, -1, 0, 0],
                 t_eval=np.linspace(0, 10, 401), args=(m, m, k, k, kc),
                 rtol=1e-9, atol=1e-12)
axes[0,1].plot(sol2.t, sol2.y[0], label="x1")
axes[0,1].plot(sol2.t, sol2.y[1], "--", label="x2")
axes[0,1].set_title(f"Anti-phase mode (ω={modes[1]['omega']:.3f})")
axes[0,1].legend(); axes[0,1].grid(True)

# 模态空间相图：同相
axes[1,0].plot(sol1.y[0], sol1.y[2])
axes[1,0].set_title("Mode 1 phase space (q1, qd1)")
axes[1,0].set_xlabel("x1 (m)"); axes[1,0].set_ylabel("v1 (m/s)")
axes[1,0].grid(True); axes[1,0].axis("equal")

# 模态空间相图：反相
axes[1,1].plot(sol2.y[0], sol2.y[2])
axes[1,1].set_title("Mode 2 phase space (q2, qd2)")
axes[1,1].set_xlabel("x1 (m)"); axes[1,1].set_ylabel("v1 (m/s)")
axes[1,1].grid(True); axes[1,1].axis("equal")

plt.tight_layout()
plt.show()
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
