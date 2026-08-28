# MEC-032 Gear（齿轮传动）

一对啮合的外齿轮在平行轴间传递转动。轮齿在节点处保持无滑动滚动接触，
产生恒定传动比。是 030 号段中唯一具有闭式动力学解析解的模型。

## 物理背景

两个外齿轮在节点处啮合，输入齿轮（半径 r₁，惯量 I₁）驱动输出齿轮
（半径 r₂，惯量 I₂）。外啮合使两齿轮反向旋转。给定输入力矩 τ_in 和
负载力矩 τ_load，系统的运动由 1-DOF 等效惯量方程决定。

**假设**：
- 外啮合直齿轮（spur gear），平行轴
- 节点处无滑动滚动接触（理想刚性啮合）
- 齿轮质心在转轴上（平衡齿轮，无重力效应）
- 无摩擦损失、无齿侧间隙
- `r1, r2 > 0`，`I1, I2 ≥ 0`

## 与已有 MEC 模型的关系

| 关系 | 说明 | 严格性 |
|------|------|--------|
| 每个齿轮是 MEC-021 定轴转动 | 绕固定轴旋转 | ✓ 概念复用 |
| 滚动接触约束类比 MEC-024 纯滚动 | v=Rω 的旋转版本 | ✓ 概念相似 |
| I_eq=I₁+i²I₂ 类比 MEC-024 m_eff=m+I/R² | 反射惯量 vs 反射质量 | ✓ 数学结构相似 |
| I₂→0 且 τ_load→0 → MEC-021 | 去除输出齿轮动力学和负载 | ✓ 严格退化（参数极限） |

**与 MEC-024 的数学类比**：MEC-024 的有效质量 m_eff = m + I/R² 将转动惯量
反射到平动；齿轮的等效惯量 I_eq = I₁ + i²·I₂ 将输出惯量反射到输入轴。
数学结构相同（native 项 + reflected 项），但物理对象不同。

## 数学模型

### 坐标系

两齿轮分别绕 O₁ 和 O₂ 旋转，角度从各自 x 轴逆时针计量。外啮合反向旋转。

### 运动学

$$i = \frac{r_1}{r_2} = \frac{z_1}{z_2}, \quad \theta_2 = -i\,\theta_1, \quad \omega_2 = -i\,\omega_1$$

### 等效惯量

$$I_{\text{eq}} = I_1 + i^2\,I_2 = \text{const}$$

### 运动方程

$$I_{\text{eq}}\,\alpha_1 = \tau_{\text{in}} - i\,\tau_{\text{load}}$$

由于 $I_{\text{eq}}$ 为常数，$\alpha_1$ 亦为常数。

### 闭式解析解

$$\alpha_1 = \frac{\tau_{\text{in}} - i\,\tau_{\text{load}}}{I_{\text{eq}}} = \text{const}$$

$$\omega_1(t) = \omega_0 + \alpha_1\,t$$

$$\theta_1(t) = \theta_0 + \omega_0\,t + \frac{1}{2}\alpha_1\,t^2$$

### 接触力

$$F = \frac{\tau_{\text{in}} - I_1\,\alpha_1}{r_1} = \frac{\tau_{\text{load}} + i\,I_2\,\alpha_1}{r_2}$$

两式给出相同结果（Newton 第三定律 + 约束一致性）。

### 功率

$$P_{\text{in}} = \tau_{\text{in}}\,\omega_1, \quad P_{\text{out}} = \tau_{\text{load}}\,i\,\omega_1$$

$$P_{\text{in}} = P_{\text{out}} + \frac{dT}{dt}$$

## 状态空间表示

```
state = [theta1, omega1]
```

- `theta1` — 输入齿轮角度（rad）
- `omega1` — 输入齿轮角速度（rad/s）

输出齿轮状态由约束唯一确定（1 自由度）。

## 守恒量

| 条件 | 守恒量 |
|------|--------|
| τ_in = i·τ_load | E = ½·I_eq·ω₁² = const（α=0）|
| τ_in ≠ i·τ_load | ΔE = (τ_in - i·τ_load)·Δθ₁（功-能定理）|

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `r1` | 输入轮节圆半径 (m) | `0.1` |
| `r2` | 输出轮节圆半径 (m) | `0.2` |
| `I1` | 输入轮转动惯量 (kg·m²) | `0.01` |
| `I2` | 输出轮转动惯量 (kg·m²) | `0.04` |
| `tau_in` | 输入力矩 (N·m) | `1.0` |
| `tau_load` | 负载力矩 (N·m) | `0.0` |

传动比 i = r₁/r₂ = z₁/z₂（z 为齿数，需相同模数 m = 2r/z）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：传动比 + 等效惯量 + 运动学 + 接触力 + 功率 + 动力学 + 闭式解析解 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 数值积分并与解析解对照 |
| `test_MEC032_consistency.py` | 运动学约束 + 传动比 + I_eff第一性原理 + 解析vs数值 + 接触力一致性 + 接触力反例 + 功率平衡 + 能量守恒 + 功-能 + 退化 + 常加速度 + 非法参数 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC032_consistency.py
```

## 依赖

- numpy
- scipy
