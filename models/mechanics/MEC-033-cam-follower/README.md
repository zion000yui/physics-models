# MEC-033 Cam-Follower（凸轮从动件机构）

旋转凸轮通过特定轮廓驱动弹簧加载平移从动件做预定往复运动。实现完整 DRRD
循环（Rise-Dwell-Return-Dwell）和三种标准轮廓（SHM、Cycloidal、3-4-5 多项式）。

## 物理背景

凸轮以角速度 ω 旋转，轮廓决定从动件位移 y(θ)。从动件由弹簧（刚度 k）保持
与凸轮接触。凸轮具有转动惯量 I_cam，从动件具有平动质量 m_f。系统为 1 自由度
（凸轮角 θ），从动件运动由轮廓约束确定。

**假设**：
- 凸轮为刚性，轮廓理想
- 从动件为平移运动（沿 y 轴），无侧向自由度
- 弹簧线性，预紧力通过 k·y 体现
- 无摩擦、无间隙
- 凸轮匀速或由输入力矩 τ 驱动
- 三种标准轮廓可选

### DRRD 循环

| 段 | 角度范围 | y |
|---|---|---|
| Rise | 0 ~ β_r | 0 → h |
| Dwell1 | β_r ~ β_r+β_d1 | h |
| Return | β_r+β_d1 ~ β_r+β_d1+β_re | h → 0 |
| Dwell2 | β_r+β_d1+β_re ~ 2π | 0 |

### 三种轮廓的加速度连续性

| 轮廓 | 加速度连续 | 边界 y'' |
|---|---|---|
| SHM (简谐) | ✗ 有有限跳变 | ±hπ²/(2β²) |
| Cycloidal (摆线) | ✓ 连续 | 0 |
| 3-4-5 多项式 | ✓ 连续 | 0 |

## 与已有 MEC 模型的关系

| 关系 | 说明 | 严格性 |
|------|------|--------|
| 凸轮旋转 | MEC-021 定轴转动 | ✓ 概念复用 |
| 从动件+弹簧 | MEC-010/012 简谐/受迫振子 | ✓ 概念复用 |
| 约束驱动运动 | MEC-030/031 闭环约束 | ✓ 概念复用 |
| 无严格退化 | 凸轮轮廓非任何已有模型的特例 | — |

## 数学模型

### 等效惯量动力学

$$I_{\text{eff}}(\theta) = I_{\text{cam}} + m_f\left(\frac{dy}{d\theta}\right)^2$$

$$I_{\text{eff}}'(\theta) = 2\,m_f\,\frac{dy}{d\theta}\,\frac{d^2y}{d\theta^2} \quad \text{(解析)}$$

$$I_{\text{eff}}\,\alpha + \frac{1}{2}I_{\text{eff}}'\,\omega^2 + k\,y\,\frac{dy}{d\theta} = \tau$$

**无闭式动力学解析解**（变等效惯量），数值积分 + 能量守恒验证。

### 接触力（逆动力学）

$$F = m_f\,\ddot{y} + k\,y, \quad \ddot{y} = \frac{d^2y}{d\theta^2}\omega^2 + \frac{dy}{d\theta}\alpha$$

F > 0：保持接触。F < 0：从动件跳脱。

### 压力角

$$\alpha_p = \arctan\frac{|dy/d\theta|}{r_b + y}$$

## 状态空间表示

```
state = [theta, omega]
```

- `theta` — 凸轮角度（rad）
- `omega` — 凸轮角速度（rad/s）

## 守恒量

| 条件 | 守恒量 |
|------|--------|
| τ=0 | E = ½·I_eff·ω² + ½·k·y² = const |
| τ≠0 | ΔE = τ·Δθ（功-能定理）|

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `h` | 升程 (m) | `0.01` |
| `beta_rise` | 升程角 (rad) | `π/2` |
| `beta_dwell1` | 远休止角 (rad) | `π/4` |
| `beta_return` | 回程角 (rad) | `π/2` |
| `I_cam` | 凸轮转动惯量 (kg·m²) | `0.001` |
| `m_f` | 从动件质量 (kg) | `0.1` |
| `k` | 弹簧刚度 (N/m) | `100.0` |
| `r_b` | 基圆半径 (m) | `0.03` |
| `tau` | 输入力矩 (N·m) | `0.0` |
| `profile` | 轮廓类型 | `'cycloidal'` |

轮廓类型：`'shm'`（简谐）、`'cycloidal'`（摆线）、`'poly345'`（3-4-5 多项式）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：DRRD 三轮廓 + 位置/速度/加速度 + I_eff + I_eff'(解析) + 接触力 + 压力角 + 动力学 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 数值积分并打印统计 |
| `test_MEC033_consistency.py` | 边界值 + 速度/加速度数值验证 + 连续性/跳变 + I_eff第一性原理 + I_eff'解析vs数值 + 能量守恒 + 功-能 + 接触力 + 压力角 + 反例验证 + 参数验证 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC033_consistency.py
```

## 依赖

- numpy
- scipy
