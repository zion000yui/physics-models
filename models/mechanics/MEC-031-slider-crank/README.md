# MEC-031 Slider-Crank（曲柄滑块机构）

曲柄滑块机构：由曲柄（半径 r）、连杆（长度 l）和滑块组成。曲柄绕固定铰
旋转，连杆连接曲柄销和滑块，滑块被约束沿直线运动。是内燃机、压缩机的核心
机构。机构拓扑为 **R-R-R-P**（3 个转动副 + 1 个移动副）。

## 物理背景

曲柄旋转带动连杆做一般平面运动，连杆推动滑块沿直线往复运动。给定曲柄角 θ，
滑块位置 x 和连杆角 φ 由几何约束唯一确定（1 自由度）。

**假设**：
- 所有杆为刚性，无变形
- 运动副为理想副（无间隙、无摩擦）
- 平面运动（2D），滑块沿 x 轴运动
- `r > 0`（曲柄半径），`l > r`（连杆长度，否则机构锁死）
- 各杆质量 `m_crank, m_rod, m_sl ≥ 0`
- 均匀杆默认：质心在中点，转动惯量 m·l²/12

## 与已有 MEC 模型的关系

| 关系 | 说明 | 严格性 |
|------|------|--------|
| 曲柄绕固定铰转动 | MEC-021 概念 | ✓ 概念复用 |
| 连杆做一般平面运动 | MEC-022 概念 | ✓ 概念复用 |
| 滑块沿直线平动 | MEC-020 概念 | ✓ 概念复用 |
| 约束减少自由度 | MEC-024 概念 | ✓ 概念复用 |
| 与 MEC-030 四连杆 | l→∞ 时摇杆弧→直线，**数学极限** | ✗ **非严格退化** |

**与 MEC-030 的关系**：曲柄滑块与四连杆具有数学上的极限联系——当四连杆的
摇杆长度 l₄→∞ 时，摇杆末端的弧线运动趋近直线运动，运动学上趋近滑块。但
R-R-R-R 变为 R-R-R-P 是**约束拓扑变化**（转动副→移动副），不是参数退化。
因此**不是严格退化**。

**l→∞ 极限**：连杆无限长时，滑块位移趋近 x ≈ r cos θ + l，即去除常数偏移
后的运动为简谐。这是数学极限，不是退化到任何已有 MEC 模型。

## 数学模型

### 坐标系

O 在原点，滑块沿 x 轴运动。θ 为曲柄角（从 x 轴逆时针），φ 为连杆角。

### 位置分析

$$x = r\cos\theta + \sqrt{l^2 - r^2\sin^2\theta}$$

$$\phi = \arcsin\frac{r\sin\theta}{l}$$

### 速度分析

$$\dot{x} = \frac{dx}{d\theta}\cdot\omega, \quad \dot{\phi} = \frac{d\phi}{d\theta}\cdot\omega$$

$$\frac{dx}{d\theta} = -r\sin\theta - \frac{r^2\sin\theta\cos\theta}{\sqrt{l^2-r^2\sin^2\theta}}$$

$$\frac{d\phi}{d\theta} = \frac{r\cos\theta}{\sqrt{l^2-r^2\sin^2\theta}}$$

### 等效惯量动力学

$$I_{\text{eff}}(\theta)\,\alpha + \frac{1}{2}I_{\text{eff}}'(\theta)\,\omega^2 + \frac{dV}{d\theta} = \tau$$

$$I_{\text{eff}} = I_O + m_{\text{rod}}\,\frac{v_{\text{cm}}^2}{\omega^2} + I_{\text{rod}}\left(\frac{d\phi}{d\theta}\right)^2 + m_{\text{sl}}\left(\frac{dx}{d\theta}\right)^2$$

其中 $I_O = I_{\text{crank}} + m_{\text{crank}}\,r_{\text{cm}}^2$，
连杆质心速度 $v_{\text{cm}}^2 = (dx_{\text{cm}}/d\theta)^2 + (dy_{\text{cm}}/d\theta)^2$。

**动力学 ODE 无闭式解析解**（变等效惯量），通过数值积分 + 能量守恒验证。

## 状态空间表示

```
state = [theta, omega]
```

- `theta` — 曲柄角度（rad）
- `omega` — 曲柄角速度（rad/s）

## 极限位置（Toggle）

曲柄与连杆共线时滑块在极限位置，滑块速度 ẋ=0：

| 位置 | θ | x | 说明 |
|------|---|---|------|
| TDC（上止点） | 0 | r + l | 伸展极限 |
| BDC（下止点） | π | l - r | 折叠极限 |

行程 = 2r（活塞冲程）。

## 守恒量

| 条件 | 守恒量 |
|------|--------|
| τ=0, g=0 | E = ½·I_eff·ω² = const |
| τ≠0, g=0 | ΔE = τ·Δθ（功-能定理）|
| g>0 | E = ½·I_eff·ω² + V = const（τ=0 时）|

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `r` | 曲柄半径 (m) | `0.3` |
| `l` | 连杆长度 (m) | `1.0` |
| `m_crank` | 曲柄质量 (kg) | `1.0` |
| `m_rod` | 连杆质量 (kg) | `1.0` |
| `m_sl` | 滑块质量 (kg) | `1.0` |
| `r_cm` | 曲柄质心距 O (m) | `r/2` |
| `l_cm` | 连杆质心距 A (m) | `l/2` |
| `I_crank` | 曲柄转动惯量 (kg·m²) | `m·r²/12` |
| `I_rod` | 连杆转动惯量 (kg·m²) | `m·l²/12` |
| `g` | 重力加速度 (m/s²) | `0.0` |
| `tau` | 输入力矩 (N·m) | `0.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：位置/速度/加速度解析 + 极限位置 + 等效惯量动力学 + 能量 |
| `scipy_solve.py` | 用 SciPy `solve_ivp` 数值积分并打印统计 |
| `test_MEC031_consistency.py` | 闭环约束 + 速度/加速度数值验证 + TDC/BDC + I_eff + 动能独立验证 + 能量守恒 + 功-能 + 反例验证 + l→∞ 极限 + 参数验证 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC031_consistency.py
```

## 依赖

- numpy
- scipy
- matplotlib（仅可视化）
