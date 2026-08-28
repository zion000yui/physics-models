# MEC-100 Elastoplasticity（弹塑性力学）

塑性屈服准则与弹塑性本构关系。在 MEC-053 三维弹性体基础上引入塑性变形。

## 物理背景

材料在小应变范围内先弹后塑：弹性阶段遵循广义胡克定律，超过屈服强度后产生永久变形。

### 屈服准则

| 准则 | 公式 | 物理意义 |
|------|------|---------|
| Tresca | $\tau_{\max} = \max\|\sigma_i - \sigma_j\|/2 \le \sigma_y/2$ | 最大剪应力 |
| von Mises | $\sigma_{eq} = \sqrt{\frac{3}{2} s_{ij} s_{ij}} \le \sigma_y$ | 畸变能 |

静水压力下两者均为零（不屈服）。

### 单轴本构

**理想塑性**：$|\sigma| \ge \sigma_y$ 时 $\sigma = \sigma_y$（应力不变，应变流动）

**线性硬化**：$\sigma = \sigma_y + H \varepsilon_p$

### 能量

- 弹性应变能：$U_e = \sigma^2/(2E)$
- 塑性耗散：$U_p = \int \sigma \, d\varepsilon_p \ge 0$（不可逆）

### 卸载

沿弹性斜率 $E$ 回弹，残余应变 $\varepsilon_{res} = \varepsilon_{max} - \sigma_{max}/E$。

## 与已有 MEC 模型的关系

| 关系 | 说明 |
|------|------|
| MEC-053 弹性体 → MEC-100 | 弹性 → 弹塑性 |
| MEC-050 梁 → MEC-100 | 塑性铰 |
| ν→0.5 → MEC-100 | 塑性流动体积守恒 |
| σ_y→∞ → MEC-053 | 退化为纯弹性 |

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `E` | 杨氏模量 (Pa) | `2.0e11` (钢) |
| `nu` | 泊松比 | `0.3` |
| `sigma_y` | 屈服强度 (Pa) | `250e6` |
| `H` | 线性硬化模量 (Pa) | `5e9` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：屈服准则 + 单轴本构 + 能量 + 卸载 + Bauschinger |
| `scipy_solve.py` | 应力-应变 + 屈服准则 + 能量 + 卸载 + 退化 |
| `test_MEC100_consistency.py` | 屈服 + 本构 + 能量 + 残余 + 退化 + 反例 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC100_consistency.py
```

## 依赖

- numpy
