# MEC-052 Shell（壳体力学）

圆柱壳力学：Kirchhoff-Love 壳理论。在薄板基础上引入曲率效应，实现薄膜-弯曲耦合。

## 物理背景

圆柱壳（半径 R、壁厚 h、长度 L），满足 Kirchhoff-Love 假设：中面法线变形后仍
垂直于中面。与 MEC-051 板的区别在于曲率引入了薄膜-弯曲耦合。

### 薄膜理论（无矩理论）

内压容器经典公式：
$$\sigma_\theta = \frac{pR}{h} \quad \text{(环向)}, \quad \sigma_x = \frac{pR}{2h} \quad \text{(轴向)}$$

环向应力恒为轴向应力的 **2 倍**。

### 弯曲理论（Donnell 简化，轴对称）

$$D \frac{d^4 w}{dx^4} + \frac{Eh}{R^2} w = p$$

等价于 Winkler 弹性基础梁方程，弹性基础刚度 $k = Eh/R^2$。

衰减特征长度：$\lambda = \left(\frac{D}{k}\right)^{1/4} = \frac{\sqrt{hR}}{[12(1-\nu^2)]^{1/4}}$

## 与已有 MEC 模型的关系

| 关系 | 说明 | 严格性 |
|------|------|--------|
| MEC-051 板 → MEC-052 壳 | 引入曲率 R，增加弹性基础项 Eh/R² | ✓ 几何推广 |
| MEC-050 梁 → 壳弯曲 | Donnell 方程 = 弹性基础梁 | ✓ 数学类比 |
| R→∞ → MEC-051 板 | 曲率项消失，退化为板 | ✓ 严格退化 |

## 数学模型

### 固有频率（轴对称）

$$\omega_n^2 = \frac{D(n\pi/L)^4 + Eh/R^2}{\rho h}$$

$n=0$ 极限：$\omega_0 = \sqrt{E/(\rho R^2)}$（薄膜频率）

### 退化验证

$R \to \infty$ 时 $Eh/R^2 \to 0$，$\omega_n \to (n\pi/L)^2 \sqrt{D/(\rho h)}$ = 板频率 ✓

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `E` | 杨氏模量 (Pa) | `2.0e11` (钢) |
| `h` | 壁厚 (m) | `0.005` |
| `nu` | 泊松比 | `0.3` |
| `rho` | 密度 (kg/m³) | `7850.0` (钢) |
| `R` | 半径 (m) | `0.5` |
| `L` | 长度 (m) | `2.0` |
| `p` | 内压 (Pa) | `1.0e6` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：薄膜力 + Donnell 弯曲解析解 + 固有频率 + 模态动力学 + FD 矩阵 |
| `scipy_solve.py` | 薄膜计算 + BVP 弯曲求解 + 模态动力学 + FD 固有频率 + 退化验证 |
| `test_MEC052_consistency.py` | 薄膜应力比 + 弯曲边界 + BVP + 频率公式 + 退化 + 能量 + FD + 反例 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC052_consistency.py
```

## 依赖

- numpy
- scipy
