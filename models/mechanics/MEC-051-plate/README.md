# MEC-051 Plate（薄板弯曲）

Kirchhoff-Love 薄板弯曲理论：欧拉-伯努利梁的二维推广。板的横向位移 w(x,y,t)
满足双调和方程。

## 物理背景

薄板（厚度远小于面内尺寸 h << a, b），满足 Kirchhoff 假设：变形前垂直于中面的
法线变形后仍垂直于中面，忽略横向剪切变形。

### 控制方程

**静态**：
$$D \nabla^4 w = q(x,y), \quad D = \frac{Eh^3}{12(1-\nu^2)}$$

**动态**：
$$D \nabla^4 w + \rho h \ddot{w} = q(x,y,t)$$

其中 $\nabla^4 = \frac{\partial^4}{\partial x^4} + 2\frac{\partial^4}{\partial x^2\partial y^2} + \frac{\partial^4}{\partial y^4}$

### 边界条件（简支矩形板）

四边铰支：$w=0$, $M=0$（弯矩为零），$M = -D\nabla^2 w$。

## 与已有 MEC 模型的关系

| 关系 | 说明 | 严格性 |
|------|------|--------|
| MEC-050 梁 → MEC-051 板 | 一维 → 二维截面惯性矩 → 板厚³/(1-ν²) | ✓ 维度推广 |
| D = Eh³/[12(1-ν²)] vs EI = E·bh³/12 | 板取单位宽度并修正泊松比 | ✓ 结构对应 |
| MEC-051 → MEC-052 壳 | 引入曲率效应 | ✓ 几何推广 |
| MEC-051 → MEC-053 3D 弹性体 | 薄板近似 → 完整 3D 弹性理论 | ✓ 理论递进 |

## 数学模型

### 静态 Navier 解（简支板均布载荷）

$$w(x,y) = \sum_m \sum_n W_{mn} \sin\frac{m\pi x}{a} \sin\frac{n\pi y}{b}$$

$$W_{mn} = \frac{q_{mn}}{D\pi^4\left(\frac{m^2}{a^2}+\frac{n^2}{b^2}\right)^2}, \quad q_{mn} = \frac{16q}{\pi^2 mn} \;\;(m,n\text{ 奇})$$

中心挠度系数（方板）：$\alpha \approx 0.00406$

### 固有频率

$$\omega_{mn} = \pi^2\left(\frac{m^2}{a^2}+\frac{n^2}{b^2}\right)\sqrt{\frac{D}{\rho h}}$$

### 模态形状

$$\varphi_{mn}(x,y) = \sin\frac{m\pi x}{a}\sin\frac{n\pi y}{b}$$

## 状态空间表示

**动态**（模态坐标）：
```
state = [q1, q2, ..., qN, q̇1, q̇2, ..., q̇N]
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `E` | 杨氏模量 (Pa) | `2.0e11` (钢) |
| `h` | 板厚 (m) | `0.01` |
| `nu` | 泊松比 | `0.3` |
| `rho` | 密度 (kg/m³) | `7850.0` (钢) |
| `a`, `b` | 板长/宽 (m) | `1.0, 1.0` |
| `q` | 均布载荷 (Pa) | `1000.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：参数验证 + Navier 静态解 + 固有频率 + 模态形状 + 模态动力学 + 能量 + FD 矩阵 |
| `scipy_solve.py` | Navier 静态求解 + 模态动力学 + FD 固有频率 + 正交性验证 |
| `test_MEC051_consistency.py` | Navier 边界 + 挠度公式 + 频率 + 模态正交 + ω²=k/m + 能量 + FD 收敛 + 反例 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC051_consistency.py
```

## 依赖

- numpy
- scipy
