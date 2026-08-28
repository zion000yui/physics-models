# MEC-050 Beam（欧拉-伯努利梁）

欧拉-伯努利梁弯曲理论：连续体力学的一维基础模型。梁的横向位移 w(x,t) 满足
4 阶偏微分方程（Euler-Bernoulli 方程）。

## 物理背景

细长梁（长径比 L/h > 10）的弯曲变形，基于平截面假定（Bernoulli-Euler 假设）：
变形前垂直于中性轴的截面变形后仍垂直于中性轴且保持平面，忽略剪切变形。

### 控制方程

**静态**：
$$EI \frac{\partial^4 w}{\partial x^4} = q(x)$$

**动态**：
$$EI \frac{\partial^4 w}{\partial x^4} + \rho A \frac{\partial^2 w}{\partial t^2} = q(x,t)$$

### 边界条件

| 支承类型 | x=0 | x=L |
|----------|-----|-----|
| 悬臂（fixed-free） | w=0, w'=0 | M=0, V=0 |
| 简支（pinned-pinned） | w=0, M=0 | w=0, M=0 |

其中 M = -EI w'' 为弯矩，V = -EI w''' 为剪力。

## 与已有 MEC 模型的关系

| 关系 | 说明 | 严格性 |
|------|------|--------|
| MEC-014 耦合振子 → 连续极限 | N 个离散振子 → 梁的连续振动 | ✓ 物理极限 |
| MEC-050 是 051/053 的基础 | 一维梁 → 二维板 → 三维弹性体 | ✓ 维度递进 |
| 静态 = 4 阶 ODE | 从 PDE 退化为常微分方程 | ✓ 极限退化 |

## 数学模型

### 静态解析解（均布载荷 q）

**悬臂梁**：
$$w(x) = \frac{q\,x^2(6L^2 - 4Lx + x^2)}{24\,EI}, \quad w_{\max} = \frac{qL^4}{8EI}$$

**简支梁**：
$$w(x) = \frac{q\,x(L^3 - 2Lx^2 + x^3)}{24\,EI}, \quad w_{\max} = \frac{5qL^4}{384EI}$$

### 固有频率

$$\omega_n = (\beta_n L)^2 \sqrt{\frac{EI}{\rho A L^4}}$$

| 支承 | 频率方程 | 特征值 β_n L |
|------|---------|-------------|
| 悬臂 | cos(βL)cosh(βL) = -1 | 1.875, 4.694, 7.855, ... |
| 简支 | sin(βL) = 0 | nπ |

### 模态形状

**悬臂**：
$$\varphi_n(x) = \cosh(\beta_n x) - \cos(\beta_n x) - \sigma_n[\sinh(\beta_n x) - \sin(\beta_n x)]$$

$$\sigma_n = \frac{\cos(\beta_n L) + \cosh(\beta_n L)}{\sin(\beta_n L) + \sinh(\beta_n L)}$$

**简支**：
$$\varphi_n(x) = \sin\!\left(\frac{n\pi x}{L}\right)$$

### 模态动力学

分离变量 w(x,t) = Σ φ_n(x) q_n(t)，得：
$$\ddot{q}_n + \omega_n^2\, q_n = F_n(t)$$

能量：$E = \frac{1}{2}\sum_n (\dot{q}_n^2 + \omega_n^2 q_n^2)$（质量归一化）

## 状态空间表示

**静态**：w(x) — 挠度曲线（连续函数）

**动态**（模态坐标）：
```
state = [q1, q2, ..., qN, q̇1, q̇2, ..., q̇N]
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `E` | 杨氏模量 (Pa) | `2.0e11` (钢) |
| `I` | 截面惯性矩 (m⁴) | `1.0e-8` (1cm 方截面) |
| `rho` | 密度 (kg/m³) | `7850.0` (钢) |
| `A` | 截面积 (m²) | `1.0e-4` |
| `L` | 梁长 (m) | `1.0` |
| `q` | 均布载荷 (N/m) | `100.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：参数验证 + 静态解析解（悬臂/简支）+ 固有频率 + 模态形状 + 模态动力学 + 能量 + FD 矩阵 |
| `scipy_solve.py` | BVP 静态求解 + 模态动力学积分 + FD 固有频率 + 正交性验证 |
| `test_MEC050_consistency.py` | 静态 BVP + 边界条件 + 频率公式 + 模态正交 + ω²=k/m + 能量守恒 + FD 收敛 + 反例 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC050_consistency.py
```

## 依赖

- numpy
- scipy
