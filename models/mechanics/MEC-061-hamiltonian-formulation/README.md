# MEC-061 Hamiltonian Formulation（哈密顿力学公式化）

通过 Legendre 变换从拉格朗日力学导出哈密顿正则方程，在相空间 (q, p) 中描述系统。

## 物理背景

哈密顿力学是分析力学的第二种等价表述。通过共轭动量 $p = \partial L / \partial \dot{q}$ 和
Legendre 变换 $H = \sum p\dot{q} - L$ 将二阶 ODE 转化为一阶系统。

### 正则方程

$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$

保守系统：$H = T + V = E$（总能量守恒）

### 重新求解的已有模型

| 模型 | H | 正则方程 |
|------|---|---------|
| MEC-001 自由质点 | p²/2m | q̇=p/m, ṗ=0 |
| MEC-002 受力质点 | p²/2m+Fx | q̇=p/m, ṗ=-F |
| MEC-010 弹簧振子 | p²/2m+½kx² | q̇=p/m, ṗ=-kx |
| MEC-006 胡克力2D | (px²+py²)/2m+½k(x²+y²) | 各方向独立 |
| MEC-011 阻尼振子 | p²/2m+½kx² | ṗ=-kx-cp/m |

### 核心性质

- **H 守恒**：保守系统 H = E 守恒
- **Liouville 定理**：相空间体积守恒（哈密顿流不可压缩）
- **泊松括号**：$\{q, p\} = 1$（基本对易关系）
- **辛结构**：哈密顿流保辛结构

## 与已有 MEC 模型的关系

| 关系 | 说明 |
|------|------|
| MEC-060 → MEC-061 | Legendre 变换 |
| MEC-061 → MEC-062 | 哈密顿约束处理 |
| MEC-061 → MEC-090 | 相空间描述 → 非线性/混沌 |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：5 个模型的哈密顿量 + 正则方程 + 泊松括号 + 相空间面积 |
| `scipy_solve.py` | 数值积分正则方程 + 能量验证 + Liouville 定理 + 泊松括号 |
| `test_MEC061_consistency.py` | 解析 vs 数值 + H 守恒/耗散 + Legendre + Liouville + 反例 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC061_consistency.py
```

## 依赖

- numpy
- scipy
