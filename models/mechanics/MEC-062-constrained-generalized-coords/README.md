# MEC-062 Constrained Generalized Coordinates（广义坐标与约束系统）

处理完整约束和非完整约束下的广义坐标系统，衔接机构模块和多体动力学。

## 物理背景

### 完整约束 vs 非完整约束

- **完整约束** $f(q, t) = 0$：可消去坐标，减少自由度
- **非完整约束** $f(q, \dot{q}, t) = 0$：不可消去，需用乘子法

### 广义坐标

$N$ 个质点（$3N$ 个笛卡尔坐标），受 $K$ 个完整约束 → $3N-K$ 个广义坐标。

### 拉格朗日乘子法

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = \sum_a \lambda_a \frac{\partial f_a}{\partial q_i}$$

## 本模型实现的约束系统

| 系统 | 约束类型 | 广义坐标 | 核心结果 |
|------|---------|---------|---------|
| 单摆 | 完整 $x^2+y^2=l^2$ | $\theta$ | $\ddot\theta = -(g/l)\sin\theta$ |
| 阿特伍德机 | 完整 $x_1+x_2=l$ | $x$ | $a = (m_1-m_2)g/(m_1+m_2)$ |
| 斜面纯滚动 | 完整 $x=R\varphi$ | $x$ | $a = g\sin\theta/(1+k)$ |

### 约束力计算

- 单摆绳张力：$T = ml\dot\theta^2 + mg\cos\theta$
- 阿特伍德绳张力：$T = 2m_1 m_2 g/(m_1+m_2)$
- 纯滚动摩擦力：$f_s = kmg\sin\theta/(1+k)$

### 纯滚动条件

$$\tan\theta \le \mu_s \frac{1+k}{k}$$

## 与已有 MEC 模型的关系

| 关系 | 说明 |
|------|------|
| MEC-030 机构 → MEC-062 | 完整约束的拉格朗日处理 |
| MEC-024 纯滚动 → MEC-062 | 广义坐标消去约束 |
| MEC-060 拉格朗日 → MEC-062 | 约束推广 |
| MEC-062 → MEC-080 | 通用多体约束框架 |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：单摆 + 阿特伍德 + 纯滚动 + 约束力 + 约束验证 |
| `scipy_solve.py` | 数值积分 + 约束力计算 + 摩擦条件 |
| `test_MEC062_consistency.py` | 能量 + 加速度 + 张力 + 摩擦 + 约束 + 反例 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC062_consistency.py
```

## 依赖

- numpy
- scipy
