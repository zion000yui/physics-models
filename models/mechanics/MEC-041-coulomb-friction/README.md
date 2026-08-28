# MEC-041 Coulomb Friction（库仑摩擦）

质点在水平面上受外力作用，与地面之间产生库仑摩擦力。静摩擦（物体静止）
和动摩擦（物体滑动）自动切换。

## 物理背景

库仑摩擦模型是最经典的干摩擦模型。当外力小于最大静摩擦力时，摩擦力完全
平衡外力，物体保持静止。当外力超过最大静摩擦力时，物体开始滑动，摩擦力
为常数 μ_k·N，方向反对运动方向。

**假设**：
- 水平面，法向力 N = m·g
- 静摩擦系数 μ_s ≥ 动摩擦系数 μ_k ≥ 0
- 外力 F_ext 沿 x 方向

## 与已有 MEC 模型的关系

| 关系 | 说明 | 严格性 |
|------|------|--------|
| μ_s=μ_k=0 → MEC-002 | 无摩擦，受力质点 | ✓ 严格退化 |
| 无外力+有初速 → 滑动减速至停止 | 摩擦力做负功 | 特例 |
| 静摩擦是约束力 | MEC-024 纯滚动静摩擦也是约束力 | ✓ 概念相似 |
| 法向力来自 MEC-040 接触 | N = m·g | ✓ 概念复用 |

## 数学模型

### 摩擦力

$$F_f = \begin{cases} -F_{\text{ext}} & |v| < \epsilon \text{ 且 } |F_{\text{ext}}| \leq \mu_s N \\ -\mu_k N \cdot \text{sign}(v) & |v| \geq \epsilon \\ -\mu_k N \cdot \text{sign}(F_{\text{ext}}) & |v| < \epsilon \text{ 且 } |F_{\text{ext}}| > \mu_s N \end{cases}$$

### 运动方程

$$m\,\ddot{x} = F_{\text{ext}} + F_f$$

### 动摩擦解析解（恒定外力）

$$a = \frac{F_{\text{ext}} - \mu_k N \cdot \text{sign}(F_{\text{ext}})}{m}$$

$$v(t) = v_0 + a\,t, \quad x(t) = v_0\,t + \frac{1}{2}a\,t^2$$

### 自由减速

$$v(t) = v_0 - \mu_k g \cdot t, \quad t_{\text{stop}} = \frac{v_0}{\mu_k g}$$

## 状态空间表示

```
state = [x, v]
```

- `x` — 位置（m）
- `v` — 速度（m/s）

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `m` | 质量 (kg) | `1.0` |
| `g` | 重力加速度 (m/s²) | `9.81` |
| `mu_s` | 静摩擦系数 | `0.3` |
| `mu_k` | 动摩擦系数 | `0.25` |
| `F_ext` | 外力 (N) | `5.0` |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：摩擦力计算 + 静/动切换 + 动力学 + 能量 + 解析解 |
| `scipy_solve.py` | 用 SciPy 数值积分（动摩擦/静摩擦/自由减速三种场景） |
| `test_MEC041_consistency.py` | 解析解 + 静摩擦 + 自由减速 + 摩擦方向 + 能量平衡 + 退化 + 切换 + 反例 + 参数验证 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC041_consistency.py
```

## 依赖

- numpy
- scipy
