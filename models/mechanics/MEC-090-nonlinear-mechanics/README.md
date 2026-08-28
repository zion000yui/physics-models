# MEC-090 Nonlinear Mechanics（非线性力学）

分岔、混沌、庞加莱截面。复用 MEC-013 双摆和 MEC-015 非线性单摆的数据。

## 物理背景

### 非线性单摆周期-振幅关系

$$T = 4\sqrt{\frac{l}{g}} K\!\left(\sin^2\frac{\theta_0}{2}\right)$$

其中 $K$ 为第一类完全椭圆积分。小角度时 $T \to T_0 = 2\pi\sqrt{l/g}$（线性极限），
$\theta_0 \to \pi$ 时 $T \to \infty$（同宿轨道）。

### 受驱阻尼摆

$$\ddot\theta + c\dot\theta + \frac{g}{l}\sin\theta = A\cos(\omega_d t)$$

非自治系统，驱动参数变化可导致分岔和混沌。

### 非线性特征量

- **庞加莱截面**：在驱动周期 $T_d = 2\pi/\omega_d$ 处采样
- **Lyapunov 指数**：衡量轨迹发散率（λ > 0 → 混沌）
- **双摆发散**：大角度初始条件下轨迹指数发散

### 旋转阈值

$$E_c = 2mgl$$

$E < E_c$：振荡运动；$E > E_c$：旋转运动。

## 与已有 MEC 模型的关系

| 关系 | 说明 |
|------|------|
| MEC-015 非线性单摆 → MEC-090 | 周期-振幅关系 |
| MEC-013 双摆 → MEC-090 | 混沌行为 |
| MEC-060 拉格朗日 → MEC-090 | 方程推导 |
| MEC-080 多体 → MEC-090 | 多体混沌 |

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | 引擎无关：单摆周期(椭圆积分) + 受驱摆 + 庞加莱截面 + Lyapunov + 双摆发散 |
| `scipy_solve.py` | 周期-振幅表 + 受驱摆 + 庞加莱 + Lyapunov + 双摆发散 |
| `test_MEC090_consistency.py` | 周期关系 + 能量 + 旋转/振荡 + 庞加莱 + 混沌 + 反例 |
| `README.md` | 物理定义、公式、运行说明 |

## 运行

```bash
python scipy_solve.py
python test_MEC090_consistency.py
```

## 依赖

- numpy
- scipy
