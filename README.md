# physics-models

Structured physics model library, with a focus on mechanics.

## 项目简介

`physics-models` 是一个计算物理模型库，用于建立可复用、可验证的物理模型。每个模型都包含统一的数学描述、数值求解器、解析解对照和自动化测试，确保数值结果的正确性与可复现性。

## 项目结构

```
physics-models/
├── models/
│   └── mechanics/
│       ├── MEC-001-free-particle
│       ├── MEC-002-forced-particle
│       ├── ...
│       ├── MEC-100-elastoplasticity
│       └── README.md
├── engines/
│   ├── scipy/
│   ├── pytorch/
│   ├── mujoco/
│   ├── modelica/
│   └── mjx/
├── examples/
├── numerical-methods/
├── docs/
├── conftest.py          # pytest 模块隔离
├── run_all_tests.py     # 统一测试入口
├── .github/workflows/   # CI 配置
└── README.md
```

## 模型列表（共 38 个）

### 000 号段 — 质点力学（MEC-001 ~ MEC-009）

| 编号 | 名称 | 核心内容 |
|------|------|---------|
| MEC-001 | free-particle | 无外力匀速直线运动，动量守恒 |
| MEC-002 | forced-particle | 恒力匀加速运动 |
| MEC-003 | projectile | 二维抛体轨迹 |
| MEC-004 | uniform-circular-motion | 匀速圆周运动，向心加速度 |
| MEC-005 | nonuniform-circular-motion | 切向+法向加速度耦合 |
| MEC-006 | central-force-hooke | 中心力 F∝-r，椭圆轨道 |
| MEC-007 | central-force-gravity | 万有引力，开普勒轨道 |
| MEC-008 | two-body-problem | 约化质量，二体→等效单体 |
| MEC-009 | particle-with-drag | 线性/二次阻力 |

### 010 号段 — 振动系统（MEC-010 ~ MEC-015）

| 编号 | 名称 | 核心内容 |
|------|------|---------|
| MEC-010 | mass-spring | 简谐振动，ω=√(k/m) |
| MEC-011 | damped-oscillator | 欠/临界/过阻尼 |
| MEC-012 | forced-oscillator | 受迫振动，共振 |
| MEC-013 | double-pendulum | 非线性耦合，混沌 |
| MEC-014 | coupled-oscillators | 简正模态 |
| MEC-015 | nonlinear-pendulum | 大角度单摆，椭圆积分周期 |

### 020 号段 — 刚体动力学（MEC-020 ~ MEC-024）

| 编号 | 名称 | 核心内容 |
|------|------|---------|
| MEC-020 | rigid-translation | 质心运动定理 |
| MEC-021 | rigid-rotation | 定轴转动，转动惯量 |
| MEC-022 | planar-rigid-body | 平面刚体 3DOF |
| MEC-023 | gyroscopic-precession | 陀螺慢进动近似 |
| MEC-024 | rolling-without-slipping | 纯滚动约束 |

### 030 号段 — 机构运动学（MEC-030 ~ MEC-033）

| 编号 | 名称 | 核心内容 |
|------|------|---------|
| MEC-030 | four-bar-linkage | 四连杆机构 |
| MEC-031 | slider-crank | 曲柄滑块机构 |
| MEC-032 | gear | 齿轮传动比 |
| MEC-033 | cam-follower | 凸轮从动件 |

### 040 号段 — 接触与碰撞（MEC-040 ~ MEC-043）

| 编号 | 名称 | 核心内容 |
|------|------|---------|
| MEC-040 | rigid-contact | 法向接触约束，惩罚法 |
| MEC-041 | coulomb-friction | 库仑摩擦，静/动摩擦 |
| MEC-042 | collision | 碰撞恢复系数 |
| MEC-043 | rolling-friction | 滚动摩擦耗散 |

### 050 号段 — 连续体力学（MEC-050 ~ MEC-053）

| 编号 | 名称 | 核心内容 |
|------|------|---------|
| MEC-050 | beam | 欧拉-伯努利梁，静态弯曲+模态振动 |
| MEC-051 | plate | Kirchhoff-Love 薄板，Navier 解 |
| MEC-052 | shell | 圆柱壳，薄膜+弯曲理论 |
| MEC-053 | 3d-elastic-body | 广义胡克定律，弹性波速 |

### 060 号段 — 分析力学（MEC-060 ~ MEC-062）

| 编号 | 名称 | 核心内容 |
|------|------|---------|
| MEC-060 | lagrangian-formulation | 拉格朗日方程重新求解已有模型 |
| MEC-061 | hamiltonian-formulation | 哈密顿正则方程，相空间 |
| MEC-062 | constrained-generalized-coords | 广义坐标与约束系统 |

### 高级力学

| 编号 | 名称 | 核心内容 |
|------|------|---------|
| MEC-080 | multibody-dynamics | N 连杆平面链，通用多体框架 |
| MEC-090 | nonlinear-mechanics | 分岔、混沌、庞加莱截面 |
| MEC-100 | elastoplasticity | 屈服准则，弹塑性本构 |

## 验证体系

每个力学模型均包含：

- **model.py** — 引擎无关的物理定义：参数、方程、解析解、能量
- **scipy_solve.py** — 基于 SciPy 的数值求解与解析对照
- **test_MECxxx_consistency.py** — 自动化一致性测试
- **README.md** — 物理定义、公式、运行说明

## 运行测试

```bash
# 统一测试入口（自动发现全部 MEC 模型）
python run_all_tests.py

# 使用 pytest（conftest.py 隔离模块导入）
python -m pytest models/mechanics/ --tb=short -q
```

## 依赖

- numpy
- scipy
- pytest（可选，用于 pytest 模式）
