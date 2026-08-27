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
│       └── MEC-003-projectile
├── engines/
│   ├── scipy/
│   ├── pytorch/
│   ├── mujoco/
│   ├── modelica/
│   └── mjx/
├── examples/
├── numerical-methods/
└── docs/
```

## Phase 1: Newtonian Particle Mechanics

第一阶段目标：建立牛顿质点力学基础模型，验证「模型定义 → 引擎求解 → 解析解对照」的完整闭环。

### 模型列表

#### MEC-001 Free Particle
自由质点：不受外力、无约束的点质量，在惯性系中保持匀速直线运动。

#### MEC-002 Forced Particle
受力质点：受恒定外力作用的质点，加速度恒为 `F/m`。

#### MEC-003 Projectile Motion
抛体运动：重力场中质点的二维运动，水平速度守恒，竖直方向受恒定重力加速度。

## 验证体系

每个力学模型均包含：

- **Mathematical formulation** — 状态空间与微分方程组
- **Numerical solver** — 基于 SciPy `solve_ivp` 的数值积分
- **Analytical solution** — 解析解作为正确性金标准
- **Automated tests** — pytest 自动化测试，持续集成于 GitHub Actions

## Roadmap

- Oscillation Systems
- Rigid Body Dynamics
- Multibody Systems
- Robotics Simulation
