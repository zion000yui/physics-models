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

#### MEC-004 Uniform Circular Motion
匀速圆周运动：质点以恒定速率沿圆周运动，向心加速度 a_c = Rω²。

#### MEC-005 Nonuniform Circular Motion
非匀速圆周运动：角速度随时间线性变化，引入切向加速度。

#### MEC-006 Central Force Hooke
胡克型中心力场：质点在 F = -k·r 作用下运动，轨迹为椭圆（力心在中心）。

#### MEC-007 Central Force Gravity
万有引力中心力场（开普勒问题）：质点在平方反比引力 F = -μm/r²·r̂ 作用下运动，束缚态轨迹为以力心为焦点的椭圆。与 MEC-006 共同验证 Bertrand 定理。

#### MEC-008 Two-Body Problem
二体问题：两个质点在互相引力作用下运动，通过约化质量和相对坐标化简为等效单体开普勒问题，是 MEC-007 的推广。

#### MEC-009 Particle with Drag
速度相关阻力下的质点运动：覆盖线性阻力（Stokes）和二次阻力（Newton），无阻力极限退化为 MEC-003 抛体运动。

#### MEC-010 Mass-Spring
标准简谐振子：质量-弹簧系统，固有角频率 ω₀=√(k/m)，建立相图规范，为后续振动模型基础。

#### MEC-011 Damped Oscillator
阻尼振子：覆盖欠阻尼/临界阻尼/过阻尼三种状态，阻尼比判据，无阻尼极限退化为 MEC-010。

#### MEC-012 Forced Oscillator
受迫振子：瞬态+稳态分解，幅频响应，共振频率，无外力退化为 MEC-011，无阻尼无外力退化为 MEC-010。

#### MEC-014 Coupled Oscillators
耦合振子：两个质量块通过弹簧耦合，简正模态分解，无耦合退化为两个独立 MEC-010。

#### MEC-015 Nonlinear Pendulum
非线性单摆：完整保留 sin(θ) 非线性，周期随振幅增长（椭圆积分），小角度退化为 MEC-010。

#### MEC-013 Double Pendulum
双摆：非线性耦合混沌系统，对初始条件敏感，小角度退化为 MEC-014 耦合振子，010 号段最终模型。

#### MEC-020 Rigid Translation
刚体平动：质心运动定理 M·a_cm = F_ext，无外力退化为 MEC-001，恒力退化为 MEC-002，重力退化为 MEC-003。

#### MEC-021 Rigid Rotation
定轴转动：核心方程 I·θ̈=τ，无力矩退化为匀速转动，弹性力矩退化为角向 MEC-010，重力力矩退化为 MEC-015。

#### MEC-024 Rolling Without Slipping
纯滚动：约束 v_cm=R·ω 耦合平动+转动，a=g·sin(α)/(1+I/(mR²))，静摩擦不做功，去约束退化为 MEC-020。

#### MEC-022 Planar Rigid Body
平面刚体 3DOF：2D 平动+1D 转动，力矩 τ=r×F 耦合平转，力过质心退化为 MEC-020，无力退化为 MEC-021。

#### MEC-023 Gyroscopic Precession
高速自旋陀螺的慢进动近似模型：Routhian 降维，Ω_p=mgl/(I₃ω_s)，明确区分精确方程与近似稳态解，不是完整 3D Euler top。

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
