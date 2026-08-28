"""MEC-021-rigid-rotation —— 模型定义（引擎无关）

定轴转动（rigid body rotation about a fixed axis）：刚体绕固定轴转动，
核心方程为 I·θ̈ = τ，其中 I 为转动惯量，τ 为外力矩。
是 020 号段刚体动力学的第二个模型，引入转动惯量和角动量概念。

状态向量 state = [theta, omega]
    theta —— 角位移（rad，相对于平衡位置）
    omega —— 角速度（rad/s）

参数：
    I —— 转动惯量（moment of inertia，kg·m²，I > 0）
    tau —— 外力矩（external torque，N·m）

动力学（一阶常微分方程）：

    刚体定轴转动方程：I·α = τ（α = θ̈ 为角加速度）

    状态空间形式：
        dθ/dt = ω
        dω/dt = τ / I

解析解（恒力矩 τ = const）：

    θ(t) = θ₀ + ω₀·t + ½·(τ/I)·t²
    ω(t) = ω₀ + (τ/I)·t

退化关系（逐项验证）：

    1. τ = 0（无力矩）→ 匀速转动（角动量守恒）
       ω = ω₀ = const，θ = θ₀ + ω₀·t
       对应 MEC-001 自由质点的转动版本

    2. τ = const（恒力矩）→ 匀角加速转动
       α = τ/I = const
       对应 MEC-002 受力质点的转动版本

    3. τ = -κ·θ（线性弹性恢复力矩）→ 角向简谐振动
       θ̈ + (κ/I)·θ = 0，令 ω₀ = √(κ/I)
       对应 MEC-010 简谐振子（κ/I ↔ k/m）

    4. τ = -mgL·sin(θ)（重力恢复力矩，复摆）→ 非线性摆动
       θ̈ + (mgL/I)·sin(θ) = 0
       当 I = mL² 时精确退化为 MEC-015 单摆方程
       MEC-015 是 MEC-021 在特定力矩下的特例

守恒量：

    角动量（绕转轴）：L = I·ω
    无力矩时角动量守恒：L = const

    机械能（保守力矩时）：
        E = ½·I·ω² + V(θ)
        其中 V(θ) 为力矩对应的势能（V = -∫τ dθ）

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(I=1.0):
    """验证基本物理参数合法性。

    参数
    ----
    I : float
        转动惯量（必须 > 0）。

    返回
    ----
    None
        条件满足时静默返回。

    抛出
    ----
    AssertionError
        参数不合法时给出明确错误信息。
    """
    assert I > 0, f"转动惯量 I 必须为正，当前 I={I}"


def angular_momentum(state, I=1.0):
    """计算角动量 L = I·ω。

    参数
    ----
    state : array_like, shape (2,)
        当前状态 [theta, omega]。
    I : float
        转动惯量。

    返回
    ----
    float
        角动量 L。
    """
    _, omega = state
    return I * omega


def dynamics(t, state, I=1.0, tau=0.0):
    """返回状态的时间导数 d(state)/dt = [dθ/dt, dω/dt]。

    刚体定轴转动方程：I·α = τ

    参数
    ----
    t : float
        当前时刻（恒力矩不依赖 t，保留参数以统一接口）。
    state : array_like, shape (2,)
        当前状态 [theta, omega]。
    I : float, optional
        转动惯量（默认 1.0 kg·m²）。
    tau : float, optional
        外力矩（默认 0.0 N·m，即无力矩）。

    返回
    ----
    np.ndarray, shape (2,)
        [omega, tau/I]
    """
    _, omega = state
    return np.array([omega, tau / I])


def analytical(t, initial_state, I=1.0, tau=0.0):
    """恒力矩定轴转动解析解。

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    initial_state : array_like, shape (2,)
        初始状态 [theta0, omega0]。
    I : float, optional
        转动惯量（默认 1.0 kg·m²）。
    tau : float, optional
        恒外力矩（默认 0.0 N·m，即无力矩，退化为匀速转动）。

    返回
    ----
    (theta, omega) : tuple
        角位移和角速度，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    theta0, omega0 = initial_state
    alpha = tau / I
    theta = theta0 + omega0 * t + 0.5 * alpha * t ** 2
    omega = omega0 + alpha * t
    return theta, omega
