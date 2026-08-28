"""MEC-032-gear — 模型定义（引擎无关）

齿轮传动（gear transmission）：一对啮合的外齿轮在平行轴间传递转动。
轮齿在节点处保持无滑动滚动接触，产生恒定传动比。

这是 030 号段中唯一具有闭式动力学解析解的模型（等效惯量为常数）。

=== 坐标系 ===

  齿轮 1（输入）绕 O₁ 旋转，角度 θ₁，角速度 ω₁
  齿轮 2（输出）绕 O₂ 旋转，角度 θ₂，角速度 ω₂
  外啮合：ω₂ = -(r₁/r₂)·ω₁（反向旋转）

=== 运动学 ===

  传动比 i = r₁/r₂ = z₁/z₂
  θ₂ = -i·θ₁, ω₂ = -i·ω₁, α₂ = -i·α₁

=== 动力学（1-DOF 等效惯量法）===

  等效惯量：I_eq = I₁ + i²·I₂ = const（不依赖 θ₁）
  运动方程：I_eq·α₁ = τ_in - i·τ_load
  解析解（闭式）：
    α₁ = (τ_in - i·τ_load) / I_eq = const
    ω₁(t) = ω₀ + α₁·t
    θ₁(t) = θ₀ + ω₀·t + ½·α₁·t²

=== 接触力 ===

  从齿轮 1：F = (τ_in - I₁·α₁) / r₁
  从齿轮 2：F = (τ_load + i·I₂·α₁) / r₂
  两者应一致（Newton 第三定律 + 约束）

=== 功率 ===

  P_in  = τ_in·ω₁
  P_out = τ_load·i·ω₁（输出到负载的功率）
  功率平衡：P_in = P_out + dT/dt（dT/dt = I_eq·α₁·ω₁）

=== 与已有 MEC 模型的关系 ===

  每个齿轮是 MEC-021 类型定轴转动：概念复用
  滚动接触约束类比 MEC-024 纯滚动：概念相似
  等效惯量 I₁+i²I₂ 类比 MEC-024 有效质量 m+I/R²：数学结构相似
  I₂→0 且 τ_load→0 时退化为 MEC-021（单刚体定轴转动）：严格退化

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(r1=0.1, r2=0.2, I1=0.01, I2=0.04,
                        tau_in=0.0, tau_load=0.0):
    """验证物理参数合法性。"""
    assert r1 > 0, f"节圆半径 r1 必须为正，当前 r1={r1}"
    assert r2 > 0, f"节圆半径 r2 必须为正，当前 r2={r2}"
    assert I1 >= 0, f"转动惯量 I1 必须非负，当前 I1={I1}"
    assert I2 >= 0, f"转动惯量 I2 必须非负，当前 I2={I2}"


def transmission_ratio(r1, r2):
    """计算传动比 i = r₁/r₂ = z₁/z₂。"""
    return r1 / r2


def equivalent_inertia(I1, I2, r1, r2):
    """计算等效到输入轴的转动惯量 I_eq = I₁ + (r₁/r₂)²·I₂。

    这是常数（不依赖 θ₁），因此动力学有闭式解析解。
    """
    i = transmission_ratio(r1, r2)
    return I1 + i**2 * I2


def output_kinematics(theta1, omega1, alpha1, r1, r2):
    """由输入齿轮状态计算输出齿轮状态。

    外啮合：θ₂ = -i·θ₁, ω₂ = -i·ω₁, α₂ = -i·α₁
    """
    i = transmission_ratio(r1, r2)
    theta2 = -i * theta1
    omega2 = -i * omega1
    alpha2 = -i * alpha1
    return theta2, omega2, alpha2


def contact_force(tau_in, I1, alpha1, r1):
    """计算节点接触力 F = (τ_in - I₁·α₁) / r₁（从齿轮 1）。"""
    return (tau_in - I1 * alpha1) / r1


def contact_force_from_output(tau_load, I2, alpha1, r1, r2):
    """计算节点接触力 F = (τ_load + i·I₂·α₁) / r₂（从齿轮 2）。

    应与 contact_force() 给出相同结果。
    """
    i = transmission_ratio(r1, r2)
    return (tau_load + i * I2 * alpha1) / r2


def power_flow(tau_in, omega1, tau_load, r1, r2):
    """计算输入和输出功率。

    P_in  = τ_in·ω₁
    P_out = τ_load·i·ω₁（输出到负载的功率）
    """
    i = transmission_ratio(r1, r2)
    P_in = tau_in * omega1
    P_out = tau_load * i * omega1
    return P_in, P_out


def mechanical_energy(state, r1, r2, I1, I2):
    """计算总机械能 E = ½·I_eq·ω₁²。

    齿轮质心在转轴上（平衡齿轮），无重力势能。
    """
    _, omega1 = state
    I_eq = equivalent_inertia(I1, I2, r1, r2)
    return 0.5 * I_eq * omega1**2


def dynamics(t, state, r1, r2, I1, I2, tau_in=0.0, tau_load=0.0):
    """返回状态时间导数 [dθ₁/dt, dω₁/dt]。

    运动方程：I_eq·α₁ = τ_in - i·τ_load
    由于 I_eq = const，α₁ = const。
    """
    _, omega1 = state
    i = transmission_ratio(r1, r2)
    I_eq = equivalent_inertia(I1, I2, r1, r2)
    alpha1 = (tau_in - i * tau_load) / I_eq
    return np.array([omega1, alpha1])


def analytical(t, initial_state, r1, r2, I1, I2, tau_in=0.0, tau_load=0.0):
    """闭式解析解（常加速度运动）。

    α₁ = (τ_in - i·τ_load) / I_eq = const
    ω₁(t) = ω₀ + α₁·t
    θ₁(t) = θ₀ + ω₀·t + ½·α₁·t²
    """
    t = np.asarray(t, dtype=float)
    theta0, omega0 = initial_state
    i = transmission_ratio(r1, r2)
    I_eq = equivalent_inertia(I1, I2, r1, r2)
    alpha = (tau_in - i * tau_load) / I_eq

    theta = theta0 + omega0 * t + 0.5 * alpha * t**2
    omega = omega0 + alpha * t
    return theta, omega
