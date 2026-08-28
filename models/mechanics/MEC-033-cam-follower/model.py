"""MEC-033-cam-follower — 模型定义（引擎无关）

凸轮从动件机构（cam-follower mechanism）：旋转凸轮通过特定轮廓驱动平移从动件
做预定往复运动。从动件由弹簧保持与凸轮接触。凸轮具有转动惯量，从动件具有
平动质量，系统为 1 自由度（凸轮角 θ）。

实现完整 DRRD 循环（Rise-Dwell-Return-Dwell）和三种标准轮廓。

=== 坐标系 ===

  凸轮绕 O 旋转，角度 θ，角速度 ω
  从动件沿 y 轴平移，位移 y(θ) 由凸轮轮廓确定
  弹簧刚度 k 将从动件压向凸轮

=== DRRD 循环 ===

  Rise:    0 ≤ θ < β_r           y: 0 → h
  Dwell1:  β_r ≤ θ < β_r+β_d1    y = h
  Return:  β_r+β_d1 ≤ θ < β_r+β_d1+β_re   y: h → 0
  Dwell2:  β_r+β_d1+β_re ≤ θ < 2π   y = 0

  β_r + β_d1 + β_re + β_d2 = 2π

=== 三种标准轮廓 ===

  Rise 段 (φ = θ/β_r, 0 ≤ φ ≤ 1):
    SHM:       y = (h/2)(1 - cos πφ)
    Cycloidal: y = h(φ - sin 2πφ / (2π))
    3-4-5:     y = h(10φ³ - 15φ⁴ + 6φ⁵)

  Return 段 (φ = (θ-β_r-β_d1)/β_re):
    SHM:       y = (h/2)(1 + cos πφ)
    Cycloidal: y = h(1 - φ + sin 2πφ / (2π))
    3-4-5:     y = h(1 - 10φ³ + 15φ⁴ - 6φ⁵)

  加速度连续性：
    SHM:       y'' 在段间有有限跳变
    Cycloidal: y'' 在段间连续（=0）
    3-4-5:     y'' 在段间连续（=0）

=== 等效惯量动力学（1-DOF）===

  I_eff(θ) = I_cam + m_f·(dy/dθ)²
  I_eff'(θ) = 2·m_f·(dy/dθ)·(d²y/dθ²)   （解析，无需数值差分）

  运动方程（拉格朗日推导）：
    I_eff·α + ½·I_eff'·ω² + k·y·(dy/dθ) = τ

  动力学 ODE 无闭式解析解（变等效惯量），数值积分 + 能量守恒验证。

=== 接触力（逆动力学）===

  F = m_f·ÿ + k·y
  ÿ = (d²y/dθ²)·ω² + (dy/dθ)·α

  F > 0: 从动件与凸轮保持接触
  F < 0: 从动件跳脱（弹簧力不足）

=== 压力角 ===

  tan(α_p) = |dy/dθ| / (r_b + y)
  α_p 越大，侧向力越大（设计约束，不影响动力学）

=== 与已有 MEC 模型的关系 ===

  凸轮旋转：MEC-021 定轴转动概念
  从动件平移 + 弹簧：MEC-010 简谐振子 + MEC-012 受迫振子概念
  约束驱动运动：MEC-030/031 闭环约束概念
  无严格退化到任何已有 MEC 模型

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(h=0.01, beta_rise=np.pi/2, beta_dwell1=np.pi/4,
                        beta_return=np.pi/2, I_cam=0.001, m_f=0.1,
                        k=100.0, r_b=0.03, tau=0.0):
    """验证物理参数合法性。"""
    assert h > 0, f"升程 h 必须为正，当前 h={h}"
    assert beta_rise > 0, f"升程角 beta_rise 必须为正"
    assert beta_dwell1 >= 0, f"远休止角 beta_dwell1 必须非负"
    assert beta_return > 0, f"回程角 beta_return 必须为正"
    total = beta_rise + beta_dwell1 + beta_return
    assert total < 2 * np.pi + 1e-10, \
        f"前三角之和 {total:.4f} 超过 2π，无空间给近休止"
    assert I_cam >= 0, f"凸轮惯量 I_cam 必须非负"
    assert m_f >= 0, f"从动件质量 m_f 必须非负"
    assert k >= 0, f"弹簧刚度 k 必须非负"
    assert r_b > 0, f"基圆半径 r_b 必须为正"


def _profile_derivatives(theta, h, beta_rise, beta_dwell1, beta_return,
                          profile='cycloidal'):
    """返回 (y, dy/dθ, d²y/dθ²) for DRRD cam profile.

    θ 自动取模 2π。
    """
    two_pi = 2 * np.pi
    tm = theta % two_pi

    b1 = beta_rise
    b2 = b1 + beta_dwell1
    b3 = b2 + beta_return

    if tm < b1:
        # === Rise ===
        phi = tm / b1
        if profile == 'shm':
            y = h / 2 * (1 - np.cos(np.pi * phi))
            yp = h * np.pi / (2 * b1) * np.sin(np.pi * phi)
            ypp = h * np.pi**2 / (2 * b1**2) * np.cos(np.pi * phi)
        elif profile == 'cycloidal':
            y = h * (phi - np.sin(two_pi * phi) / two_pi)
            yp = h * (1 - np.cos(two_pi * phi)) / b1
            ypp = h * two_pi * np.sin(two_pi * phi) / b1**2
        elif profile == 'poly345':
            y = h * (10*phi**3 - 15*phi**4 + 6*phi**5)
            yp = h * (30*phi**2 - 60*phi**3 + 30*phi**4) / b1
            ypp = h * (60*phi - 180*phi**2 + 120*phi**3) / b1**2
        else:
            raise ValueError(f"未知轮廓: {profile}")
        return y, yp, ypp

    elif tm < b2:
        # === Dwell1 (peak) ===
        return h, 0.0, 0.0

    elif tm < b3:
        # === Return ===
        phi = (tm - b2) / beta_return
        if profile == 'shm':
            y = h / 2 * (1 + np.cos(np.pi * phi))
            yp = -h * np.pi / (2 * beta_return) * np.sin(np.pi * phi)
            ypp = -h * np.pi**2 / (2 * beta_return**2) * np.cos(np.pi * phi)
        elif profile == 'cycloidal':
            y = h * (1 - phi + np.sin(two_pi * phi) / two_pi)
            yp = -h * (1 - np.cos(two_pi * phi)) / beta_return
            ypp = -h * two_pi * np.sin(two_pi * phi) / beta_return**2
        elif profile == 'poly345':
            y = h * (1 - 10*phi**3 + 15*phi**4 - 6*phi**5)
            yp = -h * (30*phi**2 - 60*phi**3 + 30*phi**4) / beta_return
            ypp = -h * (60*phi - 180*phi**2 + 120*phi**3) / beta_return**2
        else:
            raise ValueError(f"未知轮廓: {profile}")
        return y, yp, ypp

    else:
        # === Dwell2 (base) ===
        return 0.0, 0.0, 0.0


def follower_displacement(theta, h, beta_rise, beta_dwell1, beta_return,
                           profile='cycloidal'):
    """从动件位移 y(θ)。"""
    y, _, _ = _profile_derivatives(theta, h, beta_rise, beta_dwell1,
                                    beta_return, profile)
    return y


def follower_velocity_ratio(theta, h, beta_rise, beta_dwell1, beta_return,
                             profile='cycloidal'):
    """从动件速度比 dy/dθ。"""
    _, yp, _ = _profile_derivatives(theta, h, beta_rise, beta_dwell1,
                                     beta_return, profile)
    return yp


def follower_acceleration_ratio(theta, h, beta_rise, beta_dwell1, beta_return,
                                 profile='cycloidal'):
    """从动件加速度比 d²y/dθ²。"""
    _, _, ypp = _profile_derivatives(theta, h, beta_rise, beta_dwell1,
                                      beta_return, profile)
    return ypp


def equivalent_inertia(theta, h, beta_rise, beta_dwell1, beta_return,
                       I_cam, m_f, profile='cycloidal'):
    """等效到凸轮轴的转动惯量 I_eff(θ) = I_cam + m_f·(dy/dθ)²。"""
    _, yp, _ = _profile_derivatives(theta, h, beta_rise, beta_dwell1,
                                    beta_return, profile)
    return I_cam + m_f * yp**2


def equivalent_inertia_derivative(theta, h, beta_rise, beta_dwell1, beta_return,
                                   I_cam, m_f, profile='cycloidal'):
    """I_eff'(θ) = 2·m_f·(dy/dθ)·(d²y/dθ²)（解析公式）。"""
    _, yp, ypp = _profile_derivatives(theta, h, beta_rise, beta_dwell1,
                                       beta_return, profile)
    return 2 * m_f * yp * ypp


def contact_force(theta, omega, alpha, h, beta_rise, beta_dwell1, beta_return,
                   m_f, k, profile='cycloidal'):
    """接触力 F = m_f·ÿ + k·y（逆动力学）。

    ÿ = (d²y/dθ²)·ω² + (dy/dθ)·α
    F > 0 表示保持接触。
    """
    y, yp, ypp = _profile_derivatives(theta, h, beta_rise, beta_dwell1,
                                       beta_return, profile)
    y_ddot = ypp * omega**2 + yp * alpha
    return m_f * y_ddot + k * y


def pressure_angle(theta, h, beta_rise, beta_dwell1, beta_return, r_b,
                    profile='cycloidal'):
    """压力角 α_p = arctan(|dy/dθ| / (r_b + y))。"""
    y, yp, _ = _profile_derivatives(theta, h, beta_rise, beta_dwell1,
                                     beta_return, profile)
    return np.arctan(abs(yp) / (r_b + y))


def potential_energy(state, h, beta_rise, beta_dwell1, beta_return, k, profile='cycloidal'):
    """弹簧势能 V = ½·k·y(θ)²。"""
    theta, _ = state
    y = follower_displacement(theta, h, beta_rise, beta_dwell1, beta_return, profile)
    return 0.5 * k * y**2


def mechanical_energy(state, h, beta_rise, beta_dwell1, beta_return,
                      I_cam, m_f, k, profile='cycloidal'):
    """总机械能 E = ½·I_eff·ω² + ½·k·y²。"""
    theta, omega = state
    I_eff = equivalent_inertia(theta, h, beta_rise, beta_dwell1, beta_return,
                               I_cam, m_f, profile)
    V = potential_energy(state, h, beta_rise, beta_dwell1, beta_return, k, profile)
    return 0.5 * I_eff * omega**2 + V


def dynamics(t, state, h, beta_rise, beta_dwell1, beta_return,
             I_cam, m_f, k, tau=0.0, r_b=0.03, profile='cycloidal'):
    """返回状态时间导数 [dθ/dt, dω/dt]。

    运动方程：I_eff·α + ½·I_eff'·ω² + k·y·(dy/dθ) = τ

    I_eff' 用解析公式（不是数值差分）。
    """
    theta, omega = state
    y, yp, ypp = _profile_derivatives(theta, h, beta_rise, beta_dwell1,
                                       beta_return, profile)

    I_eff = I_cam + m_f * yp**2
    I_eff_prime = 2 * m_f * yp * ypp  # 解析
    dV_dtheta = k * y * yp

    alpha = (tau - 0.5 * I_eff_prime * omega**2 - dV_dtheta) / I_eff
    return np.array([omega, alpha])
