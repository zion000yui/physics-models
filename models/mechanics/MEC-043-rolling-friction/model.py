"""MEC-043-rolling-friction — 模型定义（引擎无关）

滚动摩擦（rolling friction）：圆形刚体在水平面上滚动，受到滚动阻力矩。
与 MEC-024 纯滚动对照，此处引入耗散——滚动摩擦力矩做负功，动能单调递减。

=== 物理系统 ===

  圆形刚体（球/圆柱），质量 m，半径 R，转动惯量 I
  在水平面上纯滚动（v = R·ω 约束保持）
  滚动摩擦力矩 τ_r = -μ_r·N·R·sign(ω) = -μ_r·m·g·R·sign(ω)
  其中 μ_r 为滚动摩擦系数（无量纲），N = m·g

  滚动摩擦力（等效平动）F_r = τ_r/R = -μ_r·m·g·sign(v)

=== 运动方程 ===

  平动：m·ẍ = -μ_r·m·g·sign(v)  (滚动阻力)
  转动：I·α = τ_r = -μ_r·m·g·R·sign(ω)
  约束：v = R·ω（纯滚动）

  联立（消去约束）：
    m·a = -μ_r·m·g·sign(v)
    a = -μ_r·g·sign(v)

  减速度大小为 μ_r·g（常数），方向反对运动方向。

=== 解析解 ===

  v(t) = v0 - μ_r·g·sign(v0)·t  (直至 v=0)
  x(t) = v0·t - ½·μ_r·g·sign(v0)·t²
  停止时间: t_stop = |v0| / (μ_r·g)
  停止距离: x_stop = v0² / (2·μ_r·g)

=== 能量 ===

  动能 T = ½·m·v² + ½·I·ω² = ½·(m + I/R²)·v² = ½·m_eff·v²
  其中 m_eff = m + I/R²（有效质量，同 MEC-024）

  滚动摩擦耗散功率: P = μ_r·m·g·|v|
  dE/dt = -μ_r·m·g·|v| < 0（单调递减）

=== 与已有 MEC 模型的关系 ===

  μ_r → 0 时退化为 MEC-024（纯滚动无耗散）
  有效质量 m_eff = m + I/R² 同 MEC-024
  滚动阻力类比 MEC-041 动摩擦（都是 -μ·N·sign(v)）

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(m=1.0, R=0.5, I=None, g=9.81, mu_r=0.01):
    """验证物理参数合法性。"""
    assert m > 0, f"质量 m 必须为正，当前 m={m}"
    assert R > 0, f"半径 R 必须为正，当前 R={R}"
    if I is not None:
        assert I >= 0, f"转动惯量 I 必须非负，当前 I={I}"
    assert g > 0, f"重力加速度 g 必须为正，当前 g={g}"
    assert mu_r >= 0, f"滚动摩擦系数 mu_r 必须非负，当前 mu_r={mu_r}"


def effective_mass(m, I, R):
    """有效质量 m_eff = m + I/R²（同 MEC-024）。"""
    if I is None:
        I = 0.4 * m * R**2  # 默认实心球
    return m + I / R**2


def rolling_friction_force(v, m, g, mu_r):
    """滚动摩擦等效平动力 F_r = -μ_r·m·g·sign(v)。"""
    if abs(v) < 1e-10:
        return 0.0  # 停止后无摩擦力
    return -mu_r * m * g * np.sign(v)


def rolling_friction_torque(omega, m, g, R, mu_r):
    """滚动摩擦力矩 τ_r = -μ_r·m·g·R·sign(ω)。"""
    if abs(omega) < 1e-10:
        return 0.0
    return -mu_r * m * g * R * np.sign(omega)


def mechanical_energy(state, m, R, I=None):
    """动能 E = ½·m_eff·v²（水平面无势能）。"""
    x, v = state
    m_eff = effective_mass(m, I, R)
    return 0.5 * m_eff * v**2


def dynamics(t, state, m=1.0, R=0.5, I=None, g=9.81, mu_r=0.01):
    """返回状态时间导数 [dx/dt, dv/dt]。

    a = -μ_r·g·sign(v)（纯滚动约束代入后）
    """
    _, v = state
    if I is None:
        I = 0.4 * m * R**2

    F_r = rolling_friction_force(v, m, g, mu_r)
    m_eff = effective_mass(m, I, R)
    a = F_r / m_eff  # = -μ_r·m·g/(m+I/R²) · sign(v)

    return np.array([v, a])


def analytical(t, v0, m=1.0, R=0.5, I=None, g=9.81, mu_r=0.01):
    """解析解（恒定减速度直至停止）。

    v(t) = v0 - μ_r·g·(m/m_eff)·sign(v0)·t
    """
    t = np.asarray(t, dtype=float)
    if I is None:
        I = 0.4 * m * R**2

    m_eff = effective_mass(m, I, R)
    a_decel = mu_r * m * g / m_eff  # 减速度大小

    t_stop = abs(v0) / a_decel if a_decel > 0 else float('inf')

    sign_v0 = np.sign(v0)
    v = v0 - a_decel * sign_v0 * t
    v = np.where(t < t_stop, v, 0.0)  # 停止后 v=0

    x = np.where(t < t_stop,
                 v0 * t - 0.5 * a_decel * sign_v0 * t**2,
                 v0 * t_stop - 0.5 * a_decel * sign_v0 * t_stop**2)

    return x, v
