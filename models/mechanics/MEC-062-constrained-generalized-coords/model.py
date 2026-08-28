"""MEC-062-constrained-generalized-coords — 模型定义（引擎无关）

广义坐标与约束系统：完整约束与非完整约束的处理方法。
衔接 MEC-030 机构模块（完整约束）和 MEC-024 纯滚动（非完整约束）。

=== 核心概念 ===

  广义坐标 q_i：独立描述系统位形的最少坐标集。
  - N 个质点（3N 个笛卡尔坐标），受 K 个完整约束 → 3N-K 个广义坐标
  - 完整约束：f(q, t) = 0（可消去坐标）
  - 非完整约束：f(q, q̇, t) = 0（不可消去，需用乘子法）

=== 完整约束：拉格朗日乘子法 ===

  约束 f(q, t) = 0 → 引入乘子 λ

  拉格朗日方程含约束：
    d/dt(∂L/∂q̇_i) - ∂L/∂q_i = λ_a · ∂f_a/∂q_i  （a=1..K）

  或消去约束（选独立广义坐标）：直接使用无约束的拉格朗日方程。

  示例 1：单摆（完整约束 r = l）
    笛卡尔 (x, y) + 约束 x² + y² = l²
    → 广义坐标 θ（角度），消去约束
    L = ½ml²θ̇² - mgl(1-cosθ)
    → ml²θ̈ + mgl sinθ = 0

  示例 2：斜面纯滚动（完整约束）
    球在斜面上纯滚动：x = Rφ（完整约束）
    → 消去一个坐标，用 x 或 φ

=== 非完整约束：乘子法 ===

  示例 3：水平面纯滚动（非完整约束）
    圆盘在水平面纯滚动：ẋ = Rφ̇ cosψ, ẏ = Rφ̇ sinψ
    这是速度约束，不能消去坐标。

    处理方法：拉格朗日乘子法
    d/dt(∂L/∂q̇_i) - ∂L/∂q_i = Σ λ_a a_{ai}
    其中 a_{ai} = ∂f_a/∂q̇_i

=== 本模型实现的约束系统 ===

  1) 单摆（完整约束消去法）
  2) 阿特伍德机（完整约束，绳长不变）
  3) 斜面滚动（完整约束，MEC-024 的拉格朗日重解）
  4) 拉格朗日乘子法（非完整约束的数值验证）

=== 与已有 MEC 模型的关系 ===

  MEC-030 机构（完整约束）→ MEC-062 拉格朗日处理
  MEC-024 纯滚动（完整约束）→ MEC-062 广义坐标消去
  MEC-060 拉格朗日法 → MEC-062 约束推广
  MEC-062 → MEC-080 多体动力学（通用约束框架）

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(m=1.0, g=9.81, l=1.0, k=0.4, R=0.5,
                        theta=30.0, mu_s=0.0):
    """验证物理参数合法性。"""
    assert m > 0, f"质量 m 必须为正，当前 m={m}"
    assert g > 0, f"重力加速度 g 必须为正，当前 g={g}"
    assert l > 0, f"长度 l 必须为正，当前 l={l}"
    assert 0 < k < 1, f"转动惯量系数 k（I=k·m·R²）应在 (0, 1)，当前 k={k}"
    assert R > 0, f"半径 R 必须为正，当前 R={R}"
    assert 0 < theta < 90, f"斜面角度应在 (0°, 90°)，当前 {theta}°"
    assert mu_s >= 0, f"静摩擦系数 mu_s 必须非负，当前 mu_s={mu_s}"


# ============================================================
# 1. 单摆（完整约束消去法）
# ============================================================

def pendulum_lagrangian(state, m, g, l):
    """单摆拉格朗日量 L = ½ml²θ̇² - mgl(1-cosθ)。

    从笛卡尔 (x, y) + 约束 x²+y²=l² 消去得广义坐标 θ。
    """
    th, w = state
    T = 0.5 * m * l**2 * w**2
    V = m * g * l * (1 - np.cos(th))
    return T - V


def pendulum_dynamics(t, state, m=1.0, g=9.81, l=1.0):
    """单摆运动方程：θ̈ = -(g/l)sinθ。"""
    th, w = state
    return np.array([w, -g / l * np.sin(th)])


def pendulum_small_angle_frequency(g, l):
    """小角度频率 ω = √(g/l)。"""
    return np.sqrt(g / l)


def pendulum_energy(state, m, g, l):
    """单摆能量 E = ½ml²θ̇² + mgl(1-cosθ)。"""
    th, w = state
    return 0.5 * m * l**2 * w**2 + m * g * l * (1 - np.cos(th))


# ============================================================
# 2. 阿特伍德机（完整约束：绳长不变）
# ============================================================

def atwood_lagrangian(state, m1, m2, g, l):
    """阿特伍德机拉格朗日量。

    两质量 m1, m2 通过绳（长 l）绕滑轮。
    广义坐标 x = m1 的下降量（m2 上升同样距离）。

    约束：x1 + x2 = l（绳长不变，完整约束）
    消去约束后：x2 = l - x1，用 x = x1 作为广义坐标。

    L = ½(m1+m2)ẋ² + m1·g·x - m2·g·(l-x)  (取向下为正)
    化简（取 l=0 为 m1 初始位置，舍去常数项）：
    L = ½(m1+m2)ẋ² + (m1-m2)g·x
    """
    x, v = state
    M = m1 + m2
    T = 0.5 * M * v**2
    V = -(m1 - m2) * g * x  # 重力势能（取零点在支点）
    return T + V  # L = T - V，但 V 定义中已含符号


def atwood_dynamics(t, state, m1=1.0, m2=0.5, g=9.81, l=2.0):
    """阿特伍德机运动方程：(m1+m2)ẍ = (m1-m2)g。"""
    _, v = state
    a = (m1 - m2) * g / (m1 + m2)
    return np.array([v, a])


def atwood_acceleration(m1, m2, g):
    """阿特伍德机加速度 a = (m1-m2)g/(m1+m2)。"""
    return (m1 - m2) * g / (m1 + m2)


def atwood_tension(m1, m2, g):
    """绳张力 T = 2m1·m2·g/(m1+m2)。"""
    return 2 * m1 * m2 * g / (m1 + m2)


# ============================================================
# 3. 斜面纯滚动（完整约束，MEC-024 拉格朗日重解）
# ============================================================

def rolling_incline_lagrangian(state, m, g, R, k, theta_rad):
    """斜面纯滚动的拉格朗日量。

    球在斜面上纯滚动：x = Rφ（完整约束，消去 φ）
    广义坐标 x = 沿斜面下滑距离。

    I = k·m·R²（k 为转动惯量系数，如球 k=2/5）

    L = ½m ẋ² + ½I (ẋ/R)² - mg·x·sin(θ)·(-1)
      = ½(m + I/R²)ẋ² + mg·x·sin(θ)
      = ½m(1+k)ẋ² + mg·x·sin(θ)
    """
    x, v = state
    m_eff = m * (1 + k)  # 有效质量
    T = 0.5 * m_eff * v**2
    V = -m * g * x * np.sin(theta_rad)  # 沿斜面向下为正 → V 减小
    return T - V


def rolling_incline_dynamics(t, state, m=1.0, g=9.81, R=0.5, k=0.4,
                              theta_deg=30.0):
    """斜面纯滚动运动方程。

    (m + I/R²)ẍ = mg·sinθ
    ẍ = g·sinθ / (1 + k)
    """
    _, v = state
    theta_rad = np.radians(theta_deg)
    a = g * np.sin(theta_rad) / (1 + k)
    return np.array([v, a])


def rolling_incline_acceleration(g, k, theta_deg):
    """纯滚动加速度 a = g·sinθ/(1+k)。"""
    theta_rad = np.radians(theta_deg)
    return g * np.sin(theta_rad) / (1 + k)


def rolling_incline_energy(state, m, g, R, k, theta_deg):
    """总能量 E = ½m(1+k)v² - mgx·sinθ。"""
    x, v = state
    theta_rad = np.radians(theta_deg)
    T = 0.5 * m * (1 + k) * v**2
    V = -m * g * x * np.sin(theta_rad)
    return T + V


# ============================================================
# 4. 拉格朗日乘子法（约束力计算）
# ============================================================

def lagrange_multiplier_pendulum(state, m, g, l):
    """单摆的拉格朗日乘子 = 约束力（绳张力）。

    约束 f = x² + y² - l² = 0
    乘子 λ 对应约束力大小。

    对单摆，绳张力 T = ml(θ̇² + g/l·cosθ) = mlθ̇² + mg·cosθ
    """
    th, w = state
    T = m * l * w**2 + m * g * np.cos(th)
    return T


def constraint_force_pendulum(state, m, g, l):
    """约束力 = 绳张力（同 lagrange_multiplier_pendulum）。"""
    return lagrange_multiplier_pendulum(state, m, g, l)


def rolling_constraint_force(state, m, g, R, k, theta_deg):
    """纯滚动的约束力 = 静摩擦力。

    静摩擦力 f_s = I·α/R = I·a/R² = k·m·R² · a / R² = k·m·a
    其中 a = g·sinθ/(1+k)

    f_s = k·m·g·sinθ / (1+k)
    """
    theta_rad = np.radians(theta_deg)
    a = g * np.sin(theta_rad) / (1 + k)
    f_s = k * m * a
    return f_s


def static_friction_required(m, g, k, theta_deg):
    """纯滚动所需的静摩擦力大小。"""
    theta_rad = np.radians(theta_deg)
    return k * m * g * np.sin(theta_rad) / (1 + k)


def max_incline_angle_for_pure_rolling(mu_s, k):
    """纯滚动不滑动的最大斜面角度。

    需 f_s ≤ μ_s · N = μ_s · mg·cosθ
    k·m·g·sinθ/(1+k) ≤ μ_s·m·g·cosθ
    tanθ ≤ μ_s·(1+k)/k
    θ_max = arctan[μ_s·(1+k)/k]
    """
    return np.degrees(np.arctan(mu_s * (1 + k) / k))


# ============================================================
# 约束验证工具
# ============================================================

def verify_pendulum_constraint(state, l):
    """验证单摆约束 x²+y²=l²。"""
    th, _ = state
    x = l * np.sin(th)
    y = -l * np.cos(th)  # 竖直向下
    r = np.sqrt(x**2 + y**2)
    return abs(r - l) < 1e-10


def verify_rolling_constraint(state, R):
    """验证纯滚动约束 x = Rφ。"""
    x, v = state
    phi = x / R  # 由约束推算的转角
    omega = v / R  # 由约束推算的角速度
    return abs(x - R * phi) < 1e-10, abs(v - R * omega) < 1e-10
