"""MEC-060-lagrangian-formulation — 模型定义（引擎无关）

拉格朗日力学公式化：用拉格朗日方程重新求解已有模型，
验证与牛顿法结果一致，建立分析力学框架。

=== 核心原理 ===

  拉格朗日量 L = T - V（动能 - 势能）

  欧拉-拉格朗日方程：
    d/dt(∂L/∂q̇_i) - ∂L/∂q_i = Q_i  （Q_i 为非保守广义力）

  对保守系统（Q_i = 0）：
    d/dt(∂L/∂q̇_i) - ∂L/∂q_i = 0

=== 重新求解的已有模型 ===

  1) MEC-001 自由质点（验证拉格朗日法 → 匀速直线运动）
     L = ½ m ẋ²
     ∂L/∂ẋ = m ẋ, d/dt(mẋ) = m ẍ
     ∂L/∂x = 0
     → m ẍ = 0（动量守恒）

  2) MEC-002 受力质点（验证恒力 → 匀加速）
     L = ½ m ẋ² - F x（保守力 F）
     ∂L/∂ẋ = m ẋ, ∂L/∂x = -F
     → m ẍ = F

  3) MEC-006 中心力胡克（验证 F = -kx → 简谐振动）
     L = ½ m (ẋ² + ẏ²) - ½ k (x² + y²)
     → m ẍ = -kx, m ÿ = -ky（各方向独立简谐振动）

  4) MEC-010 弹簧振子（验证 ω = √(k/m)）
     L = ½ m ẋ² - ½ k x²
     → m ẍ + k x = 0 → ω = √(k/m)

  5) MEC-013 双摆（拉格朗日法处理约束的典型范例）
     两质点 m₁, m₂，杆长 l₁, l₂，角度 θ₁, θ₂
     T = ½ m₁ l₁² θ̇₁² + ½ m₂ [l₁² θ̇₁² + l₂² θ̇₂²
         + 2 l₁ l₂ θ̇₁ θ̇₂ cos(θ₁-θ₂)]
     V = -m₁ g l₁ cos θ₁ - m₂ g (l₁ cos θ₁ + l₂ cos θ₂)
     L = T - V

     双摆的拉格朗日方程是非线性耦合的 2 自由度系统，
     是拉格朗日法相对于牛顿法优势的典型例证。

  6) MEC-011 阻尼振子（非保守力 Q = -c ẋ）
     L = ½ m ẋ² - ½ k x²
     Q = -c ẋ（阻尼力）
     → m ẍ + c ẋ + k x = 0

=== 拉格朗日法的优势 ===

  - 约束系统（如双摆）：无需显式计算约束力
  - 广义坐标自动处理几何约束
  - 对称性 → 守恒律（Noether 定理）
  - 标量运算（能量），无需矢量分解

=== 与已有 MEC 模型的关系 ===

  MEC-001~009 质点 → MEC-060 拉格朗日重新求解
  MEC-010~015 振动 → MEC-060 拉格朗日重新求解
  MEC-060 → MEC-061 哈密顿正则方程（Legendre 变换）
  MEC-060 → MEC-062 广义坐标与约束

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(m=1.0, k=1.0, g=9.81, c=0.0):
    """验证物理参数合法性。"""
    assert m > 0, f"质量 m 必须为正，当前 m={m}"
    assert k > 0, f"弹簧常数 k 必须为正，当前 k={k}"
    assert g > 0, f"重力加速度 g 必须为正，当前 g={g}"
    assert c >= 0, f"阻尼系数 c 必须非负，当前 c={c}"


# ============================================================
# 1. 自由质点 (MEC-001)
# ============================================================

def free_particle_lagrangian(state, m):
    """L = ½ m ẋ²（无势能）。"""
    _, v = state
    return 0.5 * m * v**2


def free_particle_dynamics(t, state, m=1.0):
    """拉格朗日方程：m ẍ = 0 → ẍ = 0。"""
    _, v = state
    return np.array([v, 0.0])


# ============================================================
# 2. 受力质点 (MEC-002)
# ============================================================

def forced_particle_lagrangian(state, m, F):
    """L = ½ m ẋ² - F x（保守恒力）。"""
    x, v = state
    return 0.5 * m * v**2 - F * x


def forced_particle_dynamics(t, state, m=1.0, F=1.0):
    """拉格朗日方程：m ẍ = F。"""
    _, v = state
    return np.array([v, F / m])


# ============================================================
# 3. 弹簧振子 (MEC-010)
# ============================================================

def spring_lagrangian(state, m, k):
    """L = ½ m ẋ² - ½ k x²。"""
    x, v = state
    return 0.5 * m * v**2 - 0.5 * k * x**2


def spring_dynamics(t, state, m=1.0, k=1.0):
    """拉格朗日方程：m ẍ + k x = 0。"""
    x, v = state
    return np.array([v, -k * x / m])


def spring_natural_frequency(m, k):
    """ω = √(k/m)。"""
    return np.sqrt(k / m)


# ============================================================
# 4. 阻尼振子 (MEC-011)
# ============================================================

def damped_spring_lagrangian(state, m, k):
    """L = ½ m ẋ² - ½ k x²（阻尼力作为非保守力 Q = -c ẋ）。"""
    x, v = state
    return 0.5 * m * v**2 - 0.5 * k * x**2


def damped_spring_dynamics(t, state, m=1.0, k=1.0, c=0.1):
    """拉格朗日方程 + 非保守力：m ẍ + c ẋ + k x = 0。"""
    x, v = state
    return np.array([v, (-c * v - k * x) / m])


def damping_ratio(m, k, c):
    """阻尼比 ζ = c / (2√(mk))。"""
    return c / (2 * np.sqrt(m * k))


# ============================================================
# 5. 中心力胡克 (MEC-006)
# ============================================================

def hooke_lagrangian_2d(state, m, k):
    """L = ½ m (ẋ² + ẏ²) - ½ k (x² + y²)。

    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    T = 0.5 * m * (vx**2 + vy**2)
    V = 0.5 * k * (x**2 + y**2)
    return T - V


def hooke_dynamics_2d(t, state, m=1.0, k=1.0):
    """拉格朗日方程：m ẍ = -kx, m ÿ = -ky。"""
    x, y, vx, vy = state
    ax = -k * x / m
    ay = -k * y / m
    return np.array([vx, vy, ax, ay])


# ============================================================
# 6. 双摆 (MEC-013)
# ============================================================

def double_pendulum_lagrangian(state, m1, m2, l1, l2, g):
    """双摆拉格朗日量。

    state = [θ₁, θ₂, θ̇₁, θ̇₂]
    θ 从竖直向下方向测量。
    """
    th1, th2, w1, w2 = state

    # 动能
    T = (0.5 * m1 * l1**2 * w1**2
         + 0.5 * m2 * (l1**2 * w1**2 + l2**2 * w2**2
                        + 2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)))

    # 势能（取支点为零势能面，向下为负）
    V = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)

    return T - V


def double_pendulum_dynamics(t, state, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """双摆拉格朗日运动方程。

    通过欧拉-拉格朗日方程推导得到 2×2 线性系统：

      [(m₁+m₂)l₁²    m₂l₁l₂cos(δ)] [θ̈₁]   [b₁]
      [m₂l₁l₂cos(δ)   m₂l₂²     ] [θ̈₂] = [b₂]

    其中 δ = θ₁-θ₂, det = m₂l₁²l₂²[(m₁+m₂) - m₂cos²(δ)]

      b₁ = -m₂l₁l₂sin(δ)θ̇₂² - (m₁+m₂)gl₁sin(θ₁)
      b₂ =  m₂l₁l₂sin(δ)θ̇₁² - m₂gl₂sin(θ₂)
    """
    th1, th2, w1, w2 = state
    delta = th1 - th2
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)

    # 质量矩阵
    M11 = (m1 + m2) * l1**2
    M12 = m2 * l1 * l2 * cos_d
    M22 = m2 * l2**2
    det = M11 * M22 - M12**2

    # 右端项
    b1 = -m2 * l1 * l2 * sin_d * w2**2 - (m1 + m2) * g * l1 * np.sin(th1)
    b2 = m2 * l1 * l2 * sin_d * w1**2 - m2 * g * l2 * np.sin(th2)

    # 求解 2×2 系统
    a1 = (b1 * M22 - b2 * M12) / det
    a2 = (M11 * b2 - M12 * b1) / det

    return np.array([w1, w2, a1, a2])


def double_pendulum_energy(state, m1, m2, l1, l2, g):
    """总能量 E = T + V（用于能量守恒验证）。"""
    th1, th2, w1, w2 = state
    T = (0.5 * m1 * l1**2 * w1**2
         + 0.5 * m2 * (l1**2 * w1**2 + l2**2 * w2**2
                        + 2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)))
    V = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
    return T + V


# ============================================================
# 能量验证工具
# ============================================================

def euler_lagrange_residual(L_func, state_func, t, params, dt=1e-6):
    """数值验证欧拉-拉格朗日方程。

    检查 d/dt(∂L/∂q̇) - ∂L/∂q ≈ 0（保守系统）。

    L_func: L(state, *params)
    state_func: function(t) → state
    """
    state = state_func(t)
    state_p = state_func(t + dt)
    state_m = state_func(t - dt)

    # ∂L/∂q̇ ≈ (L(q, q̇+h) - L(q, q̇-h)) / (2h)
    # d/dt(∂L/∂q̇) ≈ (∂L/∂q̇|_{t+dt} - ∂L/∂q̇|_{t-dt}) / (2dt)
    # ∂L/∂q ≈ (L(q+h, q̇) - L(q-h, q̇)) / (2h)
    # 这是示意性的；实际实现需要更仔细
    pass


def noether_charge_momentum(L_func, state, m, coord='x'):
    """Noether 定理：平移对称性 → 动量守恒。

    对自由质点：∂L/∂ẋ = mẋ = p（守恒）
    """
    if coord == 'x':
        _, v = state[:2]
        return m * v
    return None
