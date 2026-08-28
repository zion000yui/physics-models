"""MEC-031-slider-crank — 模型定义（引擎无关）

曲柄滑块机构（slider-crank mechanism）：由曲柄（半径 r）、连杆（长度 l）
和滑块组成。曲柄绕固定铰 O 旋转，连杆连接曲柄销 A 和滑块 B，滑块被约束
沿水平直线运动。这是内燃机、压缩机的核心机构。

机构拓扑为 R-R-R-P（3 个转动副 + 1 个移动副），与四连杆 R-R-R-R 不同。
l→∞ 时滑块运动学趋近简谐，但这只是数学极限，不是严格物理退化。

=== 坐标系 ===

  O 在原点 (0, 0)，滑块沿 x 轴运动
  θ：曲柄角（从 x 轴逆时针）
  φ：连杆角（从 x 轴，sin φ = (r/l) sin θ）

  A（曲柄销）= (r cos θ, r sin θ)
  B（滑块）  = (x, 0)，其中 x = r cos θ + l cos φ = r cos θ + √(l²-r²sin²θ)

=== 运动学（解析解）===

  位置：
    x(θ) = r cos θ + √(l² - r² sin²θ)
    φ(θ) = arcsin((r/l) sin θ)

  速度（ω = θ̇）：
    ẋ = -r ω sin θ - r² ω sin θ cos θ / √(l²-r²sin²θ)
    φ̇ = (r ω cos θ) / (l cos φ)

  加速度（α = θ̈）：
    ẍ = -r(α sin θ + ω² cos θ) - d/dt[r² ω sin θ cos θ / √(l²-r²sin²θ)]
    （完整表达式见代码实现）

=== 等效惯量动力学（1-DOF）===

  T = ½ I_O ω² + ½ m_rod v_cm² + ½ I_rod φ̇² + ½ m_slider ẋ²

  其中：
    I_O = I_crank + m_crank · r_cm²（曲柄绕 O 的转动惯量）
    v_cm = 连杆质心速度 = (l2·cos θ + r3·cos φ, ...) 的导数
    φ̇ = (r ω cos θ)/(l cos φ)
    ẋ = dx/dθ · ω

  等效到曲柄：
    I_eff(θ) = I_O + m_rod·(dx_cm/dθ)² + I_rod·(dφ/dθ)² + m_sl·(dx/dθ)²

  运动方程（拉格朗日推导）：
    I_eff(θ)·α + ½·I_eff'(θ)·ω² + dV/dθ = τ

  动力学 ODE 无闭式解析解（变等效惯量），数值积分 + 能量守恒验证。

=== 与已有 MEC 模型的关系 ===

  曲柄绕固定铰转动：MEC-021 概念
  连杆做一般平面运动：MEC-022 概念
  滑块沿直线平动：MEC-020 概念
  约束减少自由度：MEC-024 概念
  与 MEC-030 四连杆：l→∞ 时数学极限趋近，但 RRRR→RRRP 拓扑变化，非严格退化

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(r=0.3, l=1.0, m_crank=1.0, m_rod=1.0, m_sl=1.0,
                        r_cm=None, l_cm=None, I_crank=None, I_rod=None,
                        g=0.0, tau=0.0):
    """验证物理参数合法性。"""
    assert r > 0, f"曲柄半径 r 必须为正，当前 r={r}"
    assert l > 0, f"连杆长度 l 必须为正，当前 l={l}"
    assert l > r, f"连杆长度 l 必须大于曲柄半径 r（否则机构锁死），当前 l={l}, r={r}"
    assert m_crank >= 0, f"曲柄质量 m_crank 必须非负，当前 m_crank={m_crank}"
    assert m_rod >= 0, f"连杆质量 m_rod 必须非负，当前 m_rod={m_rod}"
    assert m_sl >= 0, f"滑块质量 m_sl 必须非负，当前 m_sl={m_sl}"
    assert g >= 0, f"重力加速度 g 必须非负，当前 g={g}"


def _resolve_defaults(r, l, m_crank, m_rod, m_sl,
                       r_cm, l_cm, I_crank, I_rod):
    """解析 None 默认值为均匀杆假设。"""
    if r_cm is None:
        r_cm = r / 2
    if l_cm is None:
        l_cm = l / 2
    if I_crank is None:
        I_crank = m_crank * r**2 / 12
    if I_rod is None:
        I_rod = m_rod * l**2 / 12
    return r_cm, l_cm, I_crank, I_rod


# ===================== 运动学（解析解）=====================

def slider_position(theta, r, l):
    """滑块位置 x = r cos θ + √(l² - r² sin²θ)。"""
    return r * np.cos(theta) + np.sqrt(l**2 - r**2 * np.sin(theta)**2)


def rod_angle(theta, r, l):
    """连杆角 φ = arcsin((r/l) sin θ)。"""
    return np.arcsin(r / l * np.sin(theta))


def slider_velocity_ratio(theta, r, l):
    """计算 dx/dθ（滑块速度比）。

    ẋ = (dx/dθ) · ω
    dx/dθ = -r sin θ - r² sin θ cos θ / √(l²-r²sin²θ)
    """
    s = np.sin(theta)
    c = np.cos(theta)
    sqrt_term = np.sqrt(l**2 - r**2 * s**2)
    return -r * s - r**2 * s * c / sqrt_term


def rod_angular_velocity_ratio(theta, r, l):
    """计算 dφ/dθ（连杆角速度比）。

    φ̇ = (dφ/dθ) · ω
    dφ/dθ = (r cos θ) / (l cos φ) = r cos θ / √(l²-r²sin²θ)
    """
    c = np.cos(theta)
    sqrt_term = np.sqrt(l**2 - r**2 * np.sin(theta)**2)
    return r * c / sqrt_term


def slider_acceleration_ratio(theta, r, l):
    """计算 d²x/dθ²（滑块加速度比，用于 α=0 时 ÿ=(d²x/dθ²)ω²）。

    对 dx/dθ 再求 θ 的导数。
    """
    s = np.sin(theta)
    c = np.cos(theta)
    D = l**2 - r**2 * s**2
    sqrt_D = np.sqrt(D)

    # d/dθ[-r s - r² s c / sqrt_D]
    # = -r c - r²(c² - s²)/sqrt_D - r² s c · (r² s c) / D^(3/2)
    # = -r c - r²(c²-s²)/sqrt_D - r⁴ s² c² / D^(3/2)
    term1 = -r * c
    term2 = -r**2 * (c**2 - s**2) / sqrt_D
    term3 = -r**4 * s**2 * c**2 / (D * sqrt_D)
    return term1 + term2 + term3


def velocity_analysis(theta, omega, r, l):
    """计算滑块速度 ẋ 和连杆角速度 φ̇。"""
    dx_dtheta = slider_velocity_ratio(theta, r, l)
    dphi_dtheta = rod_angular_velocity_ratio(theta, r, l)
    return dx_dtheta * omega, dphi_dtheta * omega


def acceleration_analysis(theta, omega, alpha, r, l):
    """计算滑块加速度 ÿ 和连杆角加速度 φ̈。

    ÿ = (d²x/dθ²)ω² + (dx/dθ)α
    φ̈ = (d²φ/dθ²)ω² + (dφ/dθ)α
    """
    dx_dtheta = slider_velocity_ratio(theta, r, l)
    d2x_dtheta2 = slider_acceleration_ratio(theta, r, l)
    dphi_dtheta = rod_angular_velocity_ratio(theta, r, l)

    # d²φ/dθ²: 对 dφ/dθ = r cos θ / sqrt_D 求导
    s = np.sin(theta)
    c = np.cos(theta)
    D = l**2 - r**2 * s**2
    sqrt_D = np.sqrt(D)
    d2phi_dtheta2 = -r * s / sqrt_D - r**3 * s * c**2 / (D * sqrt_D)

    slider_acc = d2x_dtheta2 * omega**2 + dx_dtheta * alpha
    rod_alpha = d2phi_dtheta2 * omega**2 + dphi_dtheta * alpha
    return slider_acc, rod_alpha


# ===================== 极限位置 =====================

def toggle_positions(r, l):
    """计算滑块的两个极限位置（上止点/下止点）。

    曲柄与连杆共线时滑块在极限位置，此时滑块速度 ẋ=0。

    上止点（TDC, 伸展）：θ=0, x = r + l
    下止点（BDC, 折叠）：θ=π, x = -r + l = l - r

    返回 (x_tdc, x_bdc, theta_tdc, theta_bdc)。
    """
    x_tdc = r + l
    x_bdc = l - r
    return x_tdc, x_bdc, 0.0, np.pi


# ===================== 等效惯量动力学 =====================

def equivalent_inertia(theta, r, l, m_crank=1.0, m_rod=1.0, m_sl=1.0,
                       r_cm=None, l_cm=None, I_crank=None, I_rod=None):
    """计算等效到曲柄的转动惯量 I_eff(θ)。

    T = ½ I_eff ω²
    I_eff = I_O + m_rod·v_cm²/ω² + I_rod·(dφ/dθ)² + m_sl·(dx/dθ)²

    其中：
      I_O = I_crank + m_crank·r_cm²（曲柄绕 O 的转动惯量）
      v_cm²/ω² = (dx_cm/dθ)² + (dy_cm/dθ)²
        连杆质心 = (r cos θ + l_cm cos φ, r sin θ + l_cm sin φ)
        dx_cm/dθ = -r sin θ - l_cm sin φ · dφ/dθ
        dy_cm/dθ = r cos θ + l_cm cos φ · dφ/dθ
    """
    r_cm, l_cm, I_crank, I_rod = _resolve_defaults(
        r, l, m_crank, m_rod, m_sl, r_cm, l_cm, I_crank, I_rod)

    phi = rod_angle(theta, r, l)
    dphi = rod_angular_velocity_ratio(theta, r, l)
    dx = slider_velocity_ratio(theta, r, l)

    # 曲柄绕 O 的转动惯量
    I_O = I_crank + m_crank * r_cm**2

    # 连杆质心速度比
    # 连杆质心位置 = (r cos θ + l_cm cos φ, r sin θ + l_cm sin φ)
    # d/dθ 后乘 ω 得速度
    s_t = np.sin(theta)
    c_t = np.cos(theta)
    s_p = np.sin(phi)
    c_p = np.cos(phi)

    dx_cm = -r * s_t - l_cm * s_p * dphi  # dx_cm/dθ
    dy_cm = r * c_t + l_cm * c_p * dphi   # dy_cm/dθ
    v_cm_sq = dx_cm**2 + dy_cm**2

    # 等效惯量
    I_eff = I_O + m_rod * v_cm_sq + I_rod * dphi**2 + m_sl * dx**2
    return I_eff


def potential_energy(state, r, l, m_crank=1.0, m_rod=1.0, m_sl=1.0,
                     r_cm=None, l_cm=None, g=0.0):
    """计算重力势能 V（水平面时 g=0，V=0）。

    滑块沿水平方向运动，y=0，所以滑块 PE=0。
    曲柄质心高度 y_crank = r_cm sin θ
    连杆质心高度 y_rod = r sin θ + l_cm sin φ
    """
    if g == 0:
        return 0.0
    r_cm, l_cm, _, _ = _resolve_defaults(
        r, l, m_crank, m_rod, m_sl, r_cm, l_cm, None, None)

    theta, _ = state
    phi = rod_angle(theta, r, l)
    V = (m_crank * g * r_cm * np.sin(theta)
         + m_rod * g * (r * np.sin(theta) + l_cm * np.sin(phi)))
    return V


def mechanical_energy(state, r, l, m_crank=1.0, m_rod=1.0, m_sl=1.0,
                      r_cm=None, l_cm=None, I_crank=None, I_rod=None, g=0.0):
    """计算总机械能 E = ½ I_eff ω² + V。"""
    theta, omega = state
    I_eff = equivalent_inertia(theta, r, l, m_crank, m_rod, m_sl,
                               r_cm, l_cm, I_crank, I_rod)
    V = potential_energy(state, r, l, m_crank, m_rod, m_sl,
                         r_cm, l_cm, g)
    return 0.5 * I_eff * omega**2 + V


def dynamics(t, state, r, l, m_crank=1.0, m_rod=1.0, m_sl=1.0,
             r_cm=None, l_cm=None, I_crank=None, I_rod=None,
             g=0.0, tau=0.0):
    """返回状态时间导数 [dθ/dt, dω/dt]。

    运动方程：I_eff(θ)·α + ½·I_eff'(θ)·ω² + dV/dθ = τ

    I_eff' 用中心差分数值计算。
    """
    r_cm, l_cm, I_crank, I_rod = _resolve_defaults(
        r, l, m_crank, m_rod, m_sl, r_cm, l_cm, I_crank, I_rod)

    theta, omega = state

    # 当前 I_eff
    I_eff = equivalent_inertia(theta, r, l, m_crank, m_rod, m_sl,
                               r_cm, l_cm, I_crank, I_rod)

    # I_eff' 数值中心差分
    h = 1e-7
    I_eff_p = equivalent_inertia(theta + h, r, l, m_crank, m_rod, m_sl,
                                 r_cm, l_cm, I_crank, I_rod)
    I_eff_m = equivalent_inertia(theta - h, r, l, m_crank, m_rod, m_sl,
                                 r_cm, l_cm, I_crank, I_rod)
    I_eff_prime = (I_eff_p - I_eff_m) / (2 * h)

    # 重力势能导数 dV/dθ
    if g != 0:
        phi = rod_angle(theta, r, l)
        dphi = rod_angular_velocity_ratio(theta, r, l)
        dV_dtheta = (m_crank * g * r_cm * np.cos(theta)
                     + m_rod * g * (r * np.cos(theta)
                                    + l_cm * np.cos(phi) * dphi))
    else:
        dV_dtheta = 0.0

    # 运动方程
    alpha = (tau - 0.5 * I_eff_prime * omega**2 - dV_dtheta) / I_eff

    return np.array([omega, alpha])
