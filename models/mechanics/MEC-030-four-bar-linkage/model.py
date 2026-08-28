"""MEC-030-four-bar-linkage — 模型定义（引擎无关）

平面四连杆机构（planar four-bar linkage）：四根刚性杆——固定杆（l₁, 机架）、
曲柄（l₂, 输入）、连杆（l₃, 耦合件）、摇杆（l₄, 输出）——通过四个转动副
（revolute joint）连接形成闭环。曲柄绕固定铰 O₂ 旋转，通过连杆驱动摇杆绕
固定铰 O₄ 摆动或旋转。

这是最简单的闭环低副机构，引入闭环运动学约束和等效惯量动力学。

=== 坐标系 ===

  O₂ 在原点 (0, 0)，O₄ 在 (l₁, 0)
  所有角度从 x 轴正方向逆时针计量
  θ₂：曲柄角（输入），θ₃：连杆角，θ₄：摇杆角

  A（曲柄末端）= (l₂·cosθ₂, l₂·sinθ₂)
  B（摇杆末端）= (l₁ + l₄·cosθ₄, l₄·sinθ₄)
  |B - A| = l₃（闭环约束）

=== Freudenstein 方程 ===

  K₁·cosθ₄ - K₂·cosθ₂ + K₃ = cos(θ₂ - θ₄)

  K₁ = l₁/l₂, K₂ = l₁/l₄, K₃ = (l₁² + l₂² - l₃² + l₄²)/(2·l₂·l₄)

  半角代换求解 θ₄，两个分支（open / crossed）

=== 速度分析 ===

  ω₃/ω₂ = l₂·sin(θ₂-θ₄) / (l₃·sin(θ₄-θ₃))
  ω₄/ω₂ = l₂·sin(θ₂-θ₃) / (l₄·sin(θ₄-θ₃))

=== 等效惯量动力学（1-DOF）===

  I_eff(θ₂)·α₂ + ½·I_eff'(θ₂)·ω₂² + dV/dθ₂ = τ

  I_eff = I₂_O₂ + m₃·(l₂² + r₃²·R₃² + 2·l₂·r₃·R₃·cos(θ₂-θ₃))
          + I₃·R₃² + I₄_O₄·R₄²

  其中 R₃ = ω₃/ω₂, R₄ = ω₄/ω₂,
  I₂_O₂ = I₂ + m₂·r₂², I₄_O₄ = I₄ + m₄·r₄²

  动力学 ODE 一般无闭式解析解（变等效惯量），数值积分 + 能量守恒验证。

=== 与已有 MEC 模型的关系 ===

  每根杆是 MEC-022 类型刚体（平面运动）
  曲柄、摇杆绕固定铰转动（MEC-021 概念）
  闭环约束是新概念，不在任何已有模型中
  无严格退化到任何已有 MEC 模型

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(l1, l2, l3, l4, m2=1.0, m3=1.0, m4=1.0,
                        r2=None, r3=None, r4=None,
                        I2=None, I3=None, I4=None,
                        g=0.0, tau=0.0):
    """验证基本物理参数合法性。

    参数
    ----
    l1, l2, l3, l4 : float
        四根杆的长度（必须 > 0）。
    m2, m3, m4 : float
        各杆质量（必须 >= 0，动力学需要 > 0）。
    r2, r3, r4 : float or None
        各杆质心到近端铰的距离（None 时取 l/2，即均匀杆）。
    I2, I3, I4 : float or None
        各杆绕质心的转动惯量（None 时取 m·l²/12，即均匀杆）。
    g : float
        重力加速度（默认 0，水平面运动）。
    tau : float
        作用于曲柄的输入力矩。

    返回
    ----
    None
    """
    assert l1 > 0, f"杆长 l1 必须为正，当前 l1={l1}"
    assert l2 > 0, f"杆长 l2 必须为正，当前 l2={l2}"
    assert l3 > 0, f"杆长 l3 必须为正，当前 l3={l3}"
    assert l4 > 0, f"杆长 l4 必须为正，当前 l4={l4}"
    assert m2 >= 0, f"质量 m2 必须非负，当前 m2={m2}"
    assert m3 >= 0, f"质量 m3 必须非负，当前 m3={m3}"
    assert m4 >= 0, f"质量 m4 必须非负，当前 m4={m4}"
    assert g >= 0, f"重力加速度 g 必须非负，当前 g={g}"


def grashof_criterion(l1, l2, l3, l4):
    """判定四连杆机构的 Grashof 类型。

    返回值：
    - 'double-crank'：机架最短，双曲柄
    - 'crank-rocker'：曲柄或摇杆最短，曲柄摇杆
    - 'grashof-double-rocker'：连杆最短，Grashof 双摇杆
    - 'rocker-crank'：摇杆最短（反转曲柄摇杆）
    - 'non-grashof'：非 Grashof，无杆整周转动
    """
    s = min(l1, l2, l3, l4)
    l_max = max(l1, l2, l3, l4)
    sorted_vals = sorted([l1, l2, l3, l4])
    p_plus_q = sorted_vals[1] + sorted_vals[2]

    if s + l_max > p_plus_q + 1e-14:
        return 'non-grashof'

    if s == l1:
        return 'double-crank'
    elif s == l2:
        return 'crank-rocker'
    elif s == l3:
        return 'grashof-double-rocker'
    else:
        return 'rocker-crank'


def freudenstein_coefficients(l1, l2, l3, l4):
    """计算 Freudenstein 方程系数 K1, K2, K3。"""
    K1 = l1 / l2
    K2 = l1 / l4
    K3 = (l1**2 + l2**2 - l3**2 + l4**2) / (2 * l2 * l4)
    return K1, K2, K3


def position_analysis(theta2, l1, l2, l3, l4, config='open'):
    """由曲柄角 θ₂ 求解连杆角 θ₃ 和摇杆角 θ₄。

    使用 Freudenstein 方程的半角代换求解。返回 (theta3, theta4)。

    参数
    ----
    theta2 : float
        曲柄角度（rad）。
    config : str
        'open' 或 'crossed'，选择两个构型分支。
    """
    K1, K2, K3 = freudenstein_coefficients(l1, l2, l3, l4)

    ct2 = np.cos(theta2)
    st2 = np.sin(theta2)

    A = K1 - ct2
    B = -st2
    C = K2 * ct2 - K3

    D = A**2 + B**2 - C**2
    if D < -1e-12:
        raise ValueError(
            f"机构在 θ₂={theta2:.6f} rad 处不可装配 "
            f"(判别式 D={D:.6e} < 0)"
        )
    D = max(D, 0.0)
    sqrtD = np.sqrt(D)

    denom = A + C
    if abs(denom) < 1e-14:
        # 退化为线性方程 -2B·t + (C-A) = 0
        if abs(B) < 1e-14:
            raise ValueError("退化位置：A+C=0 且 B=0")
        t = (C - A) / (2 * B)
    else:
        if config == 'open':
            t = (B + sqrtD) / denom
        elif config == 'crossed':
            t = (B - sqrtD) / denom
        else:
            raise ValueError(f"config 必须为 'open' 或 'crossed'，当前 '{config}'")

    theta4 = 2 * np.arctan(t)

    # 连杆角 θ₃ 由闭环约束直接求
    dx = l1 + l4 * np.cos(theta4) - l2 * ct2
    dy = l4 * np.sin(theta4) - l2 * st2
    theta3 = np.arctan2(dy, dx)

    return theta3, theta4


def velocity_ratios(theta2, theta3, theta4, l1, l2, l3, l4):
    """计算速度比 R3 = ω₃/ω₂ 和 R4 = ω₄/ω₂。"""
    sin_24 = np.sin(theta2 - theta4)
    sin_23 = np.sin(theta2 - theta3)
    sin_43 = np.sin(theta4 - theta3)

    if abs(sin_43) < 1e-14:
        raise ValueError(
            f"近奇异位置：sin(θ₄-θ₃)≈0，连杆与摇杆共线"
        )

    R3 = l2 * sin_24 / (l3 * sin_43)
    R4 = l2 * sin_23 / (l4 * sin_43)
    return R3, R4


def velocity_analysis(theta2, theta3, theta4, omega2, l1, l2, l3, l4):
    """计算角速度 ω₃ 和 ω₄。"""
    R3, R4 = velocity_ratios(theta2, theta3, theta4, l1, l2, l3, l4)
    return R3 * omega2, R4 * omega2


def acceleration_analysis(theta2, theta3, theta4, omega2, omega3, omega4,
                          alpha2, l1, l2, l3, l4):
    """计算角加速度 α₃ 和 α₄。

    对速度方程求导，系数矩阵与速度分析相同，右端项含 ω² 和 α₂。
    """
    sin_43 = np.sin(theta4 - theta3)
    if abs(sin_43) < 1e-14:
        raise ValueError(
            f"近奇异位置：sin(θ₄-θ₃)≈0"
        )

    # 系数矩阵（与速度分析相同）
    a11 = l3 * np.sin(theta3)
    a12 = -l4 * np.sin(theta4)
    a21 = l3 * np.cos(theta3)
    a22 = -l4 * np.cos(theta4)

    det = a11 * a22 - a12 * a21  # = l3*l4*sin(theta4-theta3)

    # 右端项
    c1 = (-l2 * np.sin(theta2) * alpha2
          - l2 * np.cos(theta2) * omega2**2
          - l3 * np.cos(theta3) * omega3**2
          + l4 * np.cos(theta4) * omega4**2)
    c2 = (-l2 * np.cos(theta2) * alpha2
          + l2 * np.sin(theta2) * omega2**2
          + l3 * np.sin(theta3) * omega3**2
          - l4 * np.sin(theta4) * omega4**2)

    # Cramer 法则
    alpha3 = (c1 * a22 - a12 * c2) / det
    alpha4 = (a11 * c2 - c1 * a21) / det
    return alpha3, alpha4


def toggle_positions(l1, l2, l3, l4):
    """计算摇杆的两个极限位置（toggle positions）。

    在极限位置处曲柄与连杆共线，摇杆角速度 ω₄ = 0。

    返回 (theta4_ext, theta4_fold, theta2_ext, theta2_fold)。
    theta4_ext：伸展极限（|O₂B| = l₂+l₃）
    theta4_fold：折叠极限（|O₂B| = |l₂-l₃|）
    """
    # 伸展极限
    cos4_ext = ((l2 + l3)**2 - l1**2 - l4**2) / (2 * l1 * l4)
    cos4_ext = np.clip(cos4_ext, -1.0, 1.0)
    theta4_ext = np.arccos(cos4_ext)

    # 折叠极限
    cos4_fold = ((l2 - l3)**2 - l1**2 - l4**2) / (2 * l1 * l4)
    cos4_fold = np.clip(cos4_fold, -1.0, 1.0)
    theta4_fold = np.arccos(cos4_fold)

    # 对应曲柄角
    phi_ext = np.arctan2(l4 * np.sin(theta4_ext),
                          l1 + l4 * np.cos(theta4_ext))
    theta2_ext = phi_ext  # l2+l3 > 0

    phi_fold = np.arctan2(l4 * np.sin(theta4_fold),
                           l1 + l4 * np.cos(theta4_fold))
    if l2 - l3 >= 0:
        theta2_fold = phi_fold
    else:
        theta2_fold = phi_fold + np.pi
        theta2_fold = (theta2_fold + np.pi) % (2 * np.pi) - np.pi

    return theta4_ext, theta4_fold, theta2_ext, theta2_fold


def _resolve_defaults(l1, l2, l3, l4, m2, m3, m4, r2, r3, r4, I2, I3, I4):
    """解析 None 默认值为均匀杆假设。"""
    if r2 is None:
        r2 = l2 / 2
    if r3 is None:
        r3 = l3 / 2
    if r4 is None:
        r4 = l4 / 2
    if I2 is None:
        I2 = m2 * l2**2 / 12
    if I3 is None:
        I3 = m3 * l3**2 / 12
    if I4 is None:
        I4 = m4 * l4**2 / 12
    return r2, r3, r4, I2, I3, I4


def equivalent_inertia(theta2, l1, l2, l3, l4, m2=1.0, m3=1.0, m4=1.0,
                       r2=None, r3=None, r4=None,
                       I2=None, I3=None, I4=None, config='open'):
    """计算等效到曲柄的转动惯量 I_eff(θ₂)。

    I_eff = I₂_O₂ + m₃·(l₂² + r₃²·R₃² + 2·l₂·r₃·R₃·cos(θ₂-θ₃))
            + I₃·R₃² + I₄_O₄·R₄²
    """
    r2, r3, r4, I2, I3, I4 = _resolve_defaults(
        l1, l2, l3, l4, m2, m3, m4, r2, r3, r4, I2, I3, I4)

    theta3, theta4 = position_analysis(theta2, l1, l2, l3, l4, config)
    R3, R4 = velocity_ratios(theta2, theta3, theta4, l1, l2, l3, l4)

    I2_O2 = I2 + m2 * r2**2
    I4_O4 = I4 + m4 * r4**2

    v_cm3_sq = (l2**2 + r3**2 * R3**2
                + 2 * l2 * r3 * R3 * np.cos(theta2 - theta3))

    I_eff = I2_O2 + m3 * v_cm3_sq + I3 * R3**2 + I4_O4 * R4**2
    return I_eff


def potential_energy(state, l1, l2, l3, l4, m2=1.0, m3=1.0, m4=1.0,
                     r2=None, r3=None, r4=None, g=0.0, config='open'):
    """计算重力势能 V（水平面时 g=0，V=0）。"""
    if g == 0:
        return 0.0
    r2, r3, r4, _, _, _ = _resolve_defaults(
        l1, l2, l3, l4, m2, m3, m4, r2, r3, r4, None, None, None)

    theta2, _ = state
    theta3, theta4 = position_analysis(theta2, l1, l2, l3, l4, config)

    V = (m2 * g * r2 * np.sin(theta2)
         + m3 * g * (l2 * np.sin(theta2) + r3 * np.sin(theta3))
         + m4 * g * r4 * np.sin(theta4))
    return V


def mechanical_energy(state, l1, l2, l3, l4, m2=1.0, m3=1.0, m4=1.0,
                      r2=None, r3=None, r4=None,
                      I2=None, I3=None, I4=None, g=0.0, config='open'):
    """计算总机械能 E = ½·I_eff·ω₂² + V。"""
    theta2, omega2 = state
    I_eff = equivalent_inertia(theta2, l1, l2, l3, l4, m2, m3, m4,
                               r2, r3, r4, I2, I3, I4, config)
    V = potential_energy(state, l1, l2, l3, l4, m2, m3, m4,
                         r2, r3, r4, g, config)
    return 0.5 * I_eff * omega2**2 + V


def dynamics(t, state, l1, l2, l3, l4, m2=1.0, m3=1.0, m4=1.0,
            r2=None, r3=None, r4=None,
            I2=None, I3=None, I4=None,
            g=0.0, tau=0.0, config='open'):
    """返回状态时间导数 [dθ₂/dt, dω₂/dt]。

    运动方程：I_eff(θ₂)·α₂ + ½·I_eff'(θ₂)·ω₂² + dV/dθ₂ = τ

    I_eff' 用中心差分数值计算。
    """
    r2, r3, r4, I2, I3, I4 = _resolve_defaults(
        l1, l2, l3, l4, m2, m3, m4, r2, r3, r4, I2, I3, I4)

    theta2, omega2 = state

    # 当前 I_eff
    I_eff = equivalent_inertia(theta2, l1, l2, l3, l4, m2, m3, m4,
                               r2, r3, r4, I2, I3, I4, config)

    # I_eff' 数值中心差分
    h = 1e-7
    I_eff_p = equivalent_inertia(theta2 + h, l1, l2, l3, l4, m2, m3, m4,
                                 r2, r3, r4, I2, I3, I4, config)
    I_eff_m = equivalent_inertia(theta2 - h, l1, l2, l3, l4, m2, m3, m4,
                                 r2, r3, r4, I2, I3, I4, config)
    I_eff_prime = (I_eff_p - I_eff_m) / (2 * h)

    # 重力势能导数 dV/dθ₂
    if g != 0:
        theta3, theta4 = position_analysis(theta2, l1, l2, l3, l4, config)
        R3, R4 = velocity_ratios(theta2, theta3, theta4, l1, l2, l3, l4)
        dV_dtheta2 = (m2 * g * r2 * np.cos(theta2)
                      + m3 * g * (l2 * np.cos(theta2)
                                  + r3 * np.cos(theta3) * R3)
                      + m4 * g * r4 * np.cos(theta4) * R4)
    else:
        dV_dtheta2 = 0.0

    # 运动方程
    alpha2 = (tau - 0.5 * I_eff_prime * omega2**2 - dV_dtheta2) / I_eff

    return np.array([omega2, alpha2])
