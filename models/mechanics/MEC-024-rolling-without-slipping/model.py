"""MEC-024-rolling-without-slipping —— 模型定义（引擎无关）

纯滚动（rolling without slipping）：圆形刚体（球、圆柱等）在斜面上做纯滚动。
通过运动学约束 v_cm = R·ω 耦合平动（MEC-020）和转动（MEC-021）。
是 MEC-020 和 MEC-021 的自然综合模型。

状态向量 state = [x_cm, theta, v_cm, omega]
    x_cm  —— 质心沿斜面的位移（m）
    theta —— 转角（rad，滚动方向为正）
    v_cm  —— 质心速度（m/s）
    omega —— 角速度（rad/s）

参数：
    m   —— 刚体质量（kg，m > 0）
    I   —— 绕质心轴的转动惯量（kg·m²，I > 0）
    R   —— 半径（m，R > 0）
    g   —— 重力加速度（m/s²，g > 0）
    alpha —— 斜面倾角（rad，alpha ≥ 0）

运动学约束（纯滚动条件）：

    v_cm = R · ω

    这是几何约束，不是力。静摩擦力 f 是维持此约束的约束力，
    理想情况下静摩擦不做功（接触点瞬时速度为零）。
    纯滚动 ≠ 无摩擦：无摩擦时刚体只滑动不旋转，不满足 v_cm = R·ω。

动力学耦合：

    平动方程（沿斜面）：m · a = m·g·sin(α) - f
    转动方程（绕质心）：I · α_rot = f · R
    约束：a = R · α_rot

    联立求解（消去 f 和 α_rot）：
        m·a = m·g·sin(α) - I·a/R²
        a·(m + I/R²) = m·g·sin(α)
        a = g·sin(α) / (1 + I/(mR²))

    有效质量：m_eff = m + I/R²（平动+转动的惯性耦合）

解析解（恒加速度）：

    x_cm(t) = x0 + v0·t + ½·a·t²
    v_cm(t) = v0 + a·t
    theta(t) = theta0 + (v0/R)·t + ½·(a/R)·t²
    omega(t) = v0/R + (a/R)·t

    其中 a = g·sin(α) / (1 + I/(mR²))

机械能：

    E = ½·m·v_cm² + ½·I·ω² + m·g·h
    其中 h = -x_cm·sin(α)（沿斜面下降时高度降低）

    纯滚动时静摩擦不做功，机械能守恒。
    （滑动摩擦才做负功并耗散机械能。）

退化关系：

    取消滚动约束、去除转动自由度后，平动部分退化为 MEC-020：
        去除约束 v_cm = R·ω 和转动方程，
        平动方程变为 m·a = m·g·sin(α)，即 a = g·sin(α)，与 MEC-020 一致。

    注意：不将 I→0 或 I→∞ 作为退化。I→0 时转动动能→0 但约束仍在；
    I→∞ 时 a→0 且 ω→0，物体不动。这些不是严格退化。

典型转动惯量比值 I/(mR²)：

    实心球：    I = (2/5)mR²,  I/(mR²) = 2/5,  a = (5/7)g·sin(α)
    实心圆柱：  I = (1/2)mR²,  I/(mR²) = 1/2,  a = (2/3)g·sin(α)
    薄壁圆筒：  I = mR²,       I/(mR²) = 1,    a = (1/2)g·sin(α)

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(m=1.0, I=1.0, R=1.0, g=9.81, alpha=0.0):
    """验证基本物理参数合法性。

    参数
    ----
    m : float
        刚体质量（必须 > 0）。
    I : float
        转动惯量（必须 > 0）。
    R : float
        半径（必须 > 0）。
    g : float
        重力加速度（必须 > 0）。
    alpha : float
        斜面倾角（必须 ≥ 0）。

    返回
    ----
    None
        条件满足时静默返回。

    抛出
    ----
    AssertionError
        参数不合法时给出明确错误信息。
    """
    assert m > 0, f"质量 m 必须为正，当前 m={m}"
    assert I > 0, f"转动惯量 I 必须为正，当前 I={I}"
    assert R > 0, f"半径 R 必须为正，当前 R={R}"
    assert g > 0, f"重力加速度 g 必须为正，当前 g={g}"
    assert alpha >= 0, f"斜面倾角 alpha 必须非负，当前 alpha={alpha}"


def effective_mass(m=1.0, I=1.0, R=1.0):
    """计算有效质量 m_eff = m + I/R²。

    有效质量体现了平动惯性和转动惯性对加速度的共同影响。

    参数
    ----
    m, I, R : float
        质量、转动惯量、半径。

    返回
    ----
    float
        有效质量 m_eff。
    """
    return m + I / R ** 2


def acceleration(g=9.81, m=1.0, I=1.0, R=1.0, alpha=0.0):
    """计算纯滚动加速度 a = g·sin(α) / (1 + I/(mR²))。

    参数
    ----
    g, m, I, R, alpha : float
        物理参数。

    返回
    ----
    float
        质心加速度 a（m/s²）。
    """
    return g * np.sin(alpha) / (1.0 + I / (m * R ** 2))


def mechanical_energy(state, m=1.0, I=1.0, R=1.0, g=9.81, alpha=0.0):
    """计算机械能 E = ½mv² + ½Iω² + mgh。

    纯滚动时静摩擦不做功，机械能守恒。

    参数
    ----
    state : array_like, shape (4,)
        状态 [x_cm, theta, v_cm, omega]。
    m, I, R, g, alpha : float
        物理参数。

    返回
    ----
    float
        机械能 E。
    """
    x_cm, _, v_cm, omega = state
    ke = 0.5 * m * v_cm ** 2 + 0.5 * I * omega ** 2
    h = -x_cm * np.sin(alpha)
    pe = m * g * h
    return ke + pe


def dynamics(t, state, m=1.0, I=1.0, R=1.0, g=9.81, alpha=0.0):
    """返回状态的时间导数 d(state)/dt。

    纯滚动约束已代入：a = g·sin(α)/(1+I/(mR²))，ω = v/R。

    参数
    ----
    t : float
        当前时刻（不依赖 t，保留以统一接口）。
    state : array_like, shape (4,)
        状态 [x_cm, theta, v_cm, omega]。
    m, I, R, g, alpha : float
        物理参数。

    返回
    ----
    np.ndarray, shape (4,)
        [v_cm, omega, a, a/R]
    """
    x_cm, theta, v_cm, omega = state
    a = acceleration(g, m, I, R, alpha)
    return np.array([v_cm, omega, a, a / R])


def analytical(t, initial_state, m=1.0, I=1.0, R=1.0, g=9.81, alpha=0.0):
    """纯滚动解析解（恒加速度）。

    参数
    ----
    t : float 或 array_like
        时间点。
    initial_state : array_like, shape (4,)
        初始状态 [x0, theta0, v0, omega0]。
    m, I, R, g, alpha : float
        物理参数。

    返回
    ----
    (x_cm, theta, v_cm, omega) : tuple
        质心位移、转角、质心速度、角速度，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    x0, theta0, v0, omega0 = initial_state
    a = acceleration(g, m, I, R, alpha)
    x_cm = x0 + v0 * t + 0.5 * a * t ** 2
    v_cm = v0 + a * t
    theta = theta0 + omega0 * t + 0.5 * (a / R) * t ** 2
    omega = omega0 + (a / R) * t
    return x_cm, theta, v_cm, omega
