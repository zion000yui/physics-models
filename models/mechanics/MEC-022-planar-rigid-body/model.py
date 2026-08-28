"""MEC-022-planar-rigid-body —— 模型定义（引擎无关）

平面刚体运动（planar rigid body motion）：刚体在 2D 平面内做自由运动，
具有 3 个自由度——2D 平动（质心运动 x_cm, y_cm）+ 1D 绕质心轴（z 轴）转动（θ）。
当外力不通过质心时，力矩 τ = r × F 将平动与转动耦合。

这是 2D 框架下的最大自由度刚体模型，不是 3D/6DOF 自由刚体。
完整 3D 欧拉方程和惯性张量留到后续独立模型。

状态向量 state = [x_cm, y_cm, vx_cm, vy_cm, theta, omega]
    x_cm, y_cm  —— 质心位置（m）
    vx_cm, vy_cm —— 质心速度（m/s）
    theta        —— 绕质心的转角（rad）
    omega        —— 角速度（rad/s）

参数：
    m  —— 刚体质量（kg，m > 0）
    I  —— 绕质心轴的转动惯量（kg·m²，I > 0）
    Fx, Fy —— 合外力分量（N）
    rx, ry —— 外力作用点相对于质心的位置（m）

    外力矩由力臂计算：tau = rx·Fy - ry·Fx
    当 Fx, Fy, rx, ry 均为零时，无外力且无力矩。

动力学（一阶常微分方程）：

    平动（质心运动定理）：m·a_cm = F_ext
        dvx_cm/dt = Fx / m
        dvy_cm/dt = Fy / m

    转动（绕质心轴）：I·α = tau_cm
        domega/dt = tau / I

    其中 tau = rx·Fy - ry·Fx（2D 叉积 τ = r × F 的 z 分量）

    因此：
        dx_cm/dt = vx_cm
        dy_cm/dt = vy_cm
        dtheta/dt = omega
        dvx_cm/dt = Fx / m
        dvy_cm/dt = Fy / m
        domega/dt = (rx·Fy - ry·Fx) / I

核心物理概念——力矩的来源：

    MEC-022 的关键新增概念不是简单地同时有"力"和"力矩"两个独立参数，
    而是：同一个外力 F 作用于偏离质心的位置 r 时，
    同时产生质心加速度（a = F/m）和角加速度（α = (r×F)/I）。
    力矩不是独立的黑盒参数，而是由力臂 r 和力 F 共同决定的。

    当力通过质心时（r = 0），力矩为零，无旋转，退化为 MEC-020。
    当外力为零但存在初始角速度时，质心不动，退化为 MEC-021 纯转动。

解析解（恒力 + 恒力臂，即恒力和恒力矩）：

    x_cm(t) = x0 + vx0·t + ½·(Fx/m)·t²
    y_cm(t) = y0 + vy0·t + ½·(Fy/m)·t²
    theta(t) = theta0 + omega0·t + ½·(tau/I)·t²
    其中 tau = rx·Fy - ry·Fx

退化关系：

    1. 力通过质心（rx=ry=0）→ tau=0，无旋转 → 退化为 MEC-020
    2. 无外力（Fx=Fy=0）且 v_cm0=0 → 质心不动，纯绕质心转动 → 退化为 MEC-021
    3. 无外力且无初始角速度 → 自由刚体匀速平动 + 无转动 → MEC-020（v=const）

守恒量：

    无外力时动量守恒：P = m·v_cm = const
    无外力矩时角动量守恒：L = I·omega = const
    保守力时机械能守恒

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(m=1.0, I=1.0):
    """验证基本物理参数合法性。

    参数
    ----
    m : float
        刚体质量（必须 > 0）。
    I : float
        绕质心轴的转动惯量（必须 > 0）。

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


def torque_from_force(Fx=0.0, Fy=0.0, rx=0.0, ry=0.0):
    """从外力和力臂计算力矩 tau = rx·Fy - ry·Fx。

    这是 2D 中叉积 τ_z = (r × F)_z 的标量形式。

    参数
    ----
    Fx, Fy : float
        外力分量（N）。
    rx, ry : float
        外力作用点相对于质心的位置（m）。

    返回
    ----
    float
        绕质心轴的力矩 tau（N·m）。
    """
    return rx * Fy - ry * Fx


def momentum(state, m=1.0):
    """计算动量 P = m·v_cm。

    参数
    ----
    state : array_like, shape (6,)
        状态 [x, y, vx, vy, theta, omega]。
    m : float
        质量。

    返回
    ----
    np.ndarray, shape (2,)
        动量 [Px, Py]。
    """
    _, _, vx, vy, _, _ = state
    return np.array([m * vx, m * vy])


def angular_momentum(state, I=1.0):
    """计算角动量 L = I·omega。

    参数
    ----
    state : array_like, shape (6,)
        状态 [x, y, vx, vy, theta, omega]。
    I : float
        转动惯量。

    返回
    ----
    float
        角动量 L。
    """
    _, _, _, _, _, omega = state
    return I * omega


def mechanical_energy(state, m=1.0, I=1.0, Fx=0.0, Fy=0.0,
                     rx=0.0, ry=0.0, x_ref=0.0, y_ref=0.0):
    """计算机械能。

    对于恒力（保守力），E = ½mv² + ½Iω² - F·(r - r_ref)。
    无外力时简化为动能 E = ½mv² + ½Iω²。

    参数
    ----
    state : array_like, shape (6,)
        状态 [x, y, vx, vy, theta, omega]。
    m, I, Fx, Fy, rx, ry : float
        物理参数。
    x_ref, y_ref : float
        势能参考点。

    返回
    ----
    float
        机械能。
    """
    x, y, vx, vy, theta, omega = state
    ke = 0.5 * m * (vx ** 2 + vy ** 2) + 0.5 * I * omega ** 2
    pe = -(Fx * (x - x_ref) + Fy * (y - y_ref))
    return ke + pe


def dynamics(t, state, m=1.0, I=1.0, Fx=0.0, Fy=0.0, rx=0.0, ry=0.0):
    """返回状态的时间导数 d(state)/dt。

    平动：m·a_cm = F_ext
    转动：I·alpha = tau，其中 tau = rx·Fy - ry·Fx

    参数
    ----
    t : float
        当前时刻（恒力不依赖 t，保留以统一接口）。
    state : array_like, shape (6,)
        状态 [x, y, vx, vy, theta, omega]。
    m : float, optional
        刚体质量（默认 1.0 kg）。
    I : float, optional
        转动惯量（默认 1.0 kg·m²）。
    Fx, Fy : float, optional
        外力分量（默认 0.0 N）。
    rx, ry : float, optional
        外力作用点相对于质心的位置（默认 0.0 m，即力通过质心）。

    返回
    ----
    np.ndarray, shape (6,)
        [vx, vy, Fx/m, Fy/m, omega, tau/I]
    """
    _, _, vx, vy, _, omega = state
    tau = torque_from_force(Fx, Fy, rx, ry)
    return np.array([vx, vy, Fx / m, Fy / m, omega, tau / I])


def analytical(t, initial_state, m=1.0, I=1.0,
               Fx=0.0, Fy=0.0, rx=0.0, ry=0.0):
    """恒力+恒力臂的平面刚体解析解。

    参数
    ----
    t : float 或 array_like
        时间点。
    initial_state : array_like, shape (6,)
        初始状态 [x0, y0, vx0, vy0, theta0, omega0]。
    m, I, Fx, Fy, rx, ry : float
        物理参数。

    返回
    ----
    (x, y, vx, vy, theta, omega) : tuple
        六个状态分量，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    x0, y0, vx0, vy0, theta0, omega0 = initial_state
    ax = Fx / m
    ay = Fy / m
    tau = torque_from_force(Fx, Fy, rx, ry)
    alpha = tau / I
    x = x0 + vx0 * t + 0.5 * ax * t ** 2
    y = y0 + vy0 * t + 0.5 * ay * t ** 2
    vx = vx0 + ax * t
    vy = vy0 + ay * t
    theta = theta0 + omega0 * t + 0.5 * alpha * t ** 2
    omega = omega0 + alpha * t
    return x, y, vx, vy, theta, omega
