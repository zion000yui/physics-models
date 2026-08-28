"""MEC-006-central-force-hooke —— 模型定义（引擎无关）

中心力场运动（胡克型向心力）：质点在指向原点的线性 restoring force
F = -k·r 作用下运动，轨迹通常为椭圆。这是系列第一次引入真实的力，
质量 m 成为显式参数，需要验证角动量守恒和机械能守恒。

状态向量 state = [x, y, vx, vy]
    x  —— 水平位置
    y  —— 垂直位置
    vx —— 水平速度
    vy —— 垂直速度

动力学（一阶常微分方程）：

    胡克型向心力：F = -k·r = -k·(x, y)
    牛顿第二定律：m·a = F
    因此：
        dvx/dt = -(k/m)·x
        dvy/dt = -(k/m)·y

    令 ω₀ = √(k/m)，则 x、y 方向各自独立满足简谐振动方程：
        d²x/dt² + ω₀²·x = 0
        d²y/dt² + ω₀²·y = 0

解析解（用于校验数值解的"金标准"）：

    由于 x(t)、y(t) 各自独立解耦为简谐振动，可直接写出：
        x(t) = x0·cos(ω₀t) + (vx0/ω₀)·sin(ω₀t)
        y(t) = y0·cos(ω₀t) + (vy0/ω₀)·sin(ω₀t)

    速度对应求导：
        vx(t) = -x0·ω₀·sin(ω₀t) + vx0·cos(ω₀t)
        vy(t) = -y0·ω₀·sin(ω₀t) + vy0·cos(ω₀t)

守恒量工具函数：

    角动量（绕原点）：
        L = m·(x·vy - y·vx)

    机械能：
        E = ½·m·(vx² + vy²) + ½·k·(x² + y²)

初始状态约束：
    任意 (x0, y0, vx0, vy0) 都是合法初始状态（这是与 MEC-004/005 的关键区别）。
    仅要求 k > 0, m > 0。

退化情形：
    当初始条件恰好满足 |r0| 恒定、v0⊥r0、|v0| = ω₀·|r0| 时，
    轨迹退化为圆形（对应 MEC-004 的匀速圆周运动，但物理起源不同）。

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(k=1.0, m=1.0):
    """验证基本物理参数合法性。

    参数
    ----
    k : float
        弹性系数（必须 > 0）。
    m : float
        质量（必须 > 0）。

    返回
    ----
    None
        条件满足时静默返回。

    抛出
    ----
    AssertionError
        参数不合法时给出明确错误信息。
    """
    assert k > 0, f"弹性系数 k 必须为正，当前 k={k}"
    assert m > 0, f"质量 m 必须为正，当前 m={m}"


def angular_momentum(state, m=1.0):
    """计算绕原点的角动量 L = m·(x·vy - y·vx)。

    参数
    ----
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    m : float, optional
        质量（默认 1.0）。

    返回
    ----
    float
        角动量 L。
    """
    x, y, vx, vy = state
    return m * (x * vy - y * vx)


def mechanical_energy(state, k=1.0, m=1.0):
    """计算机械能 E = ½m(vx²+vy²) + ½k(x²+y²)。

    参数
    ----
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    k : float, optional
        弹性系数（默认 1.0）。
    m : float, optional
        质量（默认 1.0）。

    返回
    ----
    float
        机械能 E。
    """
    x, y, vx, vy = state
    return 0.5 * m * (vx ** 2 + vy ** 2) + 0.5 * k * (x ** 2 + y ** 2)


def dynamics(t, state, k=1.0, m=1.0):
    """返回状态的时间导数 d(state)/dt = [dx/dt, dy/dt, dvx/dt, dvy/dt]。

    参数
    ----
    t : float
        当前时刻（胡克力显式不依赖 t，保留参数以统一接口）。
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    k : float, optional
        弹性系数（默认 1.0 N/m）。
    m : float, optional
        质量（默认 1.0 kg）。

    返回
    ----
    np.ndarray, shape (4,)
        [vx, vy, -(k/m)·x, -(k/m)·y]
    """
    x, y, vx, vy = state
    ax = -(k / m) * x
    ay = -(k / m) * y
    return np.array([vx, vy, ax, ay])


def analytical(t, initial_state, k=1.0, m=1.0):
    """胡克型中心力场解析解。

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    initial_state : array_like, shape (4,)
        初始状态 [x0, y0, vx0, vy0]。
    k : float, optional
        弹性系数（默认 1.0 N/m）。
    m : float, optional
        质量（默认 1.0 kg）。

    返回
    ----
    (x, y, vx, vy) : tuple
        x(t), y(t), vx(t), vy(t)，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    x0, y0, vx0, vy0 = initial_state
    omega0 = np.sqrt(k / m)
    cos_wt = np.cos(omega0 * t)
    sin_wt = np.sin(omega0 * t)
    x = x0 * cos_wt + (vx0 / omega0) * sin_wt
    y = y0 * cos_wt + (vy0 / omega0) * sin_wt
    vx = -x0 * omega0 * sin_wt + vx0 * cos_wt
    vy = -y0 * omega0 * sin_wt + vy0 * cos_wt
    return x, y, vx, vy
