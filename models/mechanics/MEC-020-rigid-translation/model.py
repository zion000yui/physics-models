"""MEC-020-rigid-translation —— 模型定义（引擎无关）

刚体平动（rigid body translation）：刚体在合外力作用下做纯平动（无旋转），
所有质点具有相同速度和加速度。根据质心运动定理，刚体平动等价于将全部质量
集中于质心的质点运动：M·a_cm = F_ext。

这是 020 号段（刚体动力学）的第一个模型，引入"刚体"概念和质心运动定理，
为后续 MEC-021（定轴转动）、MEC-024（纯滚动）、MEC-022（平面刚体）做铺垫。

状态向量 state = [x_cm, y_cm, vx_cm, vy_cm]
    x_cm, y_cm —— 质心位置（m）
    vx_cm, vy_cm —— 质心速度（m/s）

参数：
    m  —— 刚体质量（kg，m > 0）
    Fx —— x 方向合外力（N，可为任意实数）
    Fy —— y 方向合外力（N，可为任意实数）

动力学（一阶常微分方程）：

    质心运动定理：m·a_cm = F_ext

    因此：
        dx_cm/dt = vx_cm
        dy_cm/dt = vy_cm
        dvx_cm/dt = Fx / m
        dvy_cm/dt = Fy / m

解析解（恒力，用于校验数值解的"金标准"）：

    x_cm(t) = x0 + vx0·t + ½·(Fx/m)·t²
    y_cm(t) = y0 + vy0·t + ½·(Fy/m)·t²
    vx_cm(t) = vx0 + (Fx/m)·t
    vy_cm(t) = vy0 + (Fy/m)·t

退化关系：

    - F = 0（无外力）→ 退化为 MEC-001（自由质点，匀速直线运动）
    - F = const（恒力）→ 退化为 MEC-002（受力质点，匀加速运动，2D 形式）
    - F = (0, -mg)（重力）→ 退化为 MEC-003（抛体运动）

守恒量：

    无外力时（Fx=Fy=0），动量守恒：P = m·v_cm = const

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(m=1.0, Fx=0.0, Fy=0.0):
    """验证基本物理参数合法性。

    参数
    ----
    m : float
        刚体质量（必须 > 0）。
    Fx : float
        x 方向合外力（可为任意实数）。
    Fy : float
        y 方向合外力（可为任意实数）。

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


def momentum(state, m=1.0):
    """计算动量 P = m·v_cm。

    参数
    ----
    state : array_like, shape (4,)
        当前状态 [x_cm, y_cm, vx_cm, vy_cm]。
    m : float
        刚体质量。

    返回
    ----
    np.ndarray, shape (2,)
        动量 [Px, Py]。
    """
    _, _, vx, vy = state
    return np.array([m * vx, m * vy])


def dynamics(t, state, m=1.0, Fx=0.0, Fy=0.0):
    """返回状态的时间导数 d(state)/dt。

    质心运动定理：m·a_cm = F_ext

    参数
    ----
    t : float
        当前时刻（恒力下不依赖 t，保留参数以统一接口）。
    state : array_like, shape (4,)
        当前状态 [x_cm, y_cm, vx_cm, vy_cm]。
    m : float, optional
        刚体质量（默认 1.0 kg）。
    Fx : float, optional
        x 方向合外力（默认 0.0 N）。
    Fy : float, optional
        y 方向合外力（默认 0.0 N）。

    返回
    ----
    np.ndarray, shape (4,)
        [vx_cm, vy_cm, Fx/m, Fy/m]
    """
    _, _, vx, vy = state
    return np.array([vx, vy, Fx / m, Fy / m])


def analytical(t, initial_state, m=1.0, Fx=0.0, Fy=0.0):
    """刚体平动解析解（恒力）。

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    initial_state : array_like, shape (4,)
        初始状态 [x0, y0, vx0, vy0]。
    m : float, optional
        刚体质量（默认 1.0 kg）。
    Fx : float, optional
        x 方向合外力（默认 0.0 N）。
    Fy : float, optional
        y 方向合外力（默认 0.0 N）。

    返回
    ----
    (x, y, vx, vy) : tuple
        x_cm(t), y_cm(t), vx_cm(t), vy_cm(t)，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    x0, y0, vx0, vy0 = initial_state
    ax = Fx / m
    ay = Fy / m
    x = x0 + vx0 * t + 0.5 * ax * t ** 2
    y = y0 + vy0 * t + 0.5 * ay * t ** 2
    vx = vx0 + ax * t
    vy = vy0 + ay * t
    return x, y, vx, vy
