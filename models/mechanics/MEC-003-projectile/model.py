"""MEC-003-projectile —— 模型定义（引擎无关）

抛体运动：质点仅在重力作用下在二维平面内运动。

状态向量 state = [x, y, vx, vy]
    x  —— 水平位置
    y  —— 垂直位置
    vx —— 水平速度
    vy —— 垂直速度

动力学（一阶常微分方程）：
    dx/dt = vx
    dy/dt = vy
    dvx/dt = 0      （水平方向无外力，速度恒定）
    dvy/dt = -g     （竖直方向受恒定重力加速度向下）

解析解（用于校验数值解的"金标准"）：
    x(t) = x0 + vx0 * t
    y(t) = y0 + vy0 * t - 0.5 * g * t^2
    vx(t) = vx0
    vy(t) = vy0 - g * t

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def dynamics(t, state, g=9.81):
    """返回状态的时间导数 d(state)/dt = [dx/dt, dy/dt, dvx/dt, dvy/dt]。

    参数
    ----
    t : float
        当前时刻（抛体运动显式不依赖 t，保留参数以统一接口）。
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    g : float, optional
        重力加速度（默认 9.81 m/s²）。

    返回
    ----
    np.ndarray, shape (4,)
        [vx, vy, 0, -g]
    """
    x, y, vx, vy = state
    return np.array([vx, vy, 0.0, -g])


def analytical(t, initial_state, g=9.81):
    """抛体运动解析解。

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    initial_state : array_like, shape (4,)
        初始状态 [x0, y0, vx0, vy0]。
    g : float, optional
        重力加速度（默认 9.81 m/s²）。

    返回
    ----
    (x, y, vx, vy) : tuple
        x(t), y(t), vx(t), vy(t)，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    x0, y0, vx0, vy0 = initial_state
    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t ** 2
    vx = np.full_like(t, vx0, dtype=float)
    vy = vy0 - g * t
    return x, y, vx, vy
