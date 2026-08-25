"""MEC-001-free-particle —— 模型定义（引擎无关）

自由质点：不受外力、无约束的点质量。

状态向量 state = [x, v]
    x —— 位置（一维，标量）
    v —— 速度（一维，标量）

动力学（一阶常微分方程）：
    dx/dt = v
    dv/dt = 0      （无外力，加速度恒为 0）

解析解（用于校验数值解的"金标准"）：
    x(t) = x0 + v0 * t
    v(t) = v0

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def dynamics(t, state):
    """返回状态的时间导数 d(state)/dt = [dx/dt, dv/dt]。

    参数
    ----
    t : float
        当前时刻（自由质点显式不依赖 t，保留参数以统一接口）。
    state : array_like, shape (2,)
        当前状态 [x, v]。

    返回
    ----
    np.ndarray, shape (2,)
        [v, 0]
    """
    x, v = state
    return np.array([v, 0.0])


def analytical(t, x0, v0):
    """自由质点解析解。

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    x0 : float
        初始位置。
    v0 : float
        初始速度。

    返回
    ----
    (x, v) : tuple
        x(t) 与 v(t)，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    x = x0 + v0 * t
    v = np.full_like(t, v0, dtype=float)
    return x, v