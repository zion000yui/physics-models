"""MEC-002-forced-particle —— 模型定义（引擎无关）

受恒定外力作用的质点：质量为 m 的质点，受恒定外力 F 作用
（F 为标量常数，方向和大小都不随时间变化）。

状态向量 state = [x, v]
    x —— 位置（一维，标量）
    v —— 速度（一维，标量）

动力学（一阶常微分方程）：
    dx/dt = v
    dv/dt = F / m    （恒定加速度 a = F/m）

解析解（用于校验数值解的"金标准"）：
    x(t) = x0 + v0 * t + 0.5 * (F/m) * t²
    v(t) = v0 + (F/m) * t

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def dynamics(t, state, F=1.0, m=1.0):
    """返回状态的时间导数 d(state)/dt = [dx/dt, dv/dt]。

    参数
    ----
    t : float
        当前时刻（受恒定外力的质点显式不依赖 t，保留参数以统一接口）。
    state : array_like, shape (2,)
        当前状态 [x, v]。
    F : float, optional
        恒定外力（默认 1.0）。
    m : float, optional
        质点质量（默认 1.0）。

    返回
    ----
    np.ndarray, shape (2,)
        [v, F/m]
    """
    x, v = state
    return np.array([v, F / m])


def analytical(t, x0, v0, F=1.0, m=1.0):
    """受恒定外力作用的质点解析解。

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    x0 : float
        初始位置。
    v0 : float
        初始速度。
    F : float, optional
        恒定外力（默认 1.0）。
    m : float, optional
        质点质量（默认 1.0）。

    返回
    ----
    (x, v) : tuple
        x(t) 与 v(t)，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    a = F / m
    x = x0 + v0 * t + 0.5 * a * t ** 2
    v = v0 + a * t
    return x, v
