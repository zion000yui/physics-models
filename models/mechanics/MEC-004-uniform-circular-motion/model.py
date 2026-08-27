"""MEC-004-uniform-circular-motion —— 模型定义（引擎无关）

匀速圆周运动：质点以恒定速率沿圆周运动，半径 R 和角速度 ω 均保持不变。

状态向量 state = [x, y, vx, vy]
    x  —— 水平位置
    y  —— 垂直位置
    vx —— 水平速度
    vy —— 垂直速度

动力学（一阶常微分方程）：
    dx/dt = vx
    dy/dt = vy
    dvx/dt = -ω² (x - xc)   （向心加速度，指向圆心）
    dvy/dt = -ω² (y - yc)

解析解（用于校验数值解的"金标准"）：
    θ(t) = θ₀ + ω t
    x(t) = xc + R cos θ(t)
    y(t) = yc + R sin θ(t)
    vx(t) = -R ω sin θ(t)
    vy(t) =  R ω cos θ(t)

初始状态约束（必须满足才是匀速圆周运动）：
    (x0 - xc)² + (y0 - yc)² = R²
    vx0 (x0 - xc) + vy0 (y0 - yc) = 0   （速度与半径正交）
    |v0| = R |ω|

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_initial_state(initial_state, R, omega, xc=0.0, yc=0.0,
                           tol=1e-6):
    """验证初始状态是否满足匀速圆周运动条件。

    参数
    ----
    initial_state : array_like, shape (4,)
        初始状态 [x0, y0, vx0, vy0]。
    R : float
        圆周半径（必须 > 0）。
    omega : float
        角速度（可正可负）。
    xc, yc : float, optional
        圆心坐标（默认原点）。
    tol : float, optional
        容差（默认 1e-6）。

    返回
    ----
    None
        条件满足时静默返回。

    抛出
    ----
    AssertionError
        任一条件不满足时给出明确错误信息。
    """
    x0, y0, vx0, vy0 = initial_state
    assert R > 0, f"半径 R 必须为正，当前 R={R}"
    assert omega != 0, "角速度 omega 不能为零"

    dx = x0 - xc
    dy = y0 - yc
    radius_sq = dx ** 2 + dy ** 2
    assert abs(radius_sq - R ** 2) < tol, \
        f"初始位置不在圆上：|r|²={radius_sq:.6f}，R²={R**2:.6f}"

    dot = vx0 * dx + vy0 * dy
    assert abs(dot) < tol, \
        f"初速度不与半径正交：v·r={dot:.6e}"

    v0 = np.hypot(vx0, vy0)
    assert abs(v0 - R * abs(omega)) < tol, \
        f"速率不等于 R|ω|：|v|={v0:.6f}，R|ω|={R*abs(omega):.6f}"


def dynamics(t, state, R=1.0, omega=1.0, xc=0.0, yc=0.0):
    """返回状态的时间导数 d(state)/dt = [dx/dt, dy/dt, dvx/dt, dvy/dt]。

    参数
    ----
    t : float
        当前时刻（匀速圆周运动显式不依赖 t，保留参数以统一接口）。
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    R : float, optional
        圆周半径（默认 1.0 m）。
    omega : float, optional
        角速度（默认 1.0 rad/s，可正可负）。
    xc, yc : float, optional
        圆心坐标（默认原点）。

    返回
    ----
    np.ndarray, shape (4,)
        [vx, vy, -ω²(x-xc), -ω²(y-yc)]
    """
    x, y, vx, vy = state
    ax = -omega ** 2 * (x - xc)
    ay = -omega ** 2 * (y - yc)
    return np.array([vx, vy, ax, ay])


def analytical(t, initial_state, R=1.0, omega=1.0, xc=0.0, yc=0.0):
    """匀速圆周运动解析解。

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    initial_state : array_like, shape (4,)
        初始状态 [x0, y0, vx0, vy0]。
    R : float, optional
        圆周半径（默认 1.0 m）。
    omega : float, optional
        角速度（默认 1.0 rad/s，可正可负）。
    xc, yc : float, optional
        圆心坐标（默认原点）。

    返回
    ----
    (x, y, vx, vy) : tuple
        x(t), y(t), vx(t), vy(t)，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    x0, y0, vx0, vy0 = initial_state

    # 从初始位置反推初始角度
    theta0 = np.arctan2(y0 - yc, x0 - xc)

    theta = theta0 + omega * t
    x = xc + R * np.cos(theta)
    y = yc + R * np.sin(theta)
    vx = -R * omega * np.sin(theta)
    vy =  R * omega * np.cos(theta)
    return x, y, vx, vy
