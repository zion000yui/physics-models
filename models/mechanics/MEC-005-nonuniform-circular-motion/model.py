"""MEC-005-nonuniform-circular-motion —— 模型定义（引擎无关）

非匀速圆周运动：质点沿固定半径 R 的圆周运动，角速度 ω 随时间变化。
这是 MEC-004 匀速圆周运动的自然推广——引入常数角加速度 α，使质点
在切向获得加速度，速率不再恒定。

状态向量 state = [x, y, vx, vy]
    x  —— 水平位置
    y  —— 垂直位置
    vx —— 水平速度
    vy —— 垂直速度

动力学（一阶常微分方程）：

    设瞬时角速度 ω(t) = ω₀ + α·t
    瞬时角度 θ(t) = θ₀ + ω₀·t + ½·α·t²

    位置：
        x(t) = xc + R·cos θ(t)
        y(t) = yc + R·sin θ(t)

    速度：
        vx(t) = -R·ω(t)·sin θ(t)
        vy(t) =  R·ω(t)·cos θ(t)

    加速度分解为法向（向心）+ 切向：

        法向加速度：a_n = R·ω(t)²，指向圆心
        切向加速度：a_t = R·α，沿运动方向（θ 增加方向）

    合成加速度：
        a_x = -R·ω(t)²·(x-xc)/R - R·α·(y-yc)/R
            = -ω(t)²·(x-xc) - α·(y-yc)

        a_y = -R·ω(t)²·(y-yc)/R + R·α·(x-xc)/R
            = -ω(t)²·(y-yc) + α·(x-xc)

    推导说明：
        令 r = (x-xc, y-yc) = (R·cosθ, R·sinθ)
        法向单位向量 n_hat = (-cosθ, -sinθ) = -r/R（指向圆心）
        切向单位向量 t_hat = (-sinθ, cosθ)（θ 增加方向）

        a = a_n·n_hat + a_t·t_hat
          = R·ω²·(-cosθ, -sinθ) + R·α·(-sinθ, cosθ)
          = (-R·ω²·cosθ - R·α·sinθ, -R·ω²·sinθ + R·α·cosθ)

        代入 x-xc = R·cosθ, y-yc = R·sinθ 得：
            a_x = -ω²·(x-xc) - α·(y-yc)
            a_y = -ω²·(y-yc) + α·(x-xc)

解析解（用于校验数值解的"金标准"）：
    θ(t) = θ₀ + ω₀·t + ½·α·t²
    ω(t) = ω₀ + α·t
    x(t) = xc + R·cos θ(t)
    y(t) = yc + R·sin θ(t)
    vx(t) = -R·ω(t)·sin θ(t)
    vy(t) =  R·ω(t)·cos θ(t)

初始状态约束（必须满足才是圆周运动）：
    (x₀-xc)² + (y₀-yc)² = R²
    vx₀·(x₀-xc) + vy₀·(y₀-yc) = 0   （速度与半径正交）
    |v₀| = R·|ω₀|                    （速率与初始角速度匹配）

退化情形：α=0 时，模型退化为 MEC-004 匀速圆周运动。

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_initial_state(initial_state, R, omega0, xc=0.0, yc=0.0,
                           tol=1e-6):
    """验证初始状态是否满足圆周运动条件。

    参数
    ----
    initial_state : array_like, shape (4,)
        初始状态 [x0, y0, vx0, vy0]。
    R : float
        圆周半径（必须 > 0）。
    omega0 : float
        初始角速度（可正可负）。
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
    assert omega0 != 0, "初始角速度 omega0 不能为零"

    dx = x0 - xc
    dy = y0 - yc
    radius_sq = dx ** 2 + dy ** 2
    assert abs(radius_sq - R ** 2) < tol, \
        f"初始位置不在圆上：|r|²={radius_sq:.6f}，R²={R**2:.6f}"

    dot = vx0 * dx + vy0 * dy
    assert abs(dot) < tol, \
        f"初速度不与半径正交：v·r={dot:.6e}"

    v0 = np.hypot(vx0, vy0)
    assert abs(v0 - R * abs(omega0)) < tol, \
        f"速率不等于 R|ω₀|：|v|={v0:.6f}，R|ω₀|={R*abs(omega0):.6f}"


def dynamics(t, state, R=1.0, omega0=1.0, alpha=0.0, xc=0.0, yc=0.0):
    """返回状态的时间导数 d(state)/dt = [dx/dt, dy/dt, dvx/dt, dvy/dt]。

    参数
    ----
    t : float
        当前时刻。
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    R : float, optional
        圆周半径（默认 1.0 m）。
    omega0 : float, optional
        初始角速度（默认 1.0 rad/s）。
    alpha : float, optional
        常数角加速度（默认 0.0 rad/s²）。
    xc, yc : float, optional
        圆心坐标（默认原点）。

    返回
    ----
    np.ndarray, shape (4,)
        [vx, vy, ax, ay]
    """
    x, y, vx, vy = state
    omega_t = omega0 + alpha * t
    ax = -omega_t ** 2 * (x - xc) - alpha * (y - yc)
    ay = -omega_t ** 2 * (y - yc) + alpha * (x - xc)
    return np.array([vx, vy, ax, ay])


def analytical(t, initial_state, R=1.0, omega0=1.0, alpha=0.0,
               xc=0.0, yc=0.0):
    """非匀速圆周运动解析解。

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    initial_state : array_like, shape (4,)
        初始状态 [x0, y0, vx0, vy0]。
    R : float, optional
        圆周半径（默认 1.0 m）。
    omega0 : float, optional
        初始角速度（默认 1.0 rad/s）。
    alpha : float, optional
        常数角加速度（默认 0.0 rad/s²）。
    xc, yc : float, optional
        圆心坐标（默认原点）。

    返回
    ----
    (x, y, vx, vy) : tuple
        x(t), y(t), vx(t), vy(t)，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    x0, y0, vx0, vy0 = initial_state
    theta0 = np.arctan2(y0 - yc, x0 - xc)
    theta = theta0 + omega0 * t + 0.5 * alpha * t ** 2
    omega_t = omega0 + alpha * t
    x = xc + R * np.cos(theta)
    y = yc + R * np.sin(theta)
    vx = -R * omega_t * np.sin(theta)
    vy =  R * omega_t * np.cos(theta)
    return x, y, vx, vy
