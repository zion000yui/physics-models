"""MEC-009-particle-with-drag —— 模型定义（引擎无关）

速度相关阻力下的质点运动：质点在重力场中运动，同时受速度相关阻力。
覆盖两种阻力模型：线性阻力（Stokes 阻力，F = -b·v）和二次阻力
（Newton 阻力，F = -c·|v|·v）。当 b=0、c=0 时退化为 MEC-003 抛体运动。

状态向量 state = [x, y, vx, vy]
    x  —— 水平位置
    y  —— 垂直位置
    vx —— 水平速度
    vy —— 垂直速度

参数：
    g —— 重力加速度（m/s²，向下为正方向，即 dvy/dt 中含 -g）
    b —— 线性阻力系数（力 = -b·v，b ≥ 0）
    c —— 二次阻力系数（力 = -c·|v|·v，c ≥ 0）
    m —— 质点质量（kg，m > 0）

动力学（一阶常微分方程）：

    合力 = 重力 + 线性阻力 + 二次阻力
        F = -m·g·ŷ - b·v - c·|v|·v

    加速度：
        ax = -(b/m)·vx - (c/m)·|v|·vx
        ay = -g - (b/m)·vy - (c/m)·|v|·vy
    其中 |v| = √(vx² + vy²)

    因此：
        dx/dt = vx
        dy/dt = vy
        dvx/dt = -(b/m)·vx - (c/m)·|v|·vx
        dvy/dt = -g - (b/m)·vy - (c/m)·|v|·vy

与 MEC-003 的关系：
    当 b=0, c=0 时，阻力消失，模型精确退化为 MEC-003 抛体运动：
        dvx/dt = 0, dvy/dt = -g

解析解（用于校验数值解的"金标准"）：

    1. 无阻力（b=0, c=0）：标准抛体运动
        x(t) = x0 + vx0·t
        y(t) = y0 + vy0·t - ½·g·t²
        vx(t) = vx0
        vy(t) = vy0 - g·t

    2. 纯线性阻力（c=0, b>0）：闭式解析解
        令 γ = b/m（阻力衰减率）
        vx(t) = vx0·e^(-γt)
        vy(t) = (vy0 + g/γ)·e^(-γt) - g/γ
        x(t) = x0 + (vx0/γ)·(1 - e^(-γt))
        y(t) = y0 + ((vy0 + g/γ)/γ)·(1 - e^(-γt)) - (g/γ)·t

    3. 纯二次阻力，一维垂直运动（b=0, c>0, vx0=0）：分段解析解
        终端速度 v_t = √(m·g/c)
        上升阶段（vy > 0）：
            θ(t) = θ0 - (g/v_t)·t,  θ0 = arctan(vy0/v_t)
            vy(t) = v_t·tan(θ(t))
            y(t) = y0 + (v_t²/g)·ln(cos(θ(t))/cos(θ0))
        下降阶段（vy < 0）：
            φ(t) = (g/v_t)·(t - t_apex)
            vy(t) = -v_t·tanh(φ(t))
            y(t) = y(t_apex) - (v_t²/g)·ln(cosh(φ(t)))

    4. 一般二维二次阻力或混合阻力：无闭式解析解，抛出 ValueError。

终态速度（terminal velocity）：
    当 dv/dt = 0 时，质点达到终态速度（方向向下）：
        纯线性：v_t = m·g/b
        纯二次：v_t = √(m·g/c)
        混合：v_t = (-b + √(b² + 4cmg)) / (2c)

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(g=9.81, b=0.0, c=0.0, m=1.0):
    """验证基本物理参数合法性。

    参数
    ----
    g : float
        重力加速度（必须 ≥ 0）。
    b : float
        线性阻力系数（必须 ≥ 0）。
    c : float
        二次阻力系数（必须 ≥ 0）。
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
    assert g >= 0, f"重力加速度 g 必须非负，当前 g={g}"
    assert b >= 0, f"线性阻力系数 b 必须非负，当前 b={b}"
    assert c >= 0, f"二次阻力系数 c 必须非负，当前 c={c}"
    assert m > 0, f"质量 m 必须为正，当前 m={m}"


def terminal_velocity(g=9.81, b=0.0, c=0.0, m=1.0):
    """计算终态速度（向下为正）。

    当重力与阻力平衡时，质点达到终态速度：
        b·v_t + c·v_t² = m·g

    参数
    ----
    g, b, c, m : float
        物理参数，同 dynamics。

    返回
    ----
    float 或 None
        终态速度大小（向下）。如果 g=0 或 b=c=0（无阻力），返回 None。
    """
    if g == 0 or (b == 0 and c == 0):
        return None
    if c == 0:
        # 纯线性阻力：v_t = m*g/b
        return m * g / b
    if b == 0:
        # 纯二次阻力：v_t = sqrt(m*g/c)
        return np.sqrt(m * g / c)
    # 混合阻力：c*v_t² + b*v_t - m*g = 0
    return (-b + np.sqrt(b ** 2 + 4 * c * m * g)) / (2 * c)


def dynamics(t, state, g=9.81, b=0.0, c=0.0, m=1.0):
    """返回状态的时间导数 d(state)/dt = [dx/dt, dy/dt, dvx/dt, dvy/dt]。

    参数
    ----
    t : float
        当前时刻（阻力运动显式不依赖 t，保留参数以统一接口）。
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    g : float, optional
        重力加速度（默认 9.81 m/s²）。
    b : float, optional
        线性阻力系数（默认 0.0，即无线性阻力）。
    c : float, optional
        二次阻力系数（默认 0.0，即无二次阻力）。
    m : float, optional
        质量（默认 1.0 kg）。

    返回
    ----
    np.ndarray, shape (4,)
        [vx, vy, ax, ay]
    """
    x, y, vx, vy = state
    v_mag = np.sqrt(vx ** 2 + vy ** 2)
    ax = -(b / m) * vx - (c / m) * v_mag * vx
    ay = -g - (b / m) * vy - (c / m) * v_mag * vy
    return np.array([vx, vy, ax, ay])


def analytical(t, initial_state, g=9.81, b=0.0, c=0.0, m=1.0):
    """速度相关阻力下的质点运动解析解。

    根据阻力类型自动选择解析方法：
    - 无阻力（b=0, c=0）：标准抛体运动（MEC-003 极限）
    - 纯线性阻力（c=0, b>0）：闭式指数解
    - 纯二次阻力 + 一维垂直（b=0, c>0, vx0=0）：分段 tan/tanh 解
    - 其他情况：抛出 ValueError

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    initial_state : array_like, shape (4,)
        初始状态 [x0, y0, vx0, vy0]。
    g, b, c, m : float
        物理参数，同 dynamics。

    返回
    ----
    (x, y, vx, vy) : tuple
        x(t), y(t), vx(t), vy(t)，形状与 t 一致。

    抛出
    ----
    ValueError
        一般二维二次阻力或混合阻力无闭式解析解。
    """
    t = np.asarray(t, dtype=float)
    scalar_input = (t.ndim == 0)
    t = np.atleast_1d(t)

    x0, y0, vx0, vy0 = initial_state

    if b == 0 and c == 0:
        # --- 无阻力：标准抛体运动（MEC-003 极限）---
        x = x0 + vx0 * t
        y = y0 + vy0 * t - 0.5 * g * t ** 2
        vx = np.full_like(t, vx0, dtype=float)
        vy = vy0 - g * t

    elif c == 0 and b > 0:
        # --- 纯线性阻力：闭式解析解 ---
        gamma = b / m
        exp_gt = np.exp(-gamma * t)
        vx = vx0 * exp_gt
        vy = (vy0 + g / gamma) * exp_gt - g / gamma
        x = x0 + (vx0 / gamma) * (1.0 - exp_gt)
        y = y0 + ((vy0 + g / gamma) / gamma) * (1.0 - exp_gt) - (g / gamma) * t

    elif b == 0 and c > 0:
        # --- 纯二次阻力 ---
        if abs(vx0) > 1e-14:
            raise ValueError(
                "一般二维二次阻力无闭式解析解。"
                "请使用 vx0=0 的一维垂直运动，或使用 scipy_solve.py 数值求解。"
            )
        if g == 0:
            raise ValueError("g=0 时二次阻力无终态速度，不支持解析解。")

        # 一维垂直运动
        v_t = np.sqrt(m * g / c)  # 终端速度
        vy0_arr = vy0

        # 初始化输出数组
        vx = np.zeros_like(t)
        x = np.full_like(t, x0, dtype=float)

        if vy0 > 0:
            # 有上升阶段
            theta0 = np.arctan(vy0 / v_t)
            t_apex = theta0 * v_t / g

            # 上升阶段掩码
            up_mask = t < t_apex
            down_mask = ~up_mask

            # 上升阶段
            if np.any(up_mask):
                t_up = t[up_mask]
                theta = theta0 - (g / v_t) * t_up
                vy_up = v_t * np.tan(theta)
                y_up = y0 + (v_t ** 2 / g) * np.log(
                    np.cos(theta) / np.cos(theta0))
                vy = np.empty_like(t, dtype=float)
                y = np.empty_like(t, dtype=float)
                vy[up_mask] = vy_up
                y[up_mask] = y_up

            # 下降阶段
            if np.any(down_mask):
                t_down = t[down_mask]
                t_from_apex = t_down - t_apex
                phi = (g / v_t) * t_from_apex
                vy_down = -v_t * np.tanh(phi)

                # y(t_apex) 的值
                y_apex = y0 + (v_t ** 2 / g) * np.log(1.0 / np.cos(theta0))
                y_down = y_apex - (v_t ** 2 / g) * np.log(np.cosh(phi))

                if not np.any(up_mask):
                    vy = np.empty_like(t, dtype=float)
                    y = np.empty_like(t, dtype=float)
                vy[down_mask] = vy_down
                y[down_mask] = y_down
        else:
            # 初始就向下或静止，全下降阶段
            if vy0 < 0:
                phi0 = np.arctanh(-vy0 / v_t)
            else:
                # vy0 = 0
                phi0 = 0.0
            phi = phi0 + (g / v_t) * t
            vy = -v_t * np.tanh(phi)
            y = y0 - (v_t ** 2 / g) * (
                np.log(np.cosh(phi)) - np.log(np.cosh(phi0)))

    else:
        raise ValueError(
            "混合阻力（b>0 且 c>0）无闭式解析解。"
            "请使用 scipy_solve.py 数值求解。")

    if scalar_input:
        return x[0], y[0], vx[0], vy[0]
    return x, y, vx, vy
