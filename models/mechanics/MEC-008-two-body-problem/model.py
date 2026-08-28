"""MEC-008-two-body-problem —— 模型定义（引擎无关）

二体问题（Two-body problem）：两个质点在互相引力作用下运动。
通过约化质量（reduced mass）和相对坐标（relative coordinates）化简
为等效的单体开普勒问题，是 MEC-007 的直接推广。

状态向量 state = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    x1, y1    —— 质点 1 的位置
    vx1, vy1  —— 质点 1 的速度
    x2, y2    —— 质点 2 的位置
    vx2, vy2  —— 质点 2 的速度

参数：
    G  —— 引力常数
    m1 —— 质点 1 的质量
    m2 —— 质点 2 的质量

动力学（一阶常微分方程）：

    两个质点在万有引力作用下运动：
        F_12 = -G·m1·m2 / r² · r̂  （质点 1 受质点 2 的引力）
    其中 r = r1 - r2 为相对位置，r = |r|。

    加速度：
        a1 = -G·m2·(r1-r2) / r³
        a2 = -G·m1·(r2-r1) / r³

    因此：
        dx1/dt = vx1,  dy1/dt = vy1
        dx2/dt = vx2,  dy2/dt = vy2
        dvx1/dt = -G·m2·(x1-x2) / r³
        dvy1/dt = -G·m2·(y1-y2) / r³
        dvx2/dt = -G·m1·(x2-x1) / r³
        dvy2/dt = -G·m1·(y2-y1) / r³

约化质量与相对坐标：

    约化质量：μ_red = m1·m2 / (m1+m2)
    引力参数：μ = G·(m1+m2)   （注意：μ ≠ μ_red）
    质心位置：R_cm = (m1·r1 + m2·r2) / (m1+m2)
    相对位置：r_rel = r1 - r2

    相对运动方程：
        d²r_rel/dt² = -μ·r_rel / r³
    这与 MEC-007 的单体开普勒问题形式完全相同，μ = G·(m1+m2)。
    质心做匀速直线运动（无外力），因此：
        r1 = R_cm + (m2/(m1+m2))·r_rel
        r2 = R_cm - (m1/(m1+m2))·r_rel

    当 m2 >> m1 时，质点 2 几乎不动（近似为固定力心），
    质点 1 的运动退化为 MEC-007 的单体问题。

守恒量：

    总动量：P = m1·v1 + m2·v2（恒定）
    总角动量：L = m1·(x1·vy1 - y1·vx1) + m2·(x2·vy2 - y2·vx2)（恒定）
    总机械能：E = ½·m1·|v1|² + ½·m2·|v2|² - G·m1·m2/r（恒定）
    质心速度：V_cm = (m1·v1 + m2·v2) / (m1+m2)（恒定）

解析解（半解析）：

    1. 质心运动：R_cm(t) = R_cm(0) + V_cm·t（匀速直线运动）
    2. 相对运动：开普勒方程（与 MEC-007 相同的求解器）
       平近点角 M = M0 + sign(h_rel)·n·t
       开普勒方程 M = E - e·sin(E)（椭圆）或 M = e·sinh(H) - H（双曲）
    3. 重建：
       r1 = R_cm + (m2/M)·r_rel
       r2 = R_cm - (m1/M)·r_rel
       其中 M = m1 + m2

初始状态约束：
    任意 (x1, y1, vx1, vy1, x2, y2, vx2, vy2) 都是合法初始状态。
    仅要求 G > 0, m1 > 0, m2 > 0，且两质点初始位置不重合（r_rel ≠ 0）。

退化情形：
    圆轨道条件：相对速度 v_rel = √(μ/r_rel) 且 v_rel ⊥ r_rel。
    此时偏心率 e = 0，两个质点绕共同质心做匀速圆周运动。

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


# ==================================================================
# 参数验证
# ==================================================================

def validate_parameters(G=1.0, m1=1.0, m2=1.0):
    """验证基本物理参数合法性。

    参数
    ----
    G : float
        引力常数（必须 > 0）。
    m1 : float
        质点 1 的质量（必须 > 0）。
    m2 : float
        质点 2 的质量（必须 > 0）。

    返回
    ----
    None
        条件满足时静默返回。

    抛出
    ----
    AssertionError
        参数不合法时给出明确错误信息。
    """
    assert G > 0, f"引力常数 G 必须为正，当前 G={G}"
    assert m1 > 0, f"质量 m1 必须为正，当前 m1={m1}"
    assert m2 > 0, f"质量 m2 必须为正，当前 m2={m2}"


# ==================================================================
# 约化质量与引力参数
# ==================================================================

def reduced_mass(m1=1.0, m2=1.0):
    """计算约化质量 μ_red = m1·m2 / (m1+m2)。

    参数
    ----
    m1, m2 : float
        两个质点的质量。

    返回
    ----
    float
        约化质量。
    """
    return m1 * m2 / (m1 + m2)


def gravitational_parameter(G=1.0, m1=1.0, m2=1.0):
    """计算引力参数 μ = G·(m1+m2)。

    这是相对运动等效开普勒问题的引力参数，
    与 MEC-007 中的 mu 对应。

    参数
    ----
    G : float
        引力常数。
    m1, m2 : float
        两个质点的质量。

    返回
    ----
    float
        引力参数 μ = G·(m1+m2)。
    """
    return G * (m1 + m2)


# ==================================================================
# 质心与相对坐标
# ==================================================================

def center_of_mass(state, m1=1.0, m2=1.0):
    """计算质心位置和速度。

    参数
    ----
    state : array_like, shape (8,)
        状态 [x1, y1, vx1, vy1, x2, y2, vx2, vy2]。
    m1, m2 : float
        两个质点的质量。

    返回
    ----
    tuple (X, Y, Vx, Vy)
        质心位置和速度。
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
    M = m1 + m2
    X = (m1 * x1 + m2 * x2) / M
    Y = (m1 * y1 + m2 * y2) / M
    Vx = (m1 * vx1 + m2 * vx2) / M
    Vy = (m1 * vy1 + m2 * vy2) / M
    return X, Y, Vx, Vy


def relative_state(state):
    """提取相对坐标和相对速度。

    r_rel = r1 - r2, v_rel = v1 - v2

    参数
    ----
    state : array_like, shape (8,)
        状态 [x1, y1, vx1, vy1, x2, y2, vx2, vy2]。

    返回
    ----
    np.ndarray, shape (4,)
        相对状态 [x_rel, y_rel, vx_rel, vy_rel]。
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
    return np.array([x1 - x2, y1 - y2, vx1 - vx2, vy1 - vy2])


# ==================================================================
# 守恒量
# ==================================================================

def total_momentum(state, m1=1.0, m2=1.0):
    """计算总动量 P = m1·v1 + m2·v2。

    参数
    ----
    state : array_like, shape (8,)
        状态 [x1, y1, vx1, vy1, x2, y2, vx2, vy2]。
    m1, m2 : float
        两个质点的质量。

    返回
    ----
    np.ndarray, shape (2,)
        总动量 [Px, Py]。
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
    Px = m1 * vx1 + m2 * vx2
    Py = m1 * vy1 + m2 * vy2
    return np.array([Px, Py])


def total_angular_momentum(state, m1=1.0, m2=1.0):
    """计算总角动量 L = m1·(x1·vy1-y1·vx1) + m2·(x2·vy2-y2·vx2)。

    参数
    ----
    state : array_like, shape (8,)
        状态 [x1, y1, vx1, vy1, x2, y2, vx2, vy2]。
    m1, m2 : float
        两个质点的质量。

    返回
    ----
    float
        总角动量 L（z 分量）。
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
    L1 = m1 * (x1 * vy1 - y1 * vx1)
    L2 = m2 * (x2 * vy2 - y2 * vx2)
    return L1 + L2


def total_energy(state, G=1.0, m1=1.0, m2=1.0):
    """计算总机械能 E = ½·m1·|v1|² + ½·m2·|v2|² - G·m1·m2/r。

    参数
    ----
    state : array_like, shape (8,)
        状态 [x1, y1, vx1, vy1, x2, y2, vx2, vy2]。
    G : float
        引力常数。
    m1, m2 : float
        两个质点的质量。

    返回
    ----
    float
        总机械能 E。
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
    r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    ke = 0.5 * m1 * (vx1 ** 2 + vy1 ** 2) + 0.5 * m2 * (vx2 ** 2 + vy2 ** 2)
    pe = -G * m1 * m2 / r
    return ke + pe


# ==================================================================
# 相对运动的偏心率向量和轨道根数
# ==================================================================

def relative_eccentricity_vector(state, G=1.0, m1=1.0, m2=1.0):
    """计算相对运动的偏心率向量（Laplace-Runge-Lenz 向量）。

    对相对运动 r_rel = r1 - r2，等效引力参数 μ = G·(m1+m2)。
    偏心率向量 e_vec = (v_rel × h_rel) / μ - r_rel_hat。

    参数
    ----
    state : array_like, shape (8,)
        状态 [x1, y1, vx1, vy1, x2, y2, vx2, vy2]。
    G, m1, m2 : float
        引力常数和两个质点质量。

    返回
    ----
    np.ndarray, shape (2,)
        偏心率向量 [e_x, e_y]。
    """
    rel = relative_state(state)
    x_rel, y_rel, vx_rel, vy_rel = rel
    r = np.sqrt(x_rel ** 2 + y_rel ** 2)
    h = x_rel * vy_rel - y_rel * vx_rel  # 比角动量 z 分量
    mu = gravitational_parameter(G, m1, m2)
    e_x = (vy_rel * h) / mu - x_rel / r
    e_y = (-vx_rel * h) / mu - y_rel / r
    return np.array([e_x, e_y])


def relative_orbital_elements(state, G=1.0, m1=1.0, m2=1.0):
    """从状态向量计算相对运动的轨道根数。

    参数
    ----
    state : array_like, shape (8,)
        状态 [x1, y1, vx1, vy1, x2, y2, vx2, vy2]。
    G, m1, m2 : float
        引力常数和两个质点质量。

    返回
    ----
    dict
        包含 a, e, omega, p, anomaly0, M0, n, h_sign, orbit_type。

    抛出
    ----
    ValueError
        抛物线轨道（ε≈0）或径向轨道（h≈0）不支持解析解。
    """
    rel = relative_state(state)
    x_rel, y_rel, vx_rel, vy_rel = rel
    r = np.sqrt(x_rel ** 2 + y_rel ** 2)
    v2 = vx_rel ** 2 + vy_rel ** 2
    h = x_rel * vy_rel - y_rel * vx_rel
    mu = gravitational_parameter(G, m1, m2)
    eps = 0.5 * v2 - mu / r  # 比能量

    e_vec = relative_eccentricity_vector(state, G, m1, m2)
    e = np.linalg.norm(e_vec)
    omega = np.arctan2(e_vec[1], e_vec[0])
    a = -mu / (2 * eps)
    p = h ** 2 / mu
    h_sign = np.sign(h) if h != 0 else 1.0

    if abs(h) < 1e-14:
        raise ValueError("角动量接近零（径向轨道），解析解不支持")
    if abs(eps) < 1e-14:
        raise ValueError("抛物线轨道（ε≈0），解析解不支持")

    # 旋转到轨道坐标系
    cos_w = np.cos(omega)
    sin_w = np.sin(omega)
    x_orb = x_rel * cos_w + y_rel * sin_w
    y_orb = -x_rel * sin_w + y_rel * cos_w

    if eps < 0:
        # 椭圆轨道
        orbit_type = 'elliptical'
        n = np.sqrt(mu / a ** 3)
        cos_E0 = x_orb / a + e
        if e > 1e-12:
            sin_E0 = y_orb / (a * np.sqrt(1 - e ** 2))
        else:
            sin_E0 = y_orb / a
        E0 = np.arctan2(sin_E0, cos_E0)
        M0 = E0 - e * np.sin(E0)
        return {
            'a': a, 'e': e, 'omega': omega, 'p': p,
            'anomaly0': E0, 'M0': M0, 'n': n,
            'h_sign': h_sign, 'orbit_type': orbit_type,
        }
    else:
        # 双曲轨道
        orbit_type = 'hyperbolic'
        a_abs = abs(a)
        n = np.sqrt(mu / a_abs ** 3)
        sinh_H0 = y_orb / (a_abs * np.sqrt(e ** 2 - 1))
        H0 = np.arcsinh(sinh_H0)
        M0 = e * np.sinh(H0) - H0
        return {
            'a': a, 'e': e, 'omega': omega, 'p': p,
            'anomaly0': H0, 'M0': M0, 'n': n,
            'h_sign': h_sign, 'orbit_type': orbit_type,
        }


# ==================================================================
# 开普勒方程求解器（私有函数）
# ==================================================================

def _solve_kepler_elliptical(M, e, tol=1e-14, max_iter=100):
    """求解椭圆开普勒方程 M = E - e·sin(E)。

    使用牛顿迭代法。支持向量化 M。

    参数
    ----
    M : float 或 array_like
        平近点角。
    e : float
        偏心率（0 ≤ e < 1）。
    tol : float, optional
        收敛容差。
    max_iter : int, optional
        最大迭代次数。

    返回
    ----
    float 或 np.ndarray
        偏近点角 E。
    """
    M = np.asarray(M, dtype=float)
    E = M + e * np.sin(M)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = f / fp
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def _solve_kepler_hyperbolic(M, e, tol=1e-14, max_iter=100):
    """求解双曲开普勒方程 M = e·sinh(H) - H。

    使用牛顿迭代法。支持向量化 M。

    参数
    ----
    M : float 或 array_like
        平近点角。
    e : float
        偏心率（e > 1）。
    tol : float, optional
        收敛容差。
    max_iter : int, optional
        最大迭代次数。

    返回
    ----
    float 或 np.ndarray
        双曲近点角 H。
    """
    M = np.asarray(M, dtype=float)
    H = np.copy(M)
    for _ in range(max_iter):
        f = e * np.sinh(H) - H - M
        fp = e * np.cosh(H) - 1
        dH = f / fp
        H = H - dH
        if np.all(np.abs(dH) < tol):
            break
    return H


# ==================================================================
# 动力学方程
# ==================================================================

def dynamics(t, state, G=1.0, m1=1.0, m2=1.0):
    """返回状态的时间导数 d(state)/dt。

    参数
    ----
    t : float
        当前时刻（二体问题显式不依赖 t，保留参数以统一接口）。
    state : array_like, shape (8,)
        当前状态 [x1, y1, vx1, vy1, x2, y2, vx2, vy2]。
    G : float, optional
        引力常数（默认 1.0）。
    m1 : float, optional
        质点 1 的质量（默认 1.0）。
    m2 : float, optional
        质点 2 的质量（默认 1.0）。

    返回
    ----
    np.ndarray, shape (8,)
        [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]
    """
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
    dx = x1 - x2
    dy = y1 - y2
    r = np.sqrt(dx ** 2 + dy ** 2)
    r3 = r ** 3
    ax1 = -G * m2 * dx / r3
    ay1 = -G * m2 * dy / r3
    ax2 = -G * m1 * (-dx) / r3
    ay2 = -G * m1 * (-dy) / r3
    return np.array([vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2])


# ==================================================================
# 解析解
# ==================================================================

def analytical(t, initial_state, G=1.0, m1=1.0, m2=1.0):
    """二体问题半解析解。

    将二体问题分解为质心运动（匀速直线）和相对运动（开普勒问题），
    然后重建两个质点的位置和速度。

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    initial_state : array_like, shape (8,)
        初始状态 [x1, y1, vx1, vy1, x2, y2, vx2, vy2]。
    G : float, optional
        引力常数（默认 1.0）。
    m1 : float, optional
        质点 1 的质量（默认 1.0）。
    m2 : float, optional
        质点 2 的质量（默认 1.0）。

    返回
    ----
    (x1, y1, vx1, vy1, x2, y2, vx2, vy2) : tuple
        两个质点的位置和速度，形状与 t 一致。

    抛出
    ----
    ValueError
        抛物线轨道（ε≈0）或径向轨道（h≈0）不支持解析解。
    """
    t = np.asarray(t, dtype=float)
    scalar_input = (t.ndim == 0)
    t = np.atleast_1d(t)

    state = np.array(initial_state, dtype=float)
    M_total = m1 + m2
    mu = gravitational_parameter(G, m1, m2)

    # --- 质心运动 ---
    X0, Y0, Vx_cm, Vy_cm = center_of_mass(state, m1, m2)
    X_cm = X0 + Vx_cm * t
    Y_cm = Y0 + Vy_cm * t

    # --- 相对运动（开普勒问题）---
    rel0 = relative_state(state)
    elem = relative_orbital_elements(state, G, m1, m2)

    a = elem['a']
    e = elem['e']
    omega = elem['omega']
    M0 = elem['M0']
    n = elem['n']
    h_sign = elem['h_sign']
    orbit_type = elem['orbit_type']

    cos_w = np.cos(omega)
    sin_w = np.sin(omega)

    # 平近点角
    M = M0 + h_sign * n * t

    if orbit_type == 'elliptical':
        E = _solve_kepler_elliptical(M, e)
        x_orb = a * (np.cos(E) - e)
        y_orb = a * np.sqrt(1 - e ** 2) * np.sin(E)
        dE_dt = h_sign * n / (1 - e * np.cos(E))
        vx_orb = -a * np.sin(E) * dE_dt
        vy_orb = a * np.sqrt(1 - e ** 2) * np.cos(E) * dE_dt
    else:
        a_abs = abs(a)
        H = _solve_kepler_hyperbolic(M, e)
        x_orb = a_abs * (e - np.cosh(H))
        y_orb = a_abs * np.sqrt(e ** 2 - 1) * np.sinh(H)
        dH_dt = h_sign * n / (e * np.cosh(H) - 1)
        vx_orb = -a_abs * np.sinh(H) * dH_dt
        vy_orb = a_abs * np.sqrt(e ** 2 - 1) * np.cosh(H) * dH_dt

    # 旋转到惯性系（相对坐标）
    x_rel = x_orb * cos_w - y_orb * sin_w
    y_rel = x_orb * sin_w + y_orb * cos_w
    vx_rel = vx_orb * cos_w - vy_orb * sin_w
    vy_rel = vx_orb * sin_w + vy_orb * cos_w

    # --- 重建两个质点的位置和速度 ---
    frac1 = m2 / M_total  # 质点 1 相对质心的系数
    frac2 = m1 / M_total  # 质点 2 相对质心的系数

    x1 = X_cm + frac1 * x_rel
    y1 = Y_cm + frac1 * y_rel
    vx1 = Vx_cm + frac1 * vx_rel
    vy1 = Vy_cm + frac1 * vy_rel

    x2 = X_cm - frac2 * x_rel
    y2 = Y_cm - frac2 * y_rel
    vx2 = Vx_cm - frac2 * vx_rel
    vy2 = Vy_cm - frac2 * vy_rel

    if scalar_input:
        return (x1[0], y1[0], vx1[0], vy1[0],
                x2[0], y2[0], vx2[0], vy2[0])
    return (x1, y1, vx1, vy1, x2, y2, vx2, vy2)
