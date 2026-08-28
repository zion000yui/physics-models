"""MEC-007-central-force-gravity —— 模型定义（引擎无关）

平方反比中心力（万有引力 / 开普勒问题，Kepler problem）：质点在指向原点的
引力 F = -μm/r² · r̂ 作用下运动，轨迹为圆锥曲线（椭圆 / 抛物线 / 双曲线）。
对于束缚态（E < 0），轨道为以力心为焦点的椭圆。

这是 MEC-006 胡克型中心力（F ∝ -r）的自然对偶：
- MEC-006: F ∝ -r（胡克型），力心在椭圆中心，周期与振幅无关
- MEC-007: F ∝ -1/r²（万有引力），力心在椭圆焦点，周期 ∝ a^(3/2)（开普勒第三定律）
两者共同验证 Bertrand 定理——只有 r¹ 和 r⁻² 两种幂律中心力能产生闭合轨道。

状态向量 state = [x, y, vx, vy]
    x  —— 水平位置
    y  —— 垂直位置
    vx —— 水平速度
    vy —— 垂直速度

参数：
    mu —— 引力参数（gravitational parameter, μ = GM），单位 m³/s²
    m  —— 质点质量，单位 kg

动力学（一阶常微分方程）：

    引力加速度：a = -μ/r³ · r = -μ/r³ · (x, y)
    牛顿第二定律：m·a = F
    注意：加速度与质点质量 m 无关（等效原理），m 仅出现在守恒量中。

    因此：
        dx/dt = vx
        dy/dt = vy
        dvx/dt = -μ·x / r³
        dvy/dt = -μ·y / r³
    其中 r = √(x² + y²)

守恒量：

    角动量（绕原点）：
        L = m·(x·vy - y·vx)

    机械能：
        E = ½·m·(vx² + vy²) - μ·m / r

    偏心率向量（Laplace-Runge-Lenz 向量，比单位质量）：
        e_vec = (v × h) / μ - r̂
    其中 h = x·vy - y·vx 为比角动量（z 分量）。
    该向量的模等于轨道偏心率 e，方向指向近心点。

    在 2D 中：
        e_x = (vy·h) / μ - x/r
        e_y = -(vx·h) / μ - y/r

解析解（半解析，用于校验数值解的"金标准"）：

    从初始条件提取开普勒轨道根数（半长轴 a、偏心率 e、近心点幅角 ω 等），
    然后通过开普勒方程将时间映射到位置。

    对于束缚态（椭圆轨道，ε < 0，e < 1）：
        平均运动：n = √(μ/a³)
        平近点角：M = M₀ + sign(h)·n·t
        开普勒方程：M = E - e·sin(E)（E 为偏近点角，用牛顿迭代求解）
        轨道坐标系位置（近心点沿 x 轴）：
            x_orb = a·(cos(E) - e)
            y_orb = a·√(1-e²)·sin(E)
        轨道坐标系速度：
            dE/dt = sign(h)·n / (1 - e·cos(E))
            vx_orb = -a·sin(E)·dE/dt
            vy_orb = a·√(1-e²)·cos(E)·dE/dt

    对于非束缚态（双曲轨道，ε > 0，e > 1）：
        平均运动：n = √(μ/|a|³)（a < 0 为约定）
        双曲开普勒方程：M = e·sinh(H) - H（H 为双曲近点角）
        轨道坐标系位置：
            x_orb = |a|·(e - cosh(H))
            y_orb = |a|·√(e²-1)·sinh(H)

    最后旋转近心点幅角 ω 得到惯性系坐标：
        x = x_orb·cos(ω) - y_orb·sin(ω)
        y = x_orb·sin(ω) + y_orb·cos(ω)

初始状态约束：
    任意 (x0, y0, vx0, vy0) 都是合法初始状态（与 MEC-006 一致）。
    仅要求 μ > 0, m > 0。

退化情形：
    圆轨道条件：|v| = √(μ/r) 且 v ⊥ r。
    此时 e = 0，轨道退化为以力心为圆心的圆。

    抛物线轨道（ε = 0, e = 1）和径向轨道（h = 0）暂不支持解析解。

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(mu=1.0, m=1.0):
    """验证基本物理参数合法性。

    参数
    ----
    mu : float
        引力参数（必须 > 0）。
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
    assert mu > 0, f"引力参数 mu 必须为正，当前 mu={mu}"
    assert m > 0, f"质量 m 必须为正，当前 m={m}"


def specific_angular_momentum(state):
    """计算比角动量 h = x·vy - y·vx。

    参数
    ----
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。

    返回
    ----
    float
        比角动量 h（角动量 z 分量除以质量）。
    """
    x, y, vx, vy = state
    return x * vy - y * vx


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


def specific_energy(state, mu=1.0):
    """计算比机械能 ε = ½(vx²+vy²) - μ/r。

    参数
    ----
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    mu : float, optional
        引力参数（默认 1.0 m³/s²）。

    返回
    ----
    float
        比机械能 ε（机械能除以质量）。
    """
    x, y, vx, vy = state
    r = np.sqrt(x ** 2 + y ** 2)
    return 0.5 * (vx ** 2 + vy ** 2) - mu / r


def mechanical_energy(state, mu=1.0, m=1.0):
    """计算机械能 E = ½m(vx²+vy²) - μm/r。

    参数
    ----
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    mu : float, optional
        引力参数（默认 1.0 m³/s²）。
    m : float, optional
        质量（默认 1.0 kg）。

    返回
    ----
    float
        机械能 E。
    """
    x, y, vx, vy = state
    r = np.sqrt(x ** 2 + y ** 2)
    return 0.5 * m * (vx ** 2 + vy ** 2) - mu * m / r


def eccentricity_vector(state, mu=1.0):
    """计算偏心率向量（Laplace-Runge-Lenz 向量，比单位质量）。

    e_vec = (v × h) / μ - r̂

    在 2D 中（h 为比角动量 z 分量 h = x·vy - y·vx）：
        e_x = (vy·h) / μ - x/r
        e_y = -(vx·h) / μ - y/r

    参数
    ----
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    mu : float, optional
        引力参数（默认 1.0 m³/s²）。

    返回
    ----
    np.ndarray, shape (2,)
        偏心率向量 [e_x, e_y]，模等于轨道偏心率 e。
    """
    x, y, vx, vy = state
    r = np.sqrt(x ** 2 + y ** 2)
    h = x * vy - y * vx  # 比角动量 z 分量
    e_x = (vy * h) / mu - x / r
    e_y = (-vx * h) / mu - y / r
    return np.array([e_x, e_y])


def orbital_elements(state, mu=1.0):
    """从状态向量计算开普勒轨道根数。

    参数
    ----
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    mu : float, optional
        引力参数（默认 1.0 m³/s²）。

    返回
    ----
    dict
        包含以下键：
        - 'a' : 半长轴（椭圆为正，双曲为负）
        - 'e' : 偏心率
        - 'omega' : 近心点幅角（rad）
        - 'p' : 半通径 p = h²/μ
        - 'anomaly0' : 初始近点角（椭圆为 E₀，双曲为 H₀）
        - 'M0' : 初始平近点角
        - 'n' : 平均运动（恒正）
        - 'h_sign' : 角动量符号（+1 顺行，-1 逆行）
        - 'orbit_type' : 'elliptical' 或 'hyperbolic'

    抛出
    ----
    ValueError
        抛物线轨道（ε≈0）或径向轨道（h≈0）不支持解析解。
    """
    x, y, vx, vy = state
    r = np.sqrt(x ** 2 + y ** 2)
    v2 = vx ** 2 + vy ** 2
    h = x * vy - y * vx  # 比角动量
    eps = 0.5 * v2 - mu / r  # 比能量

    # 偏心率向量
    e_vec = eccentricity_vector(state, mu)
    e = np.linalg.norm(e_vec)
    omega = np.arctan2(e_vec[1], e_vec[0])  # 近心点幅角

    # 半长轴（椭圆 a>0，双曲 a<0）
    a = -mu / (2 * eps)

    # 半通径
    p = h ** 2 / mu

    # 角动量符号（轨道方向）
    h_sign = np.sign(h) if h != 0 else 1.0

    # 径向轨道（h≈0）不支持解析解
    if abs(h) < 1e-14:
        raise ValueError("角动量接近零（径向轨道），解析解不支持")

    # 抛物线轨道（ε≈0）不支持解析解
    if abs(eps) < 1e-14:
        raise ValueError("抛物线轨道（ε≈0），解析解不支持")

    # 旋转到轨道坐标系（近心点沿 x 轴）
    cos_w = np.cos(omega)
    sin_w = np.sin(omega)
    x_orb = x * cos_w + y * sin_w
    y_orb = -x * sin_w + y * cos_w

    if eps < 0:
        # 椭圆轨道
        orbit_type = 'elliptical'
        n = np.sqrt(mu / a ** 3)

        # 计算初始偏近点角 E0
        cos_E0 = x_orb / a + e
        if e > 1e-12:
            sin_E0 = y_orb / (a * np.sqrt(1 - e ** 2))
        else:
            # 圆轨道退化：E0 直接从位置确定
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

        # 计算初始双曲近点角 H0
        # cosh(H0) = e - x_orb/|a|, sinh(H0) = y_orb/(|a|·√(e²-1))
        sinh_H0 = y_orb / (a_abs * np.sqrt(e ** 2 - 1))
        H0 = np.arcsinh(sinh_H0)
        M0 = e * np.sinh(H0) - H0

        return {
            'a': a, 'e': e, 'omega': omega, 'p': p,
            'anomaly0': H0, 'M0': M0, 'n': n,
            'h_sign': h_sign, 'orbit_type': orbit_type,
        }


def _solve_kepler_elliptical(M, e, tol=1e-14, max_iter=100):
    """求解椭圆开普勒方程 M = E - e·sin(E)。

    使用牛顿迭代法。支持向量化 M（标量或数组均可）。

    参数
    ----
    M : float 或 array_like
        平近点角。
    e : float
        偏心率（0 ≤ e < 1）。
    tol : float, optional
        收敛容差（默认 1e-14）。
    max_iter : int, optional
        最大迭代次数（默认 100）。

    返回
    ----
    float 或 np.ndarray
        偏近点角 E，形状与 M 一致。
    """
    M = np.asarray(M, dtype=float)
    # 初始猜测：E ≈ M + e·sin(M)（Picard 一步迭代，对中等 e 收敛良好）
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

    使用牛顿迭代法。支持向量化 M（标量或数组均可）。

    参数
    ----
    M : float 或 array_like
        平近点角。
    e : float
        偏心率（e > 1）。
    tol : float, optional
        收敛容差（默认 1e-14）。
    max_iter : int, optional
        最大迭代次数（默认 100）。

    返回
    ----
    float 或 np.ndarray
        双曲近点角 H，形状与 M 一致。
    """
    M = np.asarray(M, dtype=float)
    # 初始猜测：H ≈ M（对中等 M 收敛良好）
    H = np.copy(M)
    for _ in range(max_iter):
        f = e * np.sinh(H) - H - M
        fp = e * np.cosh(H) - 1
        dH = f / fp
        H = H - dH
        if np.all(np.abs(dH) < tol):
            break
    return H


def dynamics(t, state, mu=1.0, m=1.0):
    """返回状态的时间导数 d(state)/dt = [dx/dt, dy/dt, dvx/dt, dvy/dt]。

    参数
    ----
    t : float
        当前时刻（万有引力显式不依赖 t，保留参数以统一接口）。
    state : array_like, shape (4,)
        当前状态 [x, y, vx, vy]。
    mu : float, optional
        引力参数（默认 1.0 m³/s²）。
    m : float, optional
        质量（默认 1.0 kg，不参与加速度计算，保留以统一接口）。

    返回
    ----
    np.ndarray, shape (4,)
        [vx, vy, -μ·x/r³, -μ·y/r³]
    """
    x, y, vx, vy = state
    r = np.sqrt(x ** 2 + y ** 2)
    r3 = r ** 3
    ax = -mu * x / r3
    ay = -mu * y / r3
    return np.array([vx, vy, ax, ay])


def analytical(t, initial_state, mu=1.0, m=1.0):
    """开普勒问题半解析解。

    从初始条件提取轨道根数，通过开普勒方程将时间映射到位置和速度。
    支持椭圆轨道（束缚态）和双曲轨道（非束缚态）。

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    initial_state : array_like, shape (4,)
        初始状态 [x0, y0, vx0, vy0]。
    mu : float, optional
        引力参数（默认 1.0 m³/s²）。
    m : float, optional
        质量（默认 1.0 kg，不参与解析解计算，保留以统一接口）。

    返回
    ----
    (x, y, vx, vy) : tuple
        x(t), y(t), vx(t), vy(t)，形状与 t 一致。

    抛出
    ----
    ValueError
        抛物线轨道（ε≈0）或径向轨道（h≈0）不支持解析解。
    """
    t = np.asarray(t, dtype=float)
    scalar_input = (t.ndim == 0)
    t = np.atleast_1d(t)

    state = np.array(initial_state, dtype=float)
    elements = orbital_elements(state, mu)

    a = elements['a']
    e = elements['e']
    omega = elements['omega']
    M0 = elements['M0']
    n = elements['n']
    h_sign = elements['h_sign']
    orbit_type = elements['orbit_type']

    cos_w = np.cos(omega)
    sin_w = np.sin(omega)

    # 平近点角
    M = M0 + h_sign * n * t

    if orbit_type == 'elliptical':
        # 椭圆轨道
        E = _solve_kepler_elliptical(M, e)
        x_orb = a * (np.cos(E) - e)
        y_orb = a * np.sqrt(1 - e ** 2) * np.sin(E)
        # 速度（轨道坐标系）
        dE_dt = h_sign * n / (1 - e * np.cos(E))
        vx_orb = -a * np.sin(E) * dE_dt
        vy_orb = a * np.sqrt(1 - e ** 2) * np.cos(E) * dE_dt
    else:
        # 双曲轨道
        a_abs = abs(a)
        H = _solve_kepler_hyperbolic(M, e)
        x_orb = a_abs * (e - np.cosh(H))
        y_orb = a_abs * np.sqrt(e ** 2 - 1) * np.sinh(H)
        # 速度（轨道坐标系）
        dH_dt = h_sign * n / (e * np.cosh(H) - 1)
        vx_orb = -a_abs * np.sinh(H) * dH_dt
        vy_orb = a_abs * np.sqrt(e ** 2 - 1) * np.cosh(H) * dH_dt

    # 旋转到惯性系
    x = x_orb * cos_w - y_orb * sin_w
    y = x_orb * sin_w + y_orb * cos_w
    vx = vx_orb * cos_w - vy_orb * sin_w
    vy = vx_orb * sin_w + vy_orb * cos_w

    if scalar_input:
        return x[0], y[0], vx[0], vy[0]
    return x, y, vx, vy
