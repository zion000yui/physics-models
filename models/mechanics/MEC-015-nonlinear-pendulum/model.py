"""MEC-015-nonlinear-pendulum —— 模型定义（引擎无关）

非线性单摆（nonlinear pendulum）：质量 m 在长度 L 的无质量刚性杆末端，
在重力 g 作用下做有限振幅摆动。不使用小角度近似，完整保留 sin(θ) 非线性。
这是从线性振动系统（MEC-010~014）进入非线性动力学的关键过渡模型。

状态向量 state = [theta, omega]
    theta —— 摆角（rad，相对于平衡位置，平衡时 θ=0）
    omega —— 角速度（rad/s）

参数：
    g —— 重力加速度（m/s²，g > 0）
    L —— 摆长（m，L > 0）
    m —— 质量（kg，m > 0，不影响运动方程，仅出现在机械能中）

动力学（一阶常微分方程）：

    非线性单摆运动方程：
        θ̈ + (g/L)·sin(θ) = 0

    状态空间形式：
        dθ/dt = ω
        dω/dt = -(g/L)·sin(θ)

    小角度线性近似（sin(θ) ≈ θ）：
        θ̈ + (g/L)·θ = 0
    这是 MEC-010 简谐振子形式，角频率 ω₀ = √(g/L)。
    小角度近似仅在 |θ| ≪ 1 时有效，不是本模型的主方程。

解析解：

    1. 小角度线性近似（|θ| ≪ 1）：
        θ(t) = θ₀·cos(ω₀·t) + (ω₀_init/ω₀)·sin(ω₀·t)
        ω(t) = -θ₀·ω₀·sin(ω₀·t) + ω₀_init·cos(ω₀·t)
        其中 ω₀ = √(g/L)，周期 T₀ = 2π/ω₀

    2. 有限振幅周期（椭圆积分理论结果）：
        振幅为 θ_max 的非线性单摆周期为：
            T = 4·√(L/g)·K(k)
        其中 K(k) 为第一类完全椭圆积分，k = sin(θ_max/2)。
        小角度极限：θ_max→0 时 K(0)=π/2，T→T₀=2π·√(L/g)。
        大振幅时周期增长，θ_max→π 时 T→∞。

    3. 一般有限振幅运动：
        无初等函数形式的时间解析解。时间演化通过数值积分获得。
        可用 Jacobi 椭圆函数表示，但本模型不依赖此表示。

机械能：
    E = ½·m·L²·ω² + m·g·L·(1 - cos(θ))
    无阻尼时机械能守恒。

相空间（θ - ω 平面）：

    - 小振幅：轨迹接近椭圆（接近线性简谐振子）
    - 有限振幅：轨迹偏离椭圆（周期变长）
    - 分离轨道（separatrix）：E = 2mgL，连接不稳定平衡点 (±π, 0)
    - 摆动（E < 2mgL）：θ 在 (-π, π) 间振荡
    - 旋转（E > 2mgL）：θ 单调递增/递减

与 MEC-010 的关系：
    小角度极限 sin(θ)→θ 时，退化为 MEC-010 简谐振子，
    ω₀ = √(g/L) 对应 MEC-010 的 ω₀ = √(k/m)。
    有效弹性系数 k_eff = mg/L（重力恢复力的线性化）。

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np
from scipy.special import ellipk


def validate_parameters(g=9.81, L=1.0, m=1.0):
    """验证基本物理参数合法性。

    参数
    ----
    g : float
        重力加速度（必须 > 0）。
    L : float
        摆长（必须 > 0）。
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
    assert g > 0, f"重力加速度 g 必须为正，当前 g={g}"
    assert L > 0, f"摆长 L 必须为正，当前 L={L}"
    assert m > 0, f"质量 m 必须为正，当前 m={m}"


def natural_frequency(g=9.81, L=1.0):
    """计算小角度线性近似的固有角频率 ω₀ = √(g/L)。

    这对应 MEC-010 的 ω₀ = √(k/m)，其中有效 k = mg/L。

    参数
    ----
    g, L : float
        重力加速度和摆长。

    返回
    ----
    float
        固有角频率 ω₀（rad/s）。
    """
    return np.sqrt(g / L)


def small_angle_period(g=9.81, L=1.0):
    """计算小角度线性近似周期 T₀ = 2π/ω₀ = 2π·√(L/g)。

    参数
    ----
    g, L : float
        重力加速度和摆长。

    返回
    ----
    float
        小角度周期 T₀（s）。
    """
    return 2.0 * np.pi * np.sqrt(L / g)


def nonlinear_period(g=9.81, L=1.0, theta_max=0.1):
    """计算有限振幅非线性单摆周期。

    T = 4·√(L/g)·K(k)，其中 k = sin(θ_max/2)，K 为第一类完全椭圆积分。

    参数
    ----
    g, L : float
        重力加速度和摆长。
    theta_max : float
        摆动振幅（rad，0 < θ_max < π）。

    返回
    ----
    float
        非线性周期 T（s）。

    抛出
    ----
    ValueError
        当 θ_max ≥ π（分离轨道以上）时周期发散。
    """
    if theta_max >= np.pi - 1e-10:
        raise ValueError("θ_max ≥ π 时周期发散（接近分离轨道）")
    k = np.sin(theta_max / 2.0)
    K_val = ellipk(k ** 2)  # scipy 的 ellipk 接收 m = k²
    return 4.0 * np.sqrt(L / g) * K_val


def mechanical_energy(state, g=9.81, L=1.0, m=1.0):
    """计算机械能 E = ½·m·L²·ω² + m·g·L·(1-cos(θ))。

    参数
    ----
    state : array_like, shape (2,)
        状态 [theta, omega]。
    g, L, m : float
        重力加速度、摆长、质量。

    返回
    ----
    float
        机械能 E。
    """
    theta, omega = state
    ke = 0.5 * m * L ** 2 * omega ** 2
    pe = m * g * L * (1.0 - np.cos(theta))
    return ke + pe


def dynamics(t, state, g=9.81, L=1.0, m=1.0):
    """返回状态的时间导数 d(state)/dt = [dθ/dt, dω/dt]。

    使用完整非线性方程 θ̈ + (g/L)·sin(θ) = 0。

    参数
    ----
    t : float
        当前时刻（保守系统不依赖 t，保留以统一接口）。
    state : array_like, shape (2,)
        状态 [theta, omega]。
    g : float, optional
        重力加速度（默认 9.81 m/s²）。
    L : float, optional
        摆长（默认 1.0 m）。
    m : float, optional
        质量（默认 1.0 kg，不参与运动方程，保留以统一接口）。

    返回
    ----
    np.ndarray, shape (2,)
        [omega, -(g/L)·sin(theta)]
    """
    theta, omega = state
    return np.array([omega, -(g / L) * np.sin(theta)])


def analytical(t, initial_state, g=9.81, L=1.0, m=1.0):
    """小角度线性近似解析解。

    仅适用于小角度情况（|θ| ≪ 1）。对于有限振幅运动，请使用
    scipy_solve.py 的数值积分，以及 nonlinear_period() 的椭圆积分结果。

    参数
    ----
    t : float 或 array_like
        时间点。
    initial_state : array_like, shape (2,)
        初始状态 [theta0, omega0]。
    g, L, m : float
        物理参数。

    返回
    ----
    (theta, omega) : tuple
        角位移和角速度，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    theta0, omega0 = initial_state
    omega_0 = natural_frequency(g, L)
    cos_wt = np.cos(omega_0 * t)
    sin_wt = np.sin(omega_0 * t)
    theta = theta0 * cos_wt + (omega0 / omega_0) * sin_wt
    omega = -theta0 * omega_0 * sin_wt + omega0 * cos_wt
    return theta, omega
