"""MEC-011-damped-oscillator —— 模型定义（引擎无关）

阻尼振子（damped harmonic oscillator）：质量 m 在弹性力 F = -k·x 和
线性阻尼力 F_d = -b·v 共同作用下沿一维方向运动。
是 MEC-010 简谐振子的自然推广，在 MEC-010 的基础上引入阻尼项。

状态向量 state = [x, v]
    x —— 位移（相对于平衡位置）
    v —— 速度

参数：
    k —— 弹性系数（spring constant，N/m，k > 0）
    m —— 质量（kg，m > 0）
    b —— 阻尼系数（damping coefficient，N·s/m，b ≥ 0）

动力学（一阶常微分方程）：

    弹性恢复力：F_s = -k·x
    线性阻尼力：F_d = -b·v
    牛顿第二定律：m·a = F_s + F_d
    因此：
        dx/dt = v
        dv/dt = -(k/m)·x - (b/m)·v

    令 ω₀ = √(k/m)（固有角频率，同 MEC-010），
    γ = b/(2m)（衰减率），则运动方程为：
        d²x/dt² + 2γ·dx/dt + ω₀²·x = 0

阻尼比与三种状态：

    阻尼比（damping ratio）：ζ = b / (2·√(m·k)) = γ/ω₀

    - 欠阻尼（underdamped，0 ≤ ζ < 1）：
        γ < ω₀，系统做衰减振荡，角频率 ω_d = ω₀·√(1-ζ²)
    - 临界阻尼（critically damped，ζ = 1）：
        γ = ω₀，系统以最快速度回到平衡位置，无振荡
    - 过阻尼（overdamped，ζ > 1）：
        γ > ω₀，系统缓慢回到平衡位置，无振荡

    当 b=0（ζ=0）时，模型精确退化为 MEC-010 无阻尼简谐振子。

解析解（用于校验数值解的"金标准"）：

    1. 欠阻尼（0 ≤ ζ < 1）：
        令 ω_d = ω₀·√(1-ζ²)（阻尼角频率）
        x(t) = e^(-γt)·[x₀·cos(ω_d·t) + ((v₀+γ·x₀)/ω_d)·sin(ω_d·t)]
        v(t) = e^(-γt)·[(v₀·ω_d - γ·(v₀+γ·x₀)/ω_d·... )·cos... ]
        简洁写法：
        x(t) = e^(-γt)·[C₁·cos(ω_d·t) + C₂·sin(ω_d·t)]
        其中 C₁ = x₀, C₂ = (v₀ + γ·x₀)/ω_d

    2. 临界阻尼（ζ = 1）：
        x(t) = e^(-γt)·[x₀ + (v₀ + γ·x₀)·t]

    3. 过阻尼（ζ > 1）：
        令 α = √(γ² - ω₀²)（衰减指数）
        x(t) = e^(-γt)·[C₁·cosh(α·t) + C₂·sinh(α·t)]
        其中 C₁ = x₀, C₂ = (v₀ + γ·x₀)/α

守恒量与耗散：

    机械能 E = ½·m·v² + ½·k·x²
    当 b > 0 时，机械能单调递减（阻尼做负功）：
        dE/dt = -b·v² ≤ 0
    当 b = 0 时，机械能守恒（退化为 MEC-010）。

相图规范（延续 MEC-010）：

    - 横轴：位移 x（m）
    - 纵轴：速度 v（m/s）
    - 欠阻尼：相轨迹为内旋螺线（向原点收敛的螺旋）
    - 临界阻尼：相轨迹以最快速度趋向原点
    - 过阻尼：相轨迹缓慢趋向原点
    - 无阻尼（b=0）：退化为 MEC-010 的闭合椭圆

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(k=1.0, m=1.0, b=0.0):
    """验证基本物理参数合法性。

    参数
    ----
    k : float
        弹性系数（必须 > 0）。
    m : float
        质量（必须 > 0）。
    b : float
        阻尼系数（必须 ≥ 0）。

    返回
    ----
    None
        条件满足时静默返回。

    抛出
    ----
    AssertionError
        参数不合法时给出明确错误信息。
    """
    assert k > 0, f"弹性系数 k 必须为正，当前 k={k}"
    assert m > 0, f"质量 m 必须为正，当前 m={m}"
    assert b >= 0, f"阻尼系数 b 必须非负，当前 b={b}"


def damping_ratio(k=1.0, m=1.0, b=0.0):
    """计算阻尼比 ζ = b / (2·√(m·k))。

    参数
    ----
    k, m, b : float
        弹性系数、质量、阻尼系数。

    返回
    ----
    float
        阻尼比 ζ。
    """
    return b / (2.0 * np.sqrt(m * k))


def natural_frequency(k=1.0, m=1.0):
    """计算固有角频率 ω₀ = √(k/m)（同 MEC-010）。

    参数
    ----
    k, m : float
        弹性系数和质量。

    返回
    ----
    float
        固有角频率 ω₀（rad/s）。
    """
    return np.sqrt(k / m)


def damped_frequency(k=1.0, m=1.0, b=0.0):
    """计算阻尼角频率 ω_d = ω₀·√(1-ζ²)。

    仅对欠阻尼（ζ < 1）有物理意义。

    参数
    ----
    k, m, b : float
        弹性系数、质量、阻尼系数。

    返回
    ----
    float
        阻尼角频率 ω_d（rad/s）。

    抛出
    ----
    ValueError
        当 ζ ≥ 1 时阻尼角频率无实数定义。
    """
    zeta = damping_ratio(k, m, b)
    if zeta >= 1.0:
        raise ValueError(f"ζ={zeta:.4f} ≥ 1，阻尼角频率无实数定义")
    omega0 = natural_frequency(k, m)
    return omega0 * np.sqrt(1.0 - zeta ** 2)


def mechanical_energy(state, k=1.0, m=1.0):
    """计算机械能 E = ½·m·v² + ½·k·x²。

    参数
    ----
    state : array_like, shape (2,)
        当前状态 [x, v]。
    k : float, optional
        弹性系数（默认 1.0 N/m）。
    m : float, optional
        质量（默认 1.0 kg）。

    返回
    ----
    float
        机械能 E。
    """
    x, v = state
    return 0.5 * m * v ** 2 + 0.5 * k * x ** 2


def dynamics(t, state, k=1.0, m=1.0, b=0.0):
    """返回状态的时间导数 d(state)/dt = [dx/dt, dv/dt]。

    参数
    ----
    t : float
        当前时刻（阻尼振子显式不依赖 t，保留参数以统一接口）。
    state : array_like, shape (2,)
        当前状态 [x, v]。
    k : float, optional
        弹性系数（默认 1.0 N/m）。
    m : float, optional
        质量（默认 1.0 kg）。
    b : float, optional
        阻尼系数（默认 0.0，即无阻尼，退化为 MEC-010）。

    返回
    ----
    np.ndarray, shape (2,)
        [v, -(k/m)·x - (b/m)·v]
    """
    x, v = state
    return np.array([v, -(k / m) * x - (b / m) * v])


def analytical(t, initial_state, k=1.0, m=1.0, b=0.0):
    """阻尼振子解析解。

    根据阻尼比自动选择三种情况的解析公式：
    - 欠阻尼（0 ≤ ζ < 1）：衰减振荡
    - 临界阻尼（ζ = 1）：临界衰减
    - 过阻尼（ζ > 1）：过衰减

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    initial_state : array_like, shape (2,)
        初始状态 [x0, v0]。
    k : float, optional
        弹性系数（默认 1.0 N/m）。
    m : float, optional
        质量（默认 1.0 kg）。
    b : float, optional
        阻尼系数（默认 0.0，即无阻尼，退化为 MEC-010）。

    返回
    ----
    (x, v) : tuple
        x(t), v(t)，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    x0, v0 = initial_state

    omega0 = natural_frequency(k, m)
    gamma = b / (2.0 * m)  # 衰减率
    zeta = damping_ratio(k, m, b)

    exp_gt = np.exp(-gamma * t)

    if zeta < 1.0 - 1e-14:
        # --- 欠阻尼 ---
        omega_d = omega0 * np.sqrt(1.0 - zeta ** 2)
        cos_wd = np.cos(omega_d * t)
        sin_wd = np.sin(omega_d * t)
        C1 = x0
        C2 = (v0 + gamma * x0) / omega_d
        x = exp_gt * (C1 * cos_wd + C2 * sin_wd)
        # v = dx/dt
        v = exp_gt * (
            -gamma * (C1 * cos_wd + C2 * sin_wd)
            + (-C1 * omega_d * sin_wd + C2 * omega_d * cos_wd)
        )
    elif zeta > 1.0 + 1e-14:
        # --- 过阻尼 ---
        alpha = np.sqrt(gamma ** 2 - omega0 ** 2)
        cosh_at = np.cosh(alpha * t)
        sinh_at = np.sinh(alpha * t)
        C1 = x0
        C2 = (v0 + gamma * x0) / alpha
        x = exp_gt * (C1 * cosh_at + C2 * sinh_at)
        v = exp_gt * (
            -gamma * (C1 * cosh_at + C2 * sinh_at)
            + (C1 * alpha * sinh_at + C2 * alpha * cosh_at)
        )
    else:
        # --- 临界阻尼 ---
        C1 = x0
        C2 = v0 + gamma * x0
        x = exp_gt * (C1 + C2 * t)
        v = exp_gt * (-gamma * (C1 + C2 * t) + C2)

    return x, v
