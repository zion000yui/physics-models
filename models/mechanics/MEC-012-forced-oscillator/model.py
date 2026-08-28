"""MEC-012-forced-oscillator —— 模型定义（引擎无关）

受迫阻尼振子（forced damped harmonic oscillator）：质量 m 在弹性力
F = -k·x、线性阻尼力 F_d = -b·v 和周期性外力 F_ext = F₀·cos(ωt)
共同作用下沿一维方向运动。是 MEC-010 和 MEC-011 的进一步推广。

状态向量 state = [x, v]
    x —— 位移（相对于平衡位置）
    v —— 速度

参数：
    k —— 弹性系数（spring constant，N/m，k > 0）
    m —— 质量（kg，m > 0）
    b —— 阻尼系数（damping coefficient，N·s/m，b ≥ 0）
    F0 —— 驱动力幅值（driving force amplitude，N，F0 ≥ 0）
    omega —— 驱动角频率（driving angular frequency，rad/s，omega ≥ 0）

动力学（一阶常微分方程）：

    合力 = 弹性力 + 阻尼力 + 驱动力
        F = -k·x - b·v + F₀·cos(ωt)

    牛顿第二定律：m·a = F
    因此：
        dx/dt = v
        dv/dt = -(k/m)·x - (b/m)·v + (F0/m)·cos(ωt)

    令 ω₀ = √(k/m)（固有角频率），γ = b/(2m)（衰减率），
    则运动方程为：
        d²x/dt² + 2γ·dx/dt + ω₀²·x = (F₀/m)·cos(ωt)

瞬态与稳态的分解：

    通解 = 瞬态解（齐次通解） + 稳态解（特解）

    瞬态解 = MEC-011 的阻尼振子解（三种状态，随时间衰减至零）
    稳态解 = 持续的受迫振荡

    稳态解的形式：
        x_ss(t) = A_ss·cos(ωt - δ)
    其中：
        A_ss = (F₀/m) / √((ω₀²-ω²)² + (2γω)²)    （稳态振幅）
        tan(δ) = 2γω / (ω₀²-ω²)                     （相位滞后）

幅频响应与共振：

    稳态振幅 A_ss 随驱动频率 ω 变化的曲线称为幅频响应曲线。
    当 b > 0 时，最大响应频率为：
        ω_max = √(ω₀² - 2γ²) = ω₀·√(1 - 2ζ²)
    仅在 ζ < 1/√2 时存在（否则振幅随 ω 单调递减，无共振峰）。
    共振时最大振幅为：
        A_max = (F₀/m) / (2γω₀·√(1-ζ²))

    当 b = 0（无阻尼）时，ω → ω₀ 会导致振幅发散（理想共振）。

能量关系：

    稳态时，驱动力注入的能量与阻尼耗散的能量平衡：
        P_in = ⟨F₀·cos(ωt)·v(t)⟩ = (1/2)·F₀·A_ss·ω·sin(δ)
        P_diss = ⟨b·v²⟩ = (1/2)·b·A_ss²·ω²
    稳态时 P_in = P_diss。

相图规范（延续 MEC-010/011）：

    - 横轴：位移 x（m）
    - 纵轴：速度 v（m/s）
    - 瞬态阶段：与 MEC-011 相同的衰减轨迹
    - 稳态阶段：相图收敛为极限环（limit cycle）——椭圆
    - 椭圆半轴：x 方向 = A_ss，v 方向 = A_ss·ω
    - 无阻尼时无极限环（振幅发散或守恒）

退化关系：

    F0=0, b=0 → MEC-010（简谐振子）
    F0=0, b>0 → MEC-011（阻尼振子）
    F0=0       → 齐次方程，瞬态解衰减至零

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(k=1.0, m=1.0, b=0.0, F0=0.0, omega=0.0):
    """验证基本物理参数合法性。

    参数
    ----
    k : float
        弹性系数（必须 > 0）。
    m : float
        质量（必须 > 0）。
    b : float
        阻尼系数（必须 ≥ 0）。
    F0 : float
        驱动力幅值（必须 ≥ 0）。
    omega : float
        驱动角频率（必须 ≥ 0）。

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
    assert F0 >= 0, f"驱动力幅值 F0 必须非负，当前 F0={F0}"
    assert omega >= 0, f"驱动角频率 omega 必须非负，当前 omega={omega}"


def natural_frequency(k=1.0, m=1.0):
    """计算固有角频率 ω₀ = √(k/m)（同 MEC-010/011）。

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


def damping_ratio(k=1.0, m=1.0, b=0.0):
    """计算阻尼比 ζ = b / (2·√(m·k))（同 MEC-011）。

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


def steady_state_amplitude(k=1.0, m=1.0, b=0.0, F0=1.0, omega=1.0):
    """计算稳态振幅 A_ss。

    A_ss = (F₀/m) / √((ω₀²-ω²)² + (2γω)²)

    当 b=0 且 ω=ω₀ 时返回 inf（理想共振）。

    参数
    ----
    k, m, b, F0, omega : float
        物理参数。

    返回
    ----
    float
        稳态振幅 A_ss。
    """
    omega0 = natural_frequency(k, m)
    gamma = b / (2.0 * m)
    denom_sq = (omega0 ** 2 - omega ** 2) ** 2 + (2 * gamma * omega) ** 2
    if denom_sq < 1e-30:
        return np.inf
    return (F0 / m) / np.sqrt(denom_sq)


def steady_state_phase(k=1.0, m=1.0, b=0.0, omega=1.0):
    """计算稳态相位滞后 δ。

    tan(δ) = 2γω / (ω₀²-ω²)
    δ = arctan2(2γω, ω₀²-ω²)

    参数
    ----
    k, m, b, omega : float
        物理参数。

    返回
    ----
    float
        相位滞后 δ（rad，0 ≤ δ ≤ π）。
    """
    omega0 = natural_frequency(k, m)
    gamma = b / (2.0 * m)
    return np.arctan2(2 * gamma * omega, omega0 ** 2 - omega ** 2)


def resonance_frequency(k=1.0, m=1.0, b=0.0):
    """计算最大响应频率 ω_max。

    ω_max = √(ω₀² - 2γ²) = ω₀·√(1 - 2ζ²)

    仅在 ζ < 1/√2 时存在。否则返回 None（振幅单调递减，无共振峰）。

    参数
    ----
    k, m, b : float
        物理参数。

    返回
    ----
    float 或 None
        最大响应频率 ω_max（rad/s），或 None（无共振峰）。
    """
    zeta = damping_ratio(k, m, b)
    omega0 = natural_frequency(k, m)
    if zeta >= 1.0 / np.sqrt(2.0):
        return None
    return omega0 * np.sqrt(1.0 - 2.0 * zeta ** 2)


def mechanical_energy(state, k=1.0, m=1.0):
    """计算机械能 E = ½·m·v² + ½·k·x²（同 MEC-010/011）。

    注意：受迫阻尼系统的机械能不守恒（驱动力注入，阻尼耗散）。

    参数
    ----
    state : array_like, shape (2,)
        当前状态 [x, v]。
    k, m : float
        弹性系数和质量。

    返回
    ----
    float
        机械能 E。
    """
    x, v = state
    return 0.5 * m * v ** 2 + 0.5 * k * x ** 2


def dynamics(t, state, k=1.0, m=1.0, b=0.0, F0=0.0, omega=0.0):
    """返回状态的时间导数 d(state)/dt = [dx/dt, dv/dt]。

    参数
    ----
    t : float
        当前时刻（受迫振子显式依赖 t，通过驱动力 F₀·cos(ωt)）。
    state : array_like, shape (2,)
        当前状态 [x, v]。
    k : float, optional
        弹性系数（默认 1.0 N/m）。
    m : float, optional
        质量（默认 1.0 kg）。
    b : float, optional
        阻尼系数（默认 0.0，即无阻尼）。
    F0 : float, optional
        驱动力幅值（默认 0.0，即无外力）。
    omega : float, optional
        驱动角频率（默认 0.0）。

    返回
    ----
    np.ndarray, shape (2,)
        [v, -(k/m)·x - (b/m)·v + (F0/m)·cos(ωt)]
    """
    x, v = state
    a = -(k / m) * x - (b / m) * v + (F0 / m) * np.cos(omega * t)
    return np.array([v, a])


def analytical(t, initial_state, k=1.0, m=1.0, b=0.0, F0=0.0, omega=0.0):
    """受迫阻尼振子解析解（瞬态 + 稳态）。

    通解 = 瞬态解（齐次通解，同 MEC-011） + 稳态解（特解）

    瞬态解根据阻尼比自动选择三种情况（欠阻尼/临界/过阻尼）。
    稳态解为 x_ss(t) = A_ss·cos(ωt - δ)。

    当 F0=0 时，稳态解为零，退化为 MEC-011。
    当 F0=0 且 b=0 时，退化为 MEC-010。

    注意：当 b=0 且 ω=ω₀（无阻尼共振）时，稳态解形式不同
    （振幅线性增长），本函数对此情况抛出 ValueError。

    参数
    ----
    t : float 或 array_like
        时间点（标量或数组均可）。
    initial_state : array_like, shape (2,)
        初始状态 [x0, v0]。
    k, m, b, F0, omega : float
        物理参数。

    返回
    ----
    (x, v) : tuple
        x(t), v(t)，形状与 t 一致。

    抛出
    ----
    ValueError
        当 b=0 且 ω=ω₀（无阻尼理想共振）时，解的形式不同（振幅线性增长），
        本函数不支持此特殊情况。
    """
    t = np.asarray(t, dtype=float)
    x0, v0 = initial_state

    omega0 = natural_frequency(k, m)
    gamma = b / (2.0 * m)
    zeta = damping_ratio(k, m, b)

    # --- 稳态解 ---
    if F0 > 0 and omega > 0:
        # 检查无阻尼共振
        if b == 0 and np.isclose(omega, omega0, rtol=1e-14):
            raise ValueError(
                "无阻尼理想共振（b=0, ω=ω₀）不支持标准解析解，"
                "稳态振幅线性增长。请使用 b>0 或 ω≠ω₀。"
            )
        A_ss = steady_state_amplitude(k, m, b, F0, omega)
        delta = steady_state_phase(k, m, b, omega)
        x_ss = A_ss * np.cos(omega * t - delta)
        v_ss = -A_ss * omega * np.sin(omega * t - delta)
    else:
        x_ss = np.zeros_like(t)
        v_ss = np.zeros_like(t)

    # --- 瞬态解的初始条件 ---
    # 瞬态解 x_tr(t) 满足 x_tr(0) = x0 - x_ss(0), v_tr(0) = v0 - v_ss(0)
    x_tr0 = x0 - x_ss[0] if isinstance(x_ss, np.ndarray) and x_ss.ndim > 0 else x0 - x_ss
    v_tr0 = v0 - v_ss[0] if isinstance(v_ss, np.ndarray) and v_ss.ndim > 0 else v0 - v_ss

    exp_gt = np.exp(-gamma * t)

    if zeta < 1.0 - 1e-14:
        # 欠阻尼
        omega_d = omega0 * np.sqrt(1.0 - zeta ** 2)
        cos_wd = np.cos(omega_d * t)
        sin_wd = np.sin(omega_d * t)
        C1 = x_tr0
        C2 = (v_tr0 + gamma * x_tr0) / omega_d
        x_tr = exp_gt * (C1 * cos_wd + C2 * sin_wd)
        v_tr = exp_gt * (
            -gamma * (C1 * cos_wd + C2 * sin_wd)
            + (-C1 * omega_d * sin_wd + C2 * omega_d * cos_wd)
        )
    elif zeta > 1.0 + 1e-14:
        # 过阻尼
        alpha = np.sqrt(gamma ** 2 - omega0 ** 2)
        cosh_at = np.cosh(alpha * t)
        sinh_at = np.sinh(alpha * t)
        C1 = x_tr0
        C2 = (v_tr0 + gamma * x_tr0) / alpha
        x_tr = exp_gt * (C1 * cosh_at + C2 * sinh_at)
        v_tr = exp_gt * (
            -gamma * (C1 * cosh_at + C2 * sinh_at)
            + (C1 * alpha * sinh_at + C2 * alpha * cosh_at)
        )
    else:
        # 临界阻尼
        C1 = x_tr0
        C2 = v_tr0 + gamma * x_tr0
        x_tr = exp_gt * (C1 + C2 * t)
        v_tr = exp_gt * (-gamma * (C1 + C2 * t) + C2)

    x = x_tr + x_ss
    v = v_tr + v_ss

    return x, v
