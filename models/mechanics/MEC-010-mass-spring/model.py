"""MEC-010-mass-spring —— 模型定义（引擎无关）

标准质量—弹簧简谐振子（mass-spring harmonic oscillator）：
质量 m 在弹性力 F = -k·x 作用下沿一维方向运动。
这是最基础的振动模型，为后续 MEC-011（阻尼振子）、MEC-012（受迫振子）、
MEC-014（耦合振子）等振动系统建立标准规范。

状态向量 state = [x, v]
    x —— 位移（相对于平衡位置）
    v —— 速度

参数：
    k —— 弹性系数（spring constant，N/m，k > 0）
    m —— 质量（kg，m > 0）

动力学（一阶常微分方程）：

    弹性恢复力：F = -k·x
    牛顿第二定律：m·a = F
    因此：
        dx/dt = v
        dv/dt = -(k/m)·x

    令 ω₀ = √(k/m)（固有角频率，natural angular frequency），则：
        d²x/dt² + ω₀²·x = 0

    这是简谐振动（simple harmonic motion, SHM）的标准形式。

    周期：T = 2π/ω₀ = 2π·√(m/k)

解析解（用于校验数值解的"金标准"）：

    通解：
        x(t) = A·cos(ω₀·t + φ)
        v(t) = -A·ω₀·sin(ω₀·t + φ)

    其中 A 为振幅，φ 为初相位，由初始条件确定：
        A = √(x0² + (v0/ω₀)²)
        φ = arctan2(-v0/ω₀, x0)

    也可直接用初始条件表示：
        x(t) = x0·cos(ω₀t) + (v0/ω₀)·sin(ω₀t)
        v(t) = -x0·ω₀·sin(ω₀t) + v0·cos(ω₀t)

守恒量：

    机械能（总能量）：
        E = ½·m·v² + ½·k·x² = ½·k·A²

    机械能守恒是简谐振动的核心特征：动能和势能交替转换，
    总和恒定。在相图（x-v 平面）上，轨迹为以原点为中心的椭圆，
    半轴分别为 A（x 方向）和 A·ω₀（v 方向）。

相图（phase portrait）规范：

    本模型建立后续振动系统将沿用的相图规范：
    - 横轴：位移 x（m）
    - 纵轴：速度 v（m/s）
    - 轨迹为等能量椭圆：x²/A² + v²/(Aω₀)² = 1
    - 椭圆面积 ∝ 能量：S = π·A·Aω₀ = π·A²·ω₀ = 2π·E/(m·ω₀)

退化情形：
    x0=0, v0=0 时 A=0，质点静止在平衡位置（零振幅）。
    x0≠0, v0=0 时 φ=0，从最大位移处静止释放。

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(k=1.0, m=1.0):
    """验证基本物理参数合法性。

    参数
    ----
    k : float
        弹性系数（必须 > 0）。
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
    assert k > 0, f"弹性系数 k 必须为正，当前 k={k}"
    assert m > 0, f"质量 m 必须为正，当前 m={m}"


def angular_frequency(k=1.0, m=1.0):
    """计算固有角频率 ω₀ = √(k/m)。

    参数
    ----
    k : float
        弹性系数。
    m : float
        质量。

    返回
    ----
    float
        固有角频率 ω₀（rad/s）。
    """
    return np.sqrt(k / m)


def period(k=1.0, m=1.0):
    """计算振动周期 T = 2π/ω₀ = 2π·√(m/k)。

    参数
    ----
    k : float
        弹性系数。
    m : float
        质量。

    返回
    ----
    float
        振动周期 T（s）。
    """
    return 2.0 * np.pi * np.sqrt(m / k)


def amplitude(x0, v0, k=1.0, m=1.0):
    """从初始条件计算振幅 A = √(x0² + (v0/ω₀)²)。

    参数
    ----
    x0 : float
        初始位移。
    v0 : float
        初始速度。
    k, m : float
        弹性系数和质量。

    返回
    ----
    float
        振幅 A。
    """
    omega0 = angular_frequency(k, m)
    return np.sqrt(x0 ** 2 + (v0 / omega0) ** 2)


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


def dynamics(t, state, k=1.0, m=1.0):
    """返回状态的时间导数 d(state)/dt = [dx/dt, dv/dt]。

    参数
    ----
    t : float
        当前时刻（简谐振动显式不依赖 t，保留参数以统一接口）。
    state : array_like, shape (2,)
        当前状态 [x, v]。
    k : float, optional
        弹性系数（默认 1.0 N/m）。
    m : float, optional
        质量（默认 1.0 kg）。

    返回
    ----
    np.ndarray, shape (2,)
        [v, -(k/m)·x]
    """
    x, v = state
    return np.array([v, -(k / m) * x])


def analytical(t, initial_state, k=1.0, m=1.0):
    """简谐振动解析解。

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

    返回
    ----
    (x, v) : tuple
        x(t), v(t)，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    x0, v0 = initial_state
    omega0 = angular_frequency(k, m)
    cos_wt = np.cos(omega0 * t)
    sin_wt = np.sin(omega0 * t)
    x = x0 * cos_wt + (v0 / omega0) * sin_wt
    v = -x0 * omega0 * sin_wt + v0 * cos_wt
    return x, v
