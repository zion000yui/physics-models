"""MEC-023-gyroscopic-precession —— 模型定义（引擎无关）

高速自旋对称陀螺的慢进动近似模型（slow precession approximation for a
fast-spinning symmetric top）。

本模型明确区分三个层次：
  1. 精确方程：Routhian 降维后的完整 Euler-Lagrange 方程（dynamics 实现）
  2. 近似方程：稳态条件下对二次方程取慢根的展开
  3. 稳态解：Ω_p = mgl/(I₃ω_s)（慢进动近似下的解析结果）

本模型不是完整 3D Euler top / Euler equations。完整模型留到后续独立模型。

=== 物理系统 ===

对称陀螺（symmetric top），尖端固定在一个点。陀螺绕自身对称轴高速自旋，
在重力力矩作用下做进动（precession）和章动（nutation）。

=== 坐标系：欧拉角 (θ, φ, ψ) ===

  θ：倾斜角（自旋轴与竖直向上方向的夹角，θ=0 为直立，θ=π 为倒挂）
  φ：进动角（自旋轴绕竖直轴的方位角）
  ψ：自旋角（陀螺绕自身对称轴的转角，循环坐标）

=== 假设 ===

  1. 对称陀螺：I₁ = I₂ ≠ I₃（横向转动惯量相等，轴向转动惯量不同）
  2. 高速自旋：ω_s = ψ̇ + φ̇cos θ >> |θ̇|, |φ̇|
     （自旋角速度远大于进动和章动角速度）
  3. 慢进动近似：稳态分析中忽略 Ω_p² 项
     （Ω_p << ω_s，即进动远慢于自旋）
  4. 定点运动：陀螺尖端固定，只有转动，无平动
  5. 自旋角速度视为常数（p_ψ = I₃ω_s 守恒，因为 ψ 是循环坐标）

=== Routhian 降维 ===

完整拉格朗日量（三个自由度 θ, φ, ψ）：
  L = ½I₁(θ̇² + φ̇²sin²θ) + ½I₃(ψ̇ + φ̇cos θ)² - mgl·cos θ

ψ 是循环坐标（L 不显含 ψ），因此其共轭动量守恒：
  p_ψ = ∂L/∂ψ̇ = I₃(ψ̇ + φ̇cos θ) = I₃ω_s = const

消去 ψ 自由度后，有效拉格朗日量（Routhian）为：
  L_eff = ½I₁(θ̇² + φ̇²sin²θ) + I₃ω_s·φ̇·cos θ - mgl·cos θ

被消去的自由度：ψ（自旋角）
由此产生的守恒量：p_ψ = I₃ω_s（轴向角动量）

另外，φ 也是循环坐标（L_eff 不显含 φ），因此：
  p_φ = ∂L_eff/∂φ̇ = I₁φ̇sin²θ + I₃ω_s·cos θ = const
  这是总角动量的竖直分量（重力力矩沿水平方向，竖直分量无外力矩）。

=== 状态变量 ===

state = [theta, theta_dot, phi, phi_dot]（4D）

消去 ψ 后的 2 自由度系统：倾斜角 θ 和进动角 φ。
自旋角速度 ω_s 作为参数传入（常数，因 p_ψ 守恒）。

  theta      —— 倾斜角（rad，自旋轴与竖直方向的夹角）
  theta_dot  —— 章动角速度 dθ/dt（rad/s）
  phi        —— 进动角（rad）
  phi_dot    —— 进动角速度 dφ/dt（rad/s）

=== 参数 ===

  m       —— 陀螺质量（kg）
  l       —— 支点到质心的距离（m）
  I1      —— 横向转动惯量 I₁ = I₂（kg·m²）
  I3      —— 轴向转动惯量 I₃（kg·m²）
  omega_s —— 自旋角速度 ω_s（rad/s，视为常数）
  g       —— 重力加速度（m/s²）

=== 核心方程（精确 Routhian Euler-Lagrange 方程） ===

对 θ：
  I₁·θ̈ = I₁·φ̇²·sin θ·cos θ - I₃·ω_s·φ̇·sin θ + mgl·sin θ

对 φ（守恒量 p_φ 对时间求导 = 0）：
  I₁·φ̈·sin²θ + 2I₁·φ̇·θ̇·sin θ·cos θ - I₃·ω_s·θ̇·sin θ = 0

量纲检查：
  I₁·θ̈     [kg·m²·rad/s²] = [N·m]（力矩） ✓
  I₁·φ̇²·sin θ·cos θ [kg·m²·(rad/s)²] = [N·m] ✓
  I₃·ω_s·φ̇·sin θ [kg·m²·(rad/s)²] = [N·m] ✓
  mgl·sin θ [kg·m/s²·m] = [N·m] ✓

=== 稳态进动解 ===

稳态条件：θ = θ₀（常数），θ̇ = θ̈ = 0，φ̇ = Ω_p（常数）。

代入 θ 方程：
  0 = sin θ₀·(I₁·Ω_p²·cos θ₀ - I₃·ω_s·Ω_p + mgl)

情况 1：sin θ₀ ≠ 0（θ₀ ≠ 0, π），两边消去 sin θ₀：
  I₁·cos θ₀·Ω_p² - I₃·ω_s·Ω_p + mgl = 0    （二次方程，无 sin θ₀ 因子）

  精确解：
    Ω_p = [I₃ω_s ± √(I₃²ω_s² - 4I₁mgl·cos θ₀)] / (2I₁·cos θ₀)

  慢进动近似（Ω_p << ω_s，忽略 Ω_p² 项）：
    Ω_p ≈ mgl / (I₃·ω_s)

  快进动（另一个根）：
    Ω_p_fast ≈ I₃·ω_s / (I₁·cos θ₀)

情况 2：sin θ₀ = 0（θ₀ = 0 或 π），原方程恒等于 0 = 0，
  任意 Ω_p 都是稳态（重力力矩为零，无重力驱动进动）。

=== sin θ₀ 因子的说明 ===

重力力矩大小 |τ| = mgl·sin θ₀
角动量变化率 |dL/dt| = Ω_p·L_s·sin θ₀ = Ω_p·I₃ω_s·sin θ₀

sin θ₀ 在分子（力矩）和分母（角动量变化率）中同时出现并消去，
因此最终 Ω_p = mgl/(I₃ω_s) 不含 sin θ₀ 因子。
但此消去仅在 sin θ₀ ≠ 0 时合法。

=== 守恒量 ===

  p_ψ = I₃·ω_s = const（自旋角动量，Routhian 降维产生）
  p_φ = I₁·φ̇·sin²θ + I₃·ω_s·cos θ = const（竖直角动量，φ 循环产生）

  有效能量（Routhian 能量）：
    E_eff = ½I₁(θ̇² + φ̇²·sin²θ) + mgl·cos θ = const
  （L_eff 不显含 t，故 E_eff 守恒。注意：这不是系统总机械能，
   总机械能 = E_eff + ½I₃ω_s²，其中 ½I₃ω_s² 是常值自旋动能。）

=== 极限分析 ===

  ω_s → ∞：Ω_p → 0（进动极慢），章动振幅 → 0（陀螺效应极强，几乎不倾倒）
    物理含义：自旋越快，陀螺越"刚性"，进动越慢，章动越小。

  ω_s = 0：高速自旋假设和慢进动近似全部失效。
    Routhian 方程本身仍然成立（退化为复摆 L_eff = ½I₁(θ̇²+φ̇²sin²θ) - mgl·cos θ，
    2 自由度复摆），但稳态公式 Ω_p = mgl/(I₃ω_s) 发散，不再适用。
    不能简单宣称"模型退化为 MEC-015 单摆"，因为：
    - MEC-015 是单自由度（1D 角度），此处是 2 自由度（θ 和 φ）
    - 需要额外约束 φ̇ = 0 才退化为 MEC-021 形式（复摆方程 I₁θ̈ = mgl sin θ）
    - 当 I₁ = mL² 时，该复摆方程与 MEC-015 单摆方程一致

=== 适用范围和局限性 ===

  适用：
    - 对称陀螺（I₁ = I₂ ≠ I₃）
    - 高速自旋（ω_s >> √(mgl/I₁)）
    - 一个固定点
    - 稳态或近稳态进动

  不适用：
    - 非对称陀螺（I₁ ≠ I₂）
    - 低速或零自旋
    - 自由陀螺（无固定点，需要 3D 欧拉方程）
    - 大幅度章动（偏离稳态过远）

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(m=1.0, l=1.0, I1=1.0, I3=1.0, omega_s=10.0, g=9.81):
    """验证基本物理参数合法性。

    参数
    ----
    m : float
        陀螺质量（必须 > 0）。
    l : float
        支点到质心距离（必须 > 0）。
    I1 : float
        横向转动惯量（必须 > 0）。
    I3 : float
        轴向转动惯量（必须 > 0）。
    omega_s : float
        自旋角速度（必须 > 0；omega_s=0 时高速自旋假设失效）。
    g : float
        重力加速度（必须 > 0）。

    返回
    ----
    None
        条件满足时静默返回。

    抛出
    ----
    AssertionError
        参数不合法时给出明确错误信息。
    ValueError
        当 omega_s=0 时发出警告（高速自旋假设失效）。
    """
    assert m > 0, f"质量 m 必须为正，当前 m={m}"
    assert l > 0, f"距离 l 必须为正，当前 l={l}"
    assert I1 > 0, f"横向转动惯量 I1 必须为正，当前 I1={I1}"
    assert I3 > 0, f"轴向转动惯量 I3 必须为正，当前 I3={I3}"
    assert omega_s >= 0, f"自旋角速度 omega_s 必须非负，当前 omega_s={omega_s}"
    assert g > 0, f"重力加速度 g 必须为正，当前 g={g}"
    if omega_s == 0:
        raise ValueError(
            "omega_s=0 时高速自旋假设失效，慢进动近似不适用。"
            "Routhian 方程本身仍成立（退化为复摆），但稳态公式发散。"
        )


def steady_state_precession(m=1.0, l=1.0, I1=1.0, I3=1.0,
                            omega_s=10.0, g=9.81, theta_0=np.pi / 4):
    """计算慢进动近似下的稳态进动角速度 Ω_p = mgl / (I₃·ω_s)。

    这是近似结果，仅在高速自旋（ω_s >> √(mgl/I₁)）时有效。
    不含 sin(θ₀) 因子（详见模块 docstring 中 sin θ₀ 因子的说明）。

    参数
    ----
    m, l, I1, I3, omega_s, g : float
        物理参数。
    theta_0 : float
        稳态倾斜角（rad）。此参数不影响 Ω_p 的近似值（sin θ₀ 已消去），
        但影响精确解（见 exact_steady_state_precession）。

    返回
    ----
    float
        慢进动角速度 Ω_p（rad/s）。
    """
    return m * g * l / (I3 * omega_s)


def exact_steady_state_precession(m=1.0, l=1.0, I1=1.0, I3=1.0,
                                  omega_s=10.0, g=9.81, theta_0=np.pi / 4):
    """计算精确稳态进动角速度（解二次方程）。

    I₁·cos θ₀·Ω_p² - I₃·ω_s·Ω_p + mgl = 0

    返回慢进动和快进动两个根。

    参数
    ----
    m, l, I1, I3, omega_s, g : float
        物理参数。
    theta_0 : float
        稳态倾斜角（rad）。

    返回
    ----
    (omega_slow, omega_fast) : tuple of float
        慢进动和快进动角速度。如果判别式为负（无稳态），返回 (None, None)。
    """
    cos_t0 = np.cos(theta_0)
    a_coeff = I1 * cos_t0
    b_coeff = -I3 * omega_s
    c_coeff = m * g * l

    if abs(a_coeff) < 1e-14:
        # cos θ₀ ≈ 0（θ₀ = π/2）：退化为线性方程 -I₃ω_s·Ω_p + mgl = 0
        omega_slow = c_coeff / (-b_coeff)
        return omega_slow, np.inf

    discriminant = b_coeff ** 2 - 4 * a_coeff * c_coeff
    if discriminant < 0:
        return None, None
    sqrt_disc = np.sqrt(discriminant)
    omega_slow = (-b_coeff - sqrt_disc) / (2 * a_coeff)
    omega_fast = (-b_coeff + sqrt_disc) / (2 * a_coeff)
    return omega_slow, omega_fast


def conjugate_momentum_phi(state, I1=1.0, I3=1.0, omega_s=10.0):
    """计算守恒量 p_φ = I₁·φ̇·sin²θ + I₃·ω_s·cos θ。

    φ 是循环坐标，p_φ 守恒（竖直角动量守恒）。

    参数
    ----
    state : array_like, shape (4,)
        状态 [theta, theta_dot, phi, phi_dot]。
    I1, I3, omega_s : float
        物理参数。

    返回
    ----
    float
        共轭动量 p_φ。
    """
    theta, _, _, phi_dot = state
    return I1 * phi_dot * np.sin(theta) ** 2 + I3 * omega_s * np.cos(theta)


def effective_energy(state, m=1.0, l=1.0, I1=1.0, I3=1.0, omega_s=10.0, g=9.81):
    """计算 Routhian 有效能量 E_eff = ½I₁(θ̇² + φ̇²sin²θ) + mgl·cos θ。

    L_eff 不显含 t，故 E_eff 守恒。
    注意：这不是系统总机械能。总机械能 = E_eff + ½I₃ω_s²。

    参数
    ----
    state : array_like, shape (4,)
        状态 [theta, theta_dot, phi, phi_dot]。
    m, l, I1, I3, omega_s, g : float
        物理参数。

    返回
    ----
    float
        有效能量 E_eff。
    """
    theta, theta_dot, _, phi_dot = state
    sin_t = np.sin(theta)
    ke = 0.5 * I1 * (theta_dot ** 2 + phi_dot ** 2 * sin_t ** 2)
    pe = m * g * l * np.cos(theta)
    return ke + pe


def dynamics(t, state, m=1.0, l=1.0, I1=1.0, I3=1.0, omega_s=10.0, g=9.81):
    """返回状态的时间导数（精确 Routhian 方程，非近似）。

    对 θ：I₁·θ̈ = I₁·φ̇²·sin θ·cos θ - I₃·ω_s·φ̇·sin θ + mgl·sin θ
    对 φ：d/dt(p_φ) = 0 的展开

    参数
    ----
    t : float
        当前时刻（保守系统不依赖 t，保留以统一接口）。
    state : array_like, shape (4,)
        状态 [theta, theta_dot, phi, phi_dot]。
    m, l, I1, I3, omega_s, g : float
        物理参数。

    返回
    ----
    np.ndarray, shape (4,)
        [theta_dot, theta_ddot, phi_dot, phi_ddot]
    """
    theta, theta_dot, _, phi_dot = state
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # θ 方程（精确 Euler-Lagrange）
    theta_ddot = (I1 * phi_dot ** 2 * sin_t * cos_t
                  - I3 * omega_s * phi_dot * sin_t
                  + m * g * l * sin_t) / I1

    # φ 方程（p_φ 守恒对时间求导）
    if abs(sin_t) < 1e-14:
        # θ ≈ 0 或 π（欧拉角奇点），避免除零
        phi_ddot = 0.0
    else:
        phi_ddot = theta_dot * (I3 * omega_s - 2 * I1 * phi_dot * cos_t) \
                   / (I1 * sin_t)

    return np.array([theta_dot, theta_ddot, phi_dot, phi_ddot])


def analytical(t, initial_state, m=1.0, l=1.0, I1=1.0, I3=1.0,
               omega_s=10.0, g=9.81):
    """稳态慢进动近似解析解。

    提供 θ = θ₀（常数）、φ = φ₀ + Ω_p·t 的稳态解。
    Ω_p = mgl / (I₃·ω_s)（慢进动近似）。

    注意：这是近似稳态解，不是一般运动的精确解析解。
    仅在以下条件下有效：
    - 高速自旋（ω_s >> √(mgl/I₁)）
    - 初始条件接近稳态（θ̇₀ ≈ 0, φ̇₀ ≈ Ω_p）
    - sin θ₀ ≠ 0（θ₀ ≠ 0, π）

    参数
    ----
    t : float 或 array_like
        时间点。
    initial_state : array_like, shape (4,)
        初始状态 [theta0, theta_dot0, phi0, phi_dot0]。
        仅使用 theta0（稳态倾斜角）和 phi0（初始进动角），
        theta_dot0 和 phi_dot0 被替换为稳态值 0 和 Ω_p。
    m, l, I1, I3, omega_s, g : float
        物理参数。

    返回
    ----
    (theta, theta_dot, phi, phi_dot) : tuple
        稳态解的四个分量，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    theta_0, _, phi_0, _ = initial_state
    omega_p = steady_state_precession(m, l, I1, I3, omega_s, g, theta_0)

    theta = np.full_like(t, theta_0)
    theta_dot = np.zeros_like(t)
    phi = phi_0 + omega_p * t
    phi_dot = np.full_like(t, omega_p)

    return theta, theta_dot, phi, phi_dot
