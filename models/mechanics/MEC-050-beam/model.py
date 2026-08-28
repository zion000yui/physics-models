"""MEC-050-beam — 模型定义（引擎无关）

欧拉-伯努利梁（Euler-Bernoulli beam）：连续体弯曲理论。
梁的横向位移 w(x,t) 满足 4 阶偏微分方程：

  EI ∂⁴w/∂x⁴ + ρA ∂²w/∂t² = q(x,t)

  E  — 杨氏模量 (Pa)
  I  — 截面惯性矩 (m⁴)
  ρ  — 密度 (kg/m³)
  A  — 截面积 (m²)
  L  — 梁长 (m)
  q  — 分布载荷 (N/m)

=== 物理系统 ===

  细长梁（长度远大于截面尺寸），满足平截面假定（Bernoulli-Euler 假设）：
  - 变形前垂直于中性轴的截面，变形后仍垂直于中性轴且保持平面
  - 忽略剪切变形和转动惯量（适用于 L/h > 10 的细长梁）

=== 静态分析 ===

  EI w''''(x) = q(x)

  边界条件（本模型覆盖两种经典支承）：

  1) 悬臂梁（cantilever, fixed-free）：
     x=0: w=0, w'=0     (固定端)
     x=L: M=0, V=0       (自由端, M=-EIw''=0, V=-EIw'''=0)

  2) 简支梁（simply supported, pinned-pinned）：
     x=0: w=0, M=0       (铰支)
     x=L: w=0, M=0       (铰支)

=== 解析解（均布载荷 q）===

  悬臂梁：
    w(x) = q x² (6L² - 4Lx + x²) / (24 EI)
    θ(x) = w'(x) = q x (3L² - 3Lx + x²) / (6 EI)
    M(x) = -EI w''(x) = -q (L-x)² / 2
    V(x) = -EI w'''(x) = q (L-x)

  简支梁：
    w(x) = q x (L³ - 2Lx² + x³) / (24 EI)
    θ(x) = w'(x) = q (L³ - 6Lx² + 4x³) / (24 EI)
    M(x) = -EI w''(x) = q x (L - x) / 2
    V(x) = -EI w'''(x) = q (L/2 - x)

  端部最大挠度：
    悬臂：w_max = w(L) = q L⁴ / (8 EI)
    简支：w_max = w(L/2) = 5 q L⁴ / (384 EI)

=== 动态分析（模态法）===

  无阻尼自由振动：q(x,t)=0
  分离变量 w(x,t) = φ(x) T(t)，得：
    EI φ''''(x) = ω² ρA φ(x)       (空间方程)
    T''(t) + ω² T(t) = 0            (时间方程)

  空间方程的通解：
    φ(x) = C1 cos(βx) + C2 sin(βx) + C3 cosh(βx) + C4 sinh(βx)
    其中 β⁴ = ρA ω² / EI，即 ω = β² √(EI / (ρA))

  频率方程与特征值：
    悬臂：cos(βL) cosh(βL) = -1
      β_n L = 1.875104, 4.694091, 7.854757, 10.995541, ...
      (近似公式：(n-1/2)π, n≥2 精度好)

    简支：sin(βL) = 0
      β_n L = nπ, n=1,2,3,...

  模态形状：
    悬臂（σ_n = (sin(β_n L) + sinh(β_n L)) / (cos(β_n L) + cosh(β_n L))）：
      φ_n(x) = cosh(β_n x) - cos(β_n x) - σ_n [sinh(β_n x) - sin(β_n x)]

    简支：
      φ_n(x) = sin(nπ x / L)

  模态质量与刚度（正交归一化后）：
    ∫₀ᴸ ρA φ_n(x) φ_m(x) dx = δ_nm  →  模态质量 = 1
    ∫₀ᴸ EI φ_n''(x) φ_m''(x) dx = ω_n² δ_nm  →  模态刚度 = ω_n²

  模态动力学（无阻尼）：
    q̈_n + ω_n² q_n = F_n(t)
    F_n(t) = ∫₀ᴸ q(x,t) φ_n(x) dx / m_n  （m_n=1 归一化后）

  能量：
    T = ½ Σ q̇_n²  (模态质量归一化)
    U = ½ Σ ω_n² q_n²
    E = T + U

=== 与已有 MEC 模型的关系 ===

  MEC-014 耦合振子的连续极限 → MEC-050 梁（离散→连续）
  MEC-050 是 MEC-051（板）和 MEC-053（3D 弹性体）的一维基础
  静态退化为 4 阶 ODE → 动态退化为无限维振动系统

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np

# ---------------------------------------------------------------------------
# 已知特征值（cantilever 前若干阶 β_n * L）
# ---------------------------------------------------------------------------
_CANTILEVER_BETA_L = np.array([
    1.8751040686, 4.6940911330, 7.8547573812, 10.9955407350,
    14.1371683910, 17.2787595330, 20.4203522510, 23.5619440560,
])


def validate_parameters(E=2.0e11, I=1.0e-8, rho=7850.0, A=1.0e-4,
                        L=1.0, q_load=0.0):
    """验证物理参数合法性。"""
    assert E > 0, f"杨氏模量 E 必须为正，当前 E={E}"
    assert I > 0, f"截面惯性矩 I 必须为正，当前 I={I}"
    assert rho > 0, f"密度 rho 必须为正，当前 rho={rho}"
    assert A > 0, f"截面积 A 必须为正，当前 A={A}"
    assert L > 0, f"梁长 L 必须为正，当前 L={L}"


def bending_stiffness(E, I):
    """抗弯刚度 EI。"""
    return E * I


def mass_per_length(rho, A):
    """单位长度质量 μ = ρA。"""
    return rho * A


# ---------------------------------------------------------------------------
# 静态解析解
# ---------------------------------------------------------------------------

def static_cantilever_uniform_load(x, q, L, E, I):
    """悬臂梁在均布载荷 q 下的静态解。

    w(x) = q x² (6L² - 4Lx + x²) / (24 EI)

    返回 (w, theta, M, V)：
      w     — 挠度
      theta — 转角 w'(x)
      M     — 弯矩 = -EI w''
      V     — 剪力 = -EI w'''
    """
    x = np.asarray(x, dtype=float)
    EI = E * I
    w = q * x**2 * (6 * L**2 - 4 * L * x + x**2) / (24 * EI)
    theta = q * x * (3 * L**2 - 3 * L * x + x**2) / (6 * EI)
    # w'' = q (L - x)² / (2 EI) → M = -EI w'' = -q (L-x)²/2
    M = -q * (L - x)**2 / 2
    # w''' = -q (L - x) / EI → V = -EI w''' = q (L - x)
    V = q * (L - x)
    return w, theta, M, V


def static_simply_supported_uniform_load(x, q, L, E, I):
    """简支梁在均布载荷 q 下的静态解。

    w(x) = q x (L³ - 2Lx² + x³) / (24 EI)

    返回 (w, theta, M, V)。
    """
    x = np.asarray(x, dtype=float)
    EI = E * I
    w = q * x * (L**3 - 2 * L * x**2 + x**3) / (24 * EI)
    theta = q * (L**3 - 6 * L * x**2 + 4 * x**3) / (24 * EI)
    # w'' = q (L - 2x) / (2 EI) → M = -q x (L - x) / 2
    M = q * x * (L - x) / 2
    # w''' = -q / EI → V = q (L/2 - x)
    V = q * (L / 2 - x)
    return w, theta, M, V


def static_cantilever_tip_load(x, P, L, E, I):
    """悬臂梁在自由端集中力 P 下的静态解。

    w(x) = P x² (3L - x) / (6 EI)
    """
    x = np.asarray(x, dtype=float)
    EI = E * I
    w = P * x**2 * (3 * L - x) / (6 * EI)
    theta = P * x * (2 * L - x) / (2 * EI)
    M = -P * (L - x)
    V = P * np.ones_like(x)
    return w, theta, M, V


def max_deflection_cantilever(q, L, E, I, load_type='uniform'):
    """悬臂梁最大挠度。"""
    EI = E * I
    if load_type == 'uniform':
        return q * L**4 / (8 * EI)
    elif load_type == 'tip':
        return q * L**3 / (3 * EI)  # q 这里是 P
    else:
        raise ValueError(f"未知载荷类型: {load_type}")


def max_deflection_simply_supported(q, L, E, I):
    """简支梁最大挠度（均布载荷，在 x=L/2 处）。"""
    EI = E * I
    return 5 * q * L**4 / (384 * EI)


# ---------------------------------------------------------------------------
# 动态：固有频率与模态形状
# ---------------------------------------------------------------------------

def cantilever_beta_L(n):
    """悬臂梁第 n 阶特征值 β_n L（n=1,2,...）。

    对 n ≤ 8 使用精确已知值，n > 8 使用渐近近似 (n-1/2)π。
    """
    if n <= len(_CANTILEVER_BETA_L):
        return _CANTILEVER_BETA_L[n - 1]
    else:
        return (n - 0.5) * np.pi


def cantilever_sigma(n):
    """悬臂梁第 n 阶模态系数 σ_n。

    σ_n = [cos(β_n L) + cosh(β_n L)] / [sin(β_n L) + sinh(β_n L)]

    推导：由 x=L 处 w''=0 条件给出。
    """
    bL = cantilever_beta_L(n)
    return (np.cos(bL) + np.cosh(bL)) / (np.sin(bL) + np.sinh(bL))


def natural_frequencies(n_modes, bc, E, I, rho, A, L):
    """计算前 n_modes 阶固有频率 ω_n (rad/s)。

    ω_n = (β_n L)² · √(EI / (ρA L⁴))

    bc: 'cantilever' 或 'simply_supported'
    """
    EI = E * I
    mu = rho * A  # 单位长度质量
    omega = np.zeros(n_modes)
    for n in range(1, n_modes + 1):
        if bc == 'cantilever':
            bL = cantilever_beta_L(n)
        elif bc == 'simply_supported':
            bL = n * np.pi
        else:
            raise ValueError(f"未知边界条件: {bc}")
        omega[n - 1] = bL**2 * np.sqrt(EI / (mu * L**4))
    return omega


def mode_shape(x, n, bc, L):
    """计算第 n 阶模态形状 φ_n(x)。

    归一化方式：max|φ| = 1（形状归一化，非质量归一化）。
    """
    x = np.asarray(x, dtype=float)
    if bc == 'cantilever':
        beta = cantilever_beta_L(n) / L
        sigma = cantilever_sigma(n)
        phi = (np.cosh(beta * x) - np.cos(beta * x)
               - sigma * (np.sinh(beta * x) - np.sin(beta * x)))
        # 归一化：max|φ| = 1
        phi_max = np.max(np.abs(phi))
        if phi_max > 0:
            phi = phi / phi_max
    elif bc == 'simply_supported':
        phi = np.sin(n * np.pi * x / L)
        # sin 最大值 = 1，已归一化
    else:
        raise ValueError(f"未知边界条件: {bc}")
    return phi


def mode_shape_second_derivative(x, n, bc, L):
    """计算第 n 阶模态形状的二阶导数 φ_n''(x)。

    用于弯矩和应变能计算。
    """
    x = np.asarray(x, dtype=float)
    if bc == 'cantilever':
        beta = cantilever_beta_L(n) / L
        sigma = cantilever_sigma(n)
        phi_pp = (beta**2 * (np.cosh(beta * x) + np.cos(beta * x)
                             - sigma * (np.sinh(beta * x) + np.sin(beta * x))))
        # 用与 mode_shape 相同的归一化因子
        phi = (np.cosh(beta * x) - np.cos(beta * x)
               - sigma * (np.sinh(beta * x) - np.sin(beta * x)))
        phi_max = np.max(np.abs(phi))
        if phi_max > 0:
            phi_pp = phi_pp / phi_max
    elif bc == 'simply_supported':
        phi_pp = -(n * np.pi / L)**2 * np.sin(n * np.pi * x / L)
    else:
        raise ValueError(f"未知边界条件: {bc}")
    return phi_pp


# ---------------------------------------------------------------------------
# 模态正交性数值积分
# ---------------------------------------------------------------------------

def modal_mass(n, bc, rho, A, L, n_points=500):
    """计算第 n 阶模态质量 m_n = ∫₀ᴸ ρA φ_n² dx。

    使用形状归一化的 φ_n，返回实际模态质量。
    """
    x = np.linspace(0, L, n_points)
    dx = x[1] - x[0]
    phi = mode_shape(x, n, bc, L)
    mu = rho * A
    m_n = mu * np.trapezoid(phi**2, x)
    return m_n


def modal_stiffness(n, bc, E, I, L, n_points=500):
    """计算第 n 阶模态刚度 k_n = ∫₀ᴸ EI (φ_n'')² dx。"""
    x = np.linspace(0, L, n_points)
    phi_pp = mode_shape_second_derivative(x, n, bc, L)
    EI = E * I
    k_n = EI * np.trapezoid(phi_pp**2, x)
    return k_n


def verify_orthogonality(n, m, bc, rho, A, L, n_points=500):
    """验证模态正交性 ∫₀ᴸ ρA φ_n φ_m dx = 0 (n≠m)。"""
    x = np.linspace(0, L, n_points)
    phi_n = mode_shape(x, n, bc, L)
    phi_m = mode_shape(x, m, bc, L)
    mu = rho * A
    result = mu * np.trapezoid(phi_n * phi_m, x)
    return result


# ---------------------------------------------------------------------------
# 动态：模态坐标 ODE
# ---------------------------------------------------------------------------

def modal_dynamics(t, state, omegas, forces_fn=None, n_modes=None):
    """模态坐标的 ODE 系统。

    state = [q1, q2, ..., qN, q̇1, q̇2, ..., q̇N]
    无阻尼：q̈_n + ω_n² q_n = F_n(t)

    omegas: 固有频率数组 [ω1, ω2, ..., ωN]
    forces_fn: function(t) → [F1, F2, ..., FN] 或 None（自由振动）
    """
    N = len(omegas) if n_modes is None else n_modes
    q = state[:N]
    qdot = state[N:]

    qddot = -omegas**2 * q
    if forces_fn is not None:
        F = np.asarray(forces_fn(t), dtype=float)
        qddot += F

    return np.concatenate([qdot, qddot])


def reconstruct_displacement(x, q_modes, bc, L):
    """从模态坐标重构位移 w(x,t) = Σ q_n φ_n(x)。

    q_modes: [q1, q2, ..., qN] 在某一时刻的值
    """
    n_modes = len(q_modes)
    w = np.zeros_like(x, dtype=float)
    for n in range(1, n_modes + 1):
        phi = mode_shape(x, n, bc, L)
        w += q_modes[n - 1] * phi
    return w


# ---------------------------------------------------------------------------
# 能量
# ---------------------------------------------------------------------------

def modal_energy(state, omegas):
    """模态能量 E = ½ Σ(q̇_n² + ω_n² q_n²)。

    注意：此处的 φ_n 必须质量归一化（m_n = 1）时才成立。
    对于形状归一化的 φ_n，需要除以 m_n。

    这里使用 ω_n² / m_n 作为有效模态刚度，
    若 state 中 q_n 已用质量归一化坐标，则 E = ½ Σ(q̇_n² + ω_n² q_n²)。
    """
    N = len(omegas)
    q = state[:N]
    qdot = state[N:]
    T = 0.5 * np.sum(qdot**2)
    U = 0.5 * np.sum(omegas**2 * q**2)
    return T + U


def strain_energy_static(w, M_x, E, I, x):
    """静态应变能 U = ½ ∫₀ᴸ EI (w'')² dx = ½ ∫ M²/EI dx。

    w: 挠度数组
    M_x: 弯矩数组
    """
    # 使用弯矩积分更准确：U = ½ ∫ M²/EI dx
    EI = E * I
    return 0.5 * np.trapezoid(M_x**2 / EI, x)


def kinetic_energy_static(w, rho, A, x):
    """静态时无动能（用于完整性接口）。"""
    return 0.0


# ---------------------------------------------------------------------------
# PDE 右端（用于有限差分法数值求解）
# ---------------------------------------------------------------------------

def beam_pde_rhs(t, w_vec, E, I, rho, A, L, N, q_load_func=None):
    """梁 PDE 的有限差分离散右端。

    将 EI w'''' + ρA ẅ = q(x,t) 转化为 2N 个一阶 ODE。
    状态向量: [w_1, ..., w_N, ẇ_1, ..., ẇ_N]

    使用中心差分近似 4 阶导数：
    w''''_i ≈ (w_{i-2} - 4w_{i-1} + 6w_i - 4w_{i+1} + w_{i+2}) / dx⁴

    边界条件通过 ghost nodes 实现。
    """
    dx = L / (N - 1)
    mu = rho * A
    EI = E * I

    w = w_vec[:N]
    wdot = w_vec[N:]

    # 4 阶导数（中心差分，内部节点）
    wpppp = np.zeros(N)
    for i in range(2, N - 2):
        wpppp[i] = (w[i - 2] - 4 * w[i - 1] + 6 * w[i]
                    - 4 * w[i + 1] + w[i + 2]) / dx**4

    # 简支梁边界条件（w=0, w''=0 at both ends）
    # w_0 = 0, w_{N-1} = 0
    # w''_0 = 0 → ghost: w_{-1} = -w_1, w''_{N-1}=0 → w_N = -w_{N-2}
    # 处理边界附近节点
    # i=1: w_{-1} = -w_1 (简支 w''=0 at x=0)
    wpppp[1] = (-w[1] - 4 * w[0] + 6 * w[1] - 4 * w[2] + w[3]) / dx**4
    wpppp[0] = 0.0  # w=0
    # i=N-2
    wpppp[N - 2] = (w[N - 4] - 4 * w[N - 3] + 6 * w[N - 2]
                    - 4 * w[N - 1] + (-w[N - 2])) / dx**4
    wpppp[N - 1] = 0.0  # w=0

    # 外力
    if q_load_func is not None:
        x_nodes = np.linspace(0, L, N)
        q = np.asarray(q_load_func(x_nodes), dtype=float)
    else:
        q = np.zeros(N)

    wddot = (-EI * wpppp + q) / mu

    # 固定位移边界
    wddot[0] = 0.0
    wddot[-1] = 0.0

    return np.concatenate([wdot, wddot])


# ---------------------------------------------------------------------------
# 有限差分特征值法（用于数值验证固有频率）
# ---------------------------------------------------------------------------

def fd_stiffness_matrix(N, L, E, I):
    """构造简支梁有限差分刚度矩阵 K。

    K 是 (N-2)×(N-2) 矩阵（内部自由度），边界 w=0 已消除。
    简支 BC: w=0 且 w''=0 at x=0,L。

    K[i,j] = EI * (中心差分 4 阶导数系数) / dx⁴
    """
    dx = L / (N - 1)
    n = N - 2  # 内部节点数
    K = np.zeros((n, n))
    EI = E * I

    for i in range(n):
        # 主对角线
        K[i, i] = 6.0 * EI / dx**4
        if i > 0:
            K[i, i - 1] = -4.0 * EI / dx**4
        if i < n - 1:
            K[i, i + 1] = -4.0 * EI / dx**4
        if i > 1:
            K[i, i - 2] = 1.0 * EI / dx**4
        if i < n - 2:
            K[i, i + 2] = 1.0 * EI / dx**4

    # 简支 w''=0 BC at boundaries:
    #   ghost w_{-1} = -w_1  →  系数从 6 变为 5（减 1）
    #   ghost w_N  = -w_{N-2} →  系数从 6 变为 5（减 1）
    K[0, 0] -= 1.0 * EI / dx**4
    K[n - 1, n - 1] -= 1.0 * EI / dx**4

    return K


def fd_mass_matrix(N, L, rho, A):
    """构造简支梁质量矩阵 M（逐点 FD 离散）。

    FD 方程是逐点的：EI w''''_i + ρA ẅ_i = 0
    无积分步骤，故 M = ρA · I（不含 dx 因子）。
    （FEM/Galerkin 弱形式才需要积分，此时 M 含 dx）
    """
    mu = rho * A
    n = N - 2
    M = np.diag(np.full(n, mu))
    return M


def fd_natural_frequencies(N, L, E, I, rho, A, n_modes=None):
    """用有限差分法计算简支梁的固有频率。

    求解广义特征值问题 K φ = ω² M φ。
    """
    K = fd_stiffness_matrix(N, L, E, I)
    M = fd_mass_matrix(N, L, rho, A)

    from scipy.linalg import eigh
    eigvals, _ = eigh(K, M)
    omegas = np.sqrt(np.maximum(eigvals, 0))

    # 排序（升序）并返回前 n_modes
    omegas = np.sort(omegas)
    if n_modes is not None:
        return omegas[:n_modes]
    return omegas
