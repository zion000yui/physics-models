"""MEC-052-shell — 模型定义（引擎无关）

圆柱壳（cylindrical shell）力学：Kirchhoff-Love 壳理论。
在 MEC-051 薄板基础上引入曲率效应，实现薄膜-弯曲耦合。

=== 物理系统 ===

  圆柱壳：半径 R，壁厚 h，长度 L
  坐标：x（轴向），θ（环向），x ∈ [0, L]，θ ∈ [0, 2π)
  位移分量：u（轴向），v（环向），w（法向，向外为正）

  Kirchhoff-Love 假设：
  - 壳体薄（h << R, h << L）
  - 中面法线变形后仍垂直于中面（忽略剪切变形）
  - 适用于 R/h > 20 的薄壳

=== 薄膜理论（无矩理论）===

  不考虑弯矩和剪力，仅由薄膜力（面内力）平衡外载荷。

  薄膜力：N_x, N_θ, N_xθ（单位长度上的面内力）

  平衡方程（圆柱壳，无体积力）：
    ∂N_x/∂x + (1/R) ∂N_xθ/∂θ = 0
    (1/R) ∂N_θ/∂θ + ∂N_xθ/∂x = 0
    N_θ/R = p  （法向平衡，p 为法向外压）

  内压 p 下的薄膜力：
    N_θ = pR  （环向薄膜力，"环向应力 × 壁厚"）
    N_x = pR/2  （轴向薄膜力，封闭端压力平衡）
    N_xθ = 0

  薄膜应力：
    σ_θ = N_θ/h = pR/h
    σ_x = N_x/h = pR/(2h)

  经典薄壁压力容器公式：
    σ_θ = pR/h （环向应力）
    σ_x = pR/(2h) （轴向应力）
    σ_θ / σ_x = 2（环向应力是轴向应力的 2 倍）

=== 弯曲理论（有矩理论）===

  考虑弯矩 M_x, M_θ 和横向剪力 Q_x。

  Donnell 简化方程（圆柱壳弯曲）：
    D ∂⁴w/∂x⁴ + (1/R²) ∂⁴w/∂θ⁴ + ... = p(x,θ)

  简化版（轴对称载荷，仅 x 依赖）：
    D d⁴w/dx⁴ + (Eh/R²) w = p(x)

    D = Eh³/[12(1-ν²)]
    Eh/R² 为"弹性基础"项（曲率效应）

  这类比 Winkler 弹性基础梁方程：
    D w'''' + k w = p,  k = Eh/R²

  特征长度（衰减长度）：
    λ = (D/k)^(1/4) = (Eh³/[12(1-ν²)] / (Eh/R²))^(1/4)
      = (h²R²/[12(1-ν²)])^(1/4)
      = √(hR) / [12(1-ν²)]^(1/4)

  轴对称弯曲解析解（齐次方程通解）：
    w(x) = e^(-αx)[C1 cos(αx) + C2 sin(αx)]
          + e^(αx)[C3 cos(αx) + C4 sin(αx)]
    其中 α = 1/λ = [12(1-ν²)]^(1/4) / √(hR)

  对长壳（L >> λ），端部效应衰减，只需保留衰减项。

=== 动态分析 ===

  圆柱壳固有频率（Donnell 简化，轴对称模态）：
    ω_n² = [D(x_n/R)⁴ + Eh/R²] / (ρh)
    其中 x_n = nπ/L

  简化：
    ω_n² = (1/(ρh)) [D(nπ/L)⁴ + Eh/R²]

  当 n→∞ 时 ω_n → ∞（弯曲波）
  当 n=0 时 ω_0 = √(E/(ρR²))（薄膜振动频率）

=== 与已有 MEC 模型的关系 ===

  MEC-051 板 → MEC-052 壳：引入曲率 R，增加"弹性基础"项 Eh/R²
  MEC-050 梁 → MEC-052 壳（轴对称弯曲）：Donnell 方程 = 弹性基础梁
  R → ∞ 时退化为 MEC-051 板（曲率项消失）

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(E=2.0e11, h=0.005, nu=0.3, rho=7850.0,
                        R=0.5, L=2.0, p=0.0):
    """验证物理参数合法性。"""
    assert E > 0, f"杨氏模量 E 必须为正，当前 E={E}"
    assert h > 0, f"壁厚 h 必须为正，当前 h={h}"
    assert 0 <= nu < 0.5, f"泊松比 ν 应在 [0, 0.5)，当前 ν={nu}"
    assert rho > 0, f"密度 rho 必须为正，当前 rho={rho}"
    assert R > 0, f"半径 R 必须为正，当前 R={R}"
    assert L > 0, f"长度 L 必须为正，当前 L={L}"


def bending_stiffness(E, h, nu):
    """壳抗弯刚度 D = E h³ / [12(1-ν²)]（同 MEC-051）。"""
    return E * h**3 / (12.0 * (1.0 - nu**2))


def membrane_stiffness(E, h, R):
    """薄膜"弹性基础"刚度 k = Eh/R²。"""
    return E * h / R**2


def characteristic_length(E, h, nu, R):
    """衰减特征长度 λ = (D/k)^(1/4)。

    λ = [h²R² / (12(1-ν²))]^(1/4) = √(hR) / [12(1-ν²)]^(1/4)
    """
    D = bending_stiffness(E, h, nu)
    k = membrane_stiffness(E, h, R)
    return (D / k)**0.25


def decay_constant(E, h, nu, R):
    """衰减常数 α = 1/λ = [12(1-ν²)]^(1/4) / √(hR)。"""
    return 1.0 / characteristic_length(E, h, nu, R)


# ---------------------------------------------------------------------------
# 薄膜理论
# ---------------------------------------------------------------------------

def membrane_forces_internal_pressure(p, R):
    """内压 p 下的薄膜力。

    N_θ = pR （环向薄膜力）
    N_x = pR/2 （轴向薄膜力，封闭端压力平衡）
    """
    N_theta = p * R
    N_x = p * R / 2.0
    return N_x, N_theta


def membrane_stresses_internal_pressure(p, R, h):
    """内压 p 下的薄膜应力。

    σ_θ = pR/h （环向应力）
    σ_x = pR/(2h) （轴向应力）
    """
    sigma_theta = p * R / h
    sigma_x = p * R / (2.0 * h)
    return sigma_x, sigma_theta


def hoop_to_axial_ratio():
    """环向应力与轴向应力之比（恒为 2）。"""
    return 2.0


# ---------------------------------------------------------------------------
# 弯曲理论（轴对称，Donnell 简化）
# ---------------------------------------------------------------------------

def axial_bending_ode_coefficients(E, h, nu, R):
    """轴对称弯曲 ODE 系数。

    D w'''' + k w = p
    D = Eh³/[12(1-ν²)], k = Eh/R²
    """
    D = bending_stiffness(E, h, nu)
    k = membrane_stiffness(E, h, R)
    return D, k


def axial_bending_analytical(x, p, E, h, nu, R, L, bc='long_cantilever'):
    """轴对称弯曲解析解。

    D w'''' + k w = p

    特解：w_p = p/k = pR²/(Eh)
    齐次解：w_h = e^{-αx}[C1 cos(αx) + C2 sin(αx)] + e^{αx}[C3 cos(αx) + C4 sin(αx)]

    对长壳（L >> λ），忽略 e^{αx} 项（指数增长），
    只保留 e^{-αx} 衰减项。

    bc='long_cantilever': x=0 处固定（w=0, w'=0），x→∞ 自由
    bc='long_simply_supported': x=0 处铰支（w=0, M=0），x→∞ 自由
    """
    x = np.asarray(x, dtype=float)
    D, k = axial_bending_ode_coefficients(E, h, nu, R)
    alpha = decay_constant(E, h, nu, R)

    # 特解
    w_p = p / k  # = pR²/(Eh)

    if bc == 'long_cantilever':
        # x=0: w=0 → C1 = -w_p
        # x=0: w'=0 → -α C1 + α C2 = 0 → C2 = C1
        C1 = -w_p
        C2 = C1  # = -w_p
        C3 = 0.0
        C4 = 0.0
    elif bc == 'long_simply_supported':
        # x=0: w=0 → C1 = -w_p
        # x=0: w''=0 → -2 α² C2 = 0 → C2 = 0
        C1 = -w_p
        C2 = 0.0
        C3 = 0.0
        C4 = 0.0
    else:
        raise ValueError(f"未知边界条件: {bc}")

    ax = alpha * x
    w = (w_p
         + np.exp(-ax) * (C1 * np.cos(ax) + C2 * np.sin(ax))
         + np.exp(ax) * (C3 * np.cos(ax) + C4 * np.sin(ax)))
    return w


def axial_bending_max_deflection(p, E, h, R):
    """轴对称弯曲最大挠度（远端，特解值）。

    w_max = p/k = pR²/(Eh)
    """
    k = membrane_stiffness(E, h, R)
    return p / k


# ---------------------------------------------------------------------------
# 动态：圆柱壳固有频率
# ---------------------------------------------------------------------------

def natural_frequencies_cylindrical(n_modes, E, h, nu, rho, R, L):
    """圆柱壳轴对称固有频率（Donnell 简化）。

    ω_n² = [D(nπ/L)⁴ + Eh/R²] / (ρh)

    返回前 n_modes 阶频率（n=1,2,...）。
    """
    D = bending_stiffness(E, h, nu)
    k = membrane_stiffness(E, h, R)
    mu = rho * h  # 单位面积质量

    omegas = []
    for n in range(1, n_modes + 1):
        kappa_n = n * np.pi / L  # 轴向波数
        omega_sq = (D * kappa_n**4 + k) / mu
        omegas.append(np.sqrt(omega_sq))
    return np.array(omegas)


def membrane_frequency(E, rho, R):
    """薄膜振动基频（n→0 极限）。

    ω_0 = √(E/(ρR²)) = (1/R)√(E/ρ) = c_L/R
    其中 c_L = √(E/ρ) 为纵波速度。
    """
    return np.sqrt(E / (rho * R**2))


def bending_frequency_limit(n, E, h, nu, rho, L):
    """高阶弯曲频率极限（n 大时，薄膜项可忽略）。

    ω_n ≈ (nπ/L)² √(D/(ρh))
    """
    D = bending_stiffness(E, h, nu)
    mu = rho * h
    return (n * np.pi / L)**2 * np.sqrt(D / mu)


# ---------------------------------------------------------------------------
# 模态动力学
# ---------------------------------------------------------------------------

def modal_dynamics_shell(t, state, omegas, forces_fn=None):
    """模态坐标 ODE 系统。"""
    N = len(omegas)
    q = state[:N]
    qdot = state[N:]
    qddot = -omegas**2 * q
    if forces_fn is not None:
        F = np.asarray(forces_fn(t), dtype=float)
        qddot += F
    return np.concatenate([qdot, qddot])


def modal_energy_shell(state, omegas):
    """模态能量。"""
    N = len(omegas)
    q = state[:N]
    qdot = state[N:]
    T = 0.5 * np.sum(qdot**2)
    U = 0.5 * np.sum(omegas**2 * q**2)
    return T + U


# ---------------------------------------------------------------------------
# 退化验证
# ---------------------------------------------------------------------------

def degradation_to_plate_check(E, h, nu, rho, L, R_large=1e6):
    """R → ∞ 时壳退化为板的验证参数。

    当 R 很大时：
    - k = Eh/R² → 0（弹性基础项消失）
    - ω_n² → D(nπ/L)⁴ / (ρh)（退化为 MEC-051 板频率）
    """
    D = bending_stiffness(E, h, nu)
    k = membrane_stiffness(E, h, R_large)
    mu = rho * h
    n = 1
    omega_shell = np.sqrt((D * (n * np.pi / L)**4 + k) / mu)
    omega_plate = (n * np.pi / L)**2 * np.sqrt(D / mu)
    return omega_shell, omega_plate, abs(omega_shell - omega_plate) / omega_plate


# ---------------------------------------------------------------------------
# 有限差分（用于数值验证轴对称弯曲方程）
# ---------------------------------------------------------------------------

def fd_shell_stiffness_matrix(N, L, E, h, nu, R):
    """轴对称壳弯曲方程的 FD 离散刚度矩阵。

    D w'''' + k w = 0 的离散：
    K = D * D4 + k * I

    简支边界（w=0, w''=0 at x=0,L）。
    """
    dx = L / (N - 1)
    n = N - 2  # 内部节点

    # 四阶导数矩阵（同 MEC-050 简支梁）
    D4 = np.zeros((n, n))
    for i in range(n):
        D4[i, i] = 6.0
        if i > 0:
            D4[i, i - 1] = -4.0
        if i < n - 1:
            D4[i, i + 1] = -4.0
        if i > 1:
            D4[i, i - 2] = 1.0
        if i < n - 2:
            D4[i, i + 2] = 1.0
    D4[0, 0] -= 1.0
    D4[-1, -1] -= 1.0
    D4 /= dx**4

    D = bending_stiffness(E, h, nu)
    k = membrane_stiffness(E, h, R)

    K = D * D4 + k * np.eye(n)
    return K


def fd_shell_mass_matrix(N, L, rho, h):
    """壳质量矩阵（逐点 FD: M = ρh·I）。"""
    mu = rho * h
    n = N - 2
    return mu * np.eye(n)


def fd_shell_natural_frequencies(N, L, E, h, nu, rho, R, n_modes=5):
    """用有限差分法计算壳固有频率。"""
    from scipy.linalg import eigh
    K = fd_shell_stiffness_matrix(N, L, E, h, nu, R)
    M = fd_shell_mass_matrix(N, L, rho, h)
    eigvals, _ = eigh(K, M)
    omegas = np.sqrt(np.maximum(eigvals, 0))
    omegas = np.sort(omegas)
    return omegas[:n_modes]
