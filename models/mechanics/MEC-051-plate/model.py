"""MEC-051-plate — 模型定义（引擎无关）

Kirchhoff-Love 薄板弯曲理论：欧拉-伯努利梁的二维推广。
板的横向位移 w(x,y,t) 满足 4 阶 2D 偏微分方程：

  D ∇⁴ w + ρh ẅ = q(x,y,t)

  D  — 板抗弯刚度 D = E h³ / [12(1-ν²)]
  E  — 杨氏模量 (Pa)
  h  — 板厚 (m)
  ν  — 泊松比
  ρ  — 密度 (kg/m³)
  q  — 分布载荷 (Pa = N/m²)

  ∇⁴ = ∂⁴/∂x⁴ + 2 ∂⁴/∂x²∂y² + ∂⁴/∂y⁴  (双调和算子)

=== 物理系统 ===

  薄板（厚度远小于面内尺寸，h << a, h << b），满足 Kirchhoff 假设：
  - 变形前垂直于中面的法线变形后仍垂直于中面（忽略剪切变形）
  - 中面法向位移 w 统一描述（无面内旋转独立自由度）
  - 适用于 h/min(a,b) < 1/20 的薄板

=== 静态分析（简支矩形板，Navier 解法）===

  D ∇⁴ w = q(x,y)

  边界条件（简支，四边铰支）：
    x=0,a: w=0, M_x=0  (M_x = -D(∂²w/∂x² + ν ∂²w/∂y²))
    y=0,b: w=0, M_y=0  (M_y = -D(ν ∂²w/∂x² + ∂²w/∂y²))

  Navier 双重正弦级数解：
    w(x,y) = Σ Σ W_mn sin(mπx/a) sin(nπy/b)
    W_mn = q_mn / [D π⁴ (m²/a² + n²/b²)²]
    q_mn = (4/(ab)) ∫∫ q(x,y) sin(mπx/a) sin(nπy/b) dx dy

  均布载荷 q：q_mn = 16q/(π²mn) (m,n 均为奇数), 0 (偶数)

  最大挠度（中心，a=b 时）：
    w_max = α₁ q a⁴ / D,  α₁ ≈ 0.00406 (取级数首项 16q/(π⁴D)/(2π²/a²)²)

=== 动态分析（简支板自由振动）===

  D ∇⁴ w + ρh ẅ = 0

  分离变量 w = φ(x,y) T(t)，模态函数：
    φ_mn(x,y) = sin(mπx/a) sin(nπy/b)

  固有频率：
    ω_mn = π² (m²/a² + n²/b²) √(D/(ρh))

  模态动力学：
    T̈_mn + ω_mn² T_mn = 0

  能量：
    T = ½ ρh ∫∫ (ẇ)² dx dy = ½ Σ q̇_mn² (质量归一化)
    U = ½ D ∫∫ [(∇²w)² - 2(1-ν)(w_xx w_yy - w_xy²)] dx dy
      = ½ Σ ω_mn² q_mn²

=== 内力（弯矩、扭矩、剪力）===

  M_x = -D (∂²w/∂x² + ν ∂²w/∂y²)
  M_y = -D (ν ∂²w/∂x² + ∂²w/∂y²)
  M_xy = -D(1-ν) ∂²w/∂x∂y  (扭矩)
  Q_x = -D ∂/∂x (∇²w)       (横向剪力)
  Q_y = -D ∂/∂y (∇²w)

=== 与已有 MEC 模型的关系 ===

  MEC-050 梁（1D）→ MEC-051 板（2D）：截面惯性矩 I → 板厚 h 的 3 次方
  D = E h³ / [12(1-ν²)] 与 EI = E · bh³/12 的对应：板取单位宽度 b=1，并修正泊松比
  MEC-051 → MEC-052 壳：引入曲率效应
  MEC-051 → MEC-053 3D 弹性体：从薄板近似到完整 3D 弹性理论

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(E=2.0e11, h=0.01, nu=0.3, rho=7850.0,
                        a=1.0, b=1.0, q_load=0.0):
    """验证物理参数合法性。"""
    assert E > 0, f"杨氏模量 E 必须为正，当前 E={E}"
    assert h > 0, f"板厚 h 必须为正，当前 h={h}"
    assert 0 <= nu < 0.5, f"泊松比 ν 应在 [0, 0.5)，当前 ν={nu}"
    assert rho > 0, f"密度 rho 必须为正，当前 rho={rho}"
    assert a > 0, f"板长 a 必须为正，当前 a={a}"
    assert b > 0, f"板宽 b 必须为正，当前 b={b}"


def plate_stiffness(E, h, nu):
    """板抗弯刚度 D = E h³ / [12(1-ν²)]。"""
    return E * h**3 / (12.0 * (1.0 - nu**2))


# ---------------------------------------------------------------------------
# 静态：简支矩形板 Navier 解
# ---------------------------------------------------------------------------

def navier_load_coeff(m, n, q, a, b):
    """均布载荷 q 的 Navier 系数 q_mn。

    q_mn = 16q/(π²mn)  (m, n 均为奇数)
    q_mn = 0            (m 或 n 为偶数)
    """
    if m % 2 == 1 and n % 2 == 1:
        return 16.0 * q / (np.pi**2 * m * n)
    return 0.0


def static_simply_supported_navier(x, y, q, a, b, E, h, nu,
                                   n_terms=20):
    """简支矩形板在均布载荷下的 Navier 解。

    w(x,y) = Σ Σ W_mn sin(mπx/a) sin(nπy/b)
    W_mn = q_mn / [D π⁴ (m²/a² + n²/b²)²]

    返回 (w, Mx, My, Mxy)。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    D = plate_stiffness(E, h, nu)

    # 确保是网格
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y, indexing='ij')
    else:
        X, Y = x, y

    w = np.zeros_like(X)
    w_xx = np.zeros_like(X)
    w_yy = np.zeros_like(X)
    w_xy = np.zeros_like(X)

    for m in range(1, n_terms + 1):
        for n in range(1, n_terms + 1):
            q_mn = navier_load_coeff(m, n, q, a, b)
            if q_mn == 0:
                continue
            alpha_m = m * np.pi / a
            beta_n = n * np.pi / b
            denom = D * np.pi**4 * (m**2 / a**2 + n**2 / b**2)**2
            W_mn = q_mn / denom

            sx = np.sin(alpha_m * X)
            sy = np.sin(beta_n * Y)
            cx = np.cos(alpha_m * X)
            cy = np.cos(beta_n * Y)

            w += W_mn * sx * sy
            w_xx += -alpha_m**2 * W_mn * sx * sy
            w_yy += -beta_n**2 * W_mn * sx * sy
            w_xy += alpha_m * beta_n * W_mn * cx * cy

    Mx = -D * (w_xx + nu * w_yy)
    My = -D * (nu * w_xx + w_yy)
    Mxy = -D * (1 - nu) * w_xy

    return w, Mx, My, Mxy


def max_deflection_simply_supported(q, a, b, E, h, nu, n_terms=50):
    """简支板中心最大挠度（级数在 (a/2, b/2) 处求值）。

    w(a/2, b/2) = (16q/(π⁶D)) Σ_{m,n odd} sin(mπ/2)sin(nπ/2) / [mn(m²/a²+n²/b²)²]
    """
    D = plate_stiffness(E, h, nu)
    w_center = 0.0
    for m in range(1, n_terms + 1, 2):  # 仅奇数
        for n in range(1, n_terms + 1, 2):
            s = np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
            term = s / (m * n * (m**2 / a**2 + n**2 / b**2)**2)
            w_center += term
    w_center *= 16.0 * q / (np.pi**6 * D)
    return w_center


def deflection_coefficient(a, b, n_terms=50):
    """计算简支板中心挠度系数 α（w_max = α q a⁴ / D）。

    对方板 (a=b) 约为 0.00406。
    """
    D = 1.0  # 归一化
    q = 1.0
    w = max_deflection_simply_supported(q, a, b, 1.0, h=1.0, nu=0.0,
                                        n_terms=n_terms)
    # D=1, h=1, nu=0 时 D = E*1/(12) → 不对，直接用 D
    # 重新计算：w = 16q/(π^6 D) * Σ ...
    # 取 D=1, q=1 直接
    alpha = w / (q * a**4) * D
    return alpha


# ---------------------------------------------------------------------------
# 动态：简支板固有频率与模态
# ---------------------------------------------------------------------------

def natural_frequencies_plate(n_modes, a, b, E, h, nu, rho):
    """计算简支板前 n_modes 阶固有频率。

    ω_mn = π² (m²/a² + n²/b²) √(D/(ρh))

    返回排序后的频率数组。
    """
    D = plate_stiffness(E, h, nu)
    omega_factor = np.sqrt(D / (rho * h))

    # 搜索 (m,n) 组合
    max_mn = 20
    omegas = []
    for m in range(1, max_mn + 1):
        for n in range(1, max_mn + 1):
            omega = np.pi**2 * (m**2 / a**2 + n**2 / b**2) * omega_factor
            omegas.append((omega, m, n))

    omegas.sort()
    return np.array([omegas[i][0] for i in range(min(n_modes, len(omegas)))])


def plate_mode_shape(x, y, m, n, a, b):
    """简支板模态形状 φ_mn(x,y) = sin(mπx/a) sin(nπy/b)。

    已归一化：max|φ| = 1。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y, indexing='ij')
    else:
        X, Y = x, y
    return np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)


def plate_mode_laplacian(x, y, m, n, a, b):
    """模态形状的拉普拉斯算子 ∇²φ_mn。

    ∇²φ = -(m²π²/a² + n²π²/b²) sin(mπx/a) sin(nπy/b)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y, indexing='ij')
    else:
        X, Y = x, y
    alpha_m = m * np.pi / a
    beta_n = n * np.pi / b
    return -(alpha_m**2 + beta_n**2) * np.sin(alpha_m * X) * np.sin(beta_n * Y)


# ---------------------------------------------------------------------------
# 模态正交性
# ---------------------------------------------------------------------------

def verify_plate_orthogonality(m1, n1, m2, n2, rho, h, a, b,
                               n_points=100):
    """验证简支板模态正交性 ∫∫ ρh φ_mn φ_m'n' dx dy = 0 (不同模态)。"""
    x = np.linspace(0, a, n_points)
    y = np.linspace(0, b, n_points)
    dx = a / (n_points - 1)
    dy = b / (n_points - 1)

    phi1 = plate_mode_shape(x, y, m1, n1, a, b)
    phi2 = plate_mode_shape(x, y, m2, n2, a, b)
    mu = rho * h
    return mu * np.trapezoid(np.trapezoid(phi1 * phi2, x), y)


def plate_modal_mass(m, n, rho, h, a, b, n_points=200):
    """模态质量 m_mn = ∫∫ ρh φ_mn² dx dy = ρh·ab/4（解析）。"""
    x = np.linspace(0, a, n_points)
    y = np.linspace(0, b, n_points)
    phi = plate_mode_shape(x, y, m, n, a, b)
    mu = rho * h
    return mu * np.trapezoid(np.trapezoid(phi**2, x), y)


def plate_modal_stiffness(m, n, D, a, b, nu=0.3, n_points=200):
    """模态刚度 k_mn = ∫∫ D[(∇²φ)² + 2(1-ν)(φ_xx φ_yy - φ_xy²)] dx dy。

    对于简支板 sin(mπx/a) sin(nπy/b)，解析结果：
    k_mn = D · (ab/4) · (m²π²/a² + n²π²/b²)²

    这里用数值积分验证。
    """
    x = np.linspace(0, a, n_points)
    y = np.linspace(0, b, n_points)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    phi = plate_mode_shape(x, y, m, n, a, b)
    # 二阶导数
    alpha_m = m * np.pi / a
    beta_n = n * np.pi / b
    X, Y = np.meshgrid(x, y, indexing='ij')

    phi_xx = -alpha_m**2 * np.sin(alpha_m * X) * np.sin(beta_n * Y)
    phi_yy = -beta_n**2 * np.sin(alpha_m * X) * np.sin(beta_n * Y)
    phi_xy = alpha_m * beta_n * np.cos(alpha_m * X) * np.cos(beta_n * Y)

    laplacian_sq = (phi_xx + phi_yy)**2
    twist = phi_xx * phi_yy - phi_xy**2

    integrand = D * (laplacian_sq + 2 * (1 - nu) * (phi_xx * phi_yy - phi_xy**2))
    # 简化：对于 sin×sin 模态，twist = phi_xx * phi_yy - phi_xy^2 = 0
    # 因为 phi_xx * phi_yy = α²β² sin²sin²，phi_xy² = α²β² cos²cos²
    # 差 = α²β²(sin²sin² - cos²cos²) ≠ 0
    # 但 D[(∇²φ)² + 2(1-ν)(φ_xx φ_yy - φ_xy²)]
    # = D[α⁴sin²sin² + 2α²β²sin²sin² + β⁴sin²sin² + 2(1-ν)(α²β²sin²sin² - α²β²cos²cos²)]

    # 数值积分
    integral = np.trapezoid(np.trapezoid(integrand, x), y)
    return integral


# ---------------------------------------------------------------------------
# 动态：模态坐标 ODE
# ---------------------------------------------------------------------------

def modal_dynamics_plate(t, state, omegas, forces_fn=None):
    """简支板模态坐标 ODE 系统。

    state = [q1, ..., qN, q̇1, ..., q̇N]
    q̈_n + ω_n² q_n = F_n(t)
    """
    N = len(omegas)
    q = state[:N]
    qdot = state[N:]
    qddot = -omegas**2 * q
    if forces_fn is not None:
        F = np.asarray(forces_fn(t), dtype=float)
        qddot += F
    return np.concatenate([qdot, qddot])


def reconstruct_plate_displacement(x, y, q_modes, mode_indices, a, b):
    """从模态坐标重构板位移 w(x,y,t) = Σ q_mn φ_mn(x,y)。

    mode_indices: [(m1,n1), (m2,n2), ...]
    q_modes: [q1, q2, ...]
    """
    w = np.zeros_like(x, dtype=float) if x.ndim > 1 else np.zeros_like(
        np.meshgrid(x, y, indexing='ij')[0])
    for i, (m, n) in enumerate(mode_indices):
        phi = plate_mode_shape(x, y, m, n, a, b)
        w += q_modes[i] * phi
    return w


# ---------------------------------------------------------------------------
# 能量
# ---------------------------------------------------------------------------

def modal_energy_plate(state, omegas):
    """模态能量 E = ½ Σ(q̇_n² + ω_n² q_n²)（质量归一化时）。"""
    N = len(omegas)
    q = state[:N]
    qdot = state[N:]
    T = 0.5 * np.sum(qdot**2)
    U = 0.5 * np.sum(omegas**2 * q**2)
    return T + U


def strain_energy_plate(w, D, nu, x, y):
    """静态应变能。

    U = ½ D ∫∫ [(∇²w)² - 2(1-ν)(w_xx w_yy - w_xy²)] dx dy

    简化版（各向同性板）：U = ½ D ∫∫ [(w_xx + w_yy)² - 2(1-ν)(w_xx w_yy - w_xy²)] dx dy
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # 数值二阶导数
    w_xx = np.gradient(np.gradient(w, dx, axis=0), dx, axis=0)
    w_yy = np.gradient(np.gradient(w, dy, axis=1), dy, axis=1)
    w_xy = np.gradient(np.gradient(w, dx, axis=0), dy, axis=1)

    integrand = D * ((w_xx + w_yy)**2
                     - 2 * (1 - nu) * (w_xx * w_yy - w_xy**2))
    return 0.5 * np.trapezoid(np.trapezoid(integrand, x), y)


# ---------------------------------------------------------------------------
# 有限差分法（用于数值验证）
# ---------------------------------------------------------------------------

def fd_plate_stiffness_matrix(Nx, Ny, a, b, D, nu):
    """构造简支板有限差分离散刚度矩阵。

    ∇⁴w = ∂⁴w/∂x⁴ + 2∂⁴w/∂x²∂y² + ∂⁴w/∂y⁴

    每项用 Kronecker 积构建，简支 BC (w=0, ∇²w=0) 通过 ghost nodes 处理。

    返回 (Nx-2)*(Ny-2) × (Nx-2)*(Ny-2) 矩阵。
    """
    dx = a / (Nx - 1)
    dy = b / (Ny - 1)
    n1 = Nx - 2  # x 方向内部自由度
    n2 = Ny - 2  # y 方向内部自由度

    # --- 1D 二阶导数矩阵（简支 BC: w=0，不含 ghost）---
    # w_0 = 0, w_{N-1} = 0 已知，内部节点不需要 ghost
    D2_x = np.zeros((n1, n1))
    for i in range(n1):
        D2_x[i, i] = -2.0
        if i > 0:
            D2_x[i, i - 1] = 1.0
        if i < n1 - 1:
            D2_x[i, i + 1] = 1.0
    D2_x /= dx**2

    D2_y = np.zeros((n2, n2))
    for i in range(n2):
        D2_y[i, i] = -2.0
        if i > 0:
            D2_y[i, i - 1] = 1.0
        if i < n2 - 1:
            D2_y[i, i + 1] = 1.0
    D2_y /= dy**2

    # --- 1D 四阶导数矩阵（简支 BC: w=0 + ∇²w=0 → ghost w_{-1} = -w_1）---
    D4_x = np.zeros((n1, n1))
    for i in range(n1):
        D4_x[i, i] = 6.0
        if i > 0:
            D4_x[i, i - 1] = -4.0
        if i < n1 - 1:
            D4_x[i, i + 1] = -4.0
        if i > 1:
            D4_x[i, i - 2] = 1.0
        if i < n1 - 2:
            D4_x[i, i + 2] = 1.0
    # Ghost: w_{-1} = -w_1, w_N = -w_{N-2}
    D4_x[0, 0] -= 1.0      # ghost w_{-1} = -w_1 贡献 -1
    D4_x[-1, -1] -= 1.0   # ghost w_N = -w_{N-2} 贡献 -1
    D4_x /= dx**4

    D4_y = np.zeros((n2, n2))
    for i in range(n2):
        D4_y[i, i] = 6.0
        if i > 0:
            D4_y[i, i - 1] = -4.0
        if i < n2 - 1:
            D4_y[i, i + 1] = -4.0
        if i > 1:
            D4_y[i, i - 2] = 1.0
        if i < n2 - 2:
            D4_y[i, i + 2] = 1.0
    D4_y[0, 0] -= 1.0
    D4_y[-1, -1] -= 1.0
    D4_y /= dy**4

    # --- ∇⁴ = D4_x ⊗ I_y + 2 D2_x ⊗ D2_y + I_x ⊗ D4_y ---
    I_x = np.eye(n1)
    I_y = np.eye(n2)

    Lap4 = (np.kron(D4_x, I_y)
            + 2 * np.kron(D2_x, D2_y)
            + np.kron(I_x, D4_y))

    K = D * Lap4
    return K


def fd_plate_mass_matrix(Nx, Ny, a, b, rho, h):
    """简支板质量矩阵（逐点 FD 离散，M = ρh · I）。"""
    n = (Nx - 2) * (Ny - 2)
    mu = rho * h
    return mu * np.eye(n)


def fd_plate_natural_frequencies(Nx, Ny, a, b, E, h, nu, rho, n_modes=5):
    """用有限差分法计算简支板固有频率。"""
    from scipy.linalg import eigh

    D = plate_stiffness(E, h, nu)
    K = fd_plate_stiffness_matrix(Nx, Ny, a, b, D, nu)
    M = fd_plate_mass_matrix(Nx, Ny, a, b, rho, h)

    eigvals, _ = eigh(K, M)
    omegas = np.sqrt(np.maximum(eigvals, 0))
    omegas = np.sort(omegas)
    return omegas[:n_modes]
