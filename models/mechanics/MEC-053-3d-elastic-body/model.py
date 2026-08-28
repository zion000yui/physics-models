"""MEC-053-3d-elastic-body — 模型定义（引擎无关）

三维线弹性体（3D linear elastic body）：广义胡克定律。
从一维梁（MEC-050）→ 二维板（MEC-051）→ 三维弹性体的完整递进。

=== 物理系统 ===

  各向同性线弹性体，在小变形范围内描述。
  状态变量：位移 u = (u_x, u_y, u_z)，应变 ε (6 分量)，应力 σ (6 分量)

=== 应变-位移关系（小变形几何方程）===

  ε_xx = ∂u_x/∂x,  ε_yy = ∂u_y/∂y,  ε_zz = ∂u_z/∂z
  ε_xy = ½(∂u_x/∂y + ∂u_y/∂x)
  ε_yz = ½(∂u_y/∂z + ∂u_z/∂y)
  ε_zx = ½(∂u_z/∂x + ∂u_x/∂z)

  Voight 记号：ε = [ε_xx, ε_yy, ε_zz, 2ε_yz, 2ε_zx, 2ε_xy]
  （工程剪应变 γ_ij = 2ε_ij）

=== 广义胡克定律（本构关系）===

  各向同性弹性体的应力-应变关系（3D）：

  [ε_xx]   1/E [ 1  -ν  -ν   0   0   0] [σ_xx]
  [ε_yy] = 1/E [-ν   1  -ν   0   0   0] [σ_yy]
  [ε_zz]   1/E [-ν  -ν   1   0   0   0] [σ_zz]
  [γ_yz] = 1/G [ 0   0   0   1   0   0] [τ_yz]
  [γ_zx] = 1/G [ 0   0   0   0   1   0] [τ_zx]
  [γ_xy] = 1/G [ 0   0   0   0   0   1] [τ_xy]

  或逆形式（应力 = 刚度 × 应变）：

  σ_xx = 2G ε_xx + λ (ε_xx + ε_yy + ε_zz)
  σ_yy = 2G ε_yy + λ (ε_xx + ε_yy + ε_zz)
  σ_zz = 2G ε_zz + λ (ε_xx + ε_yy + ε_zz)
  τ_yz = G γ_yz,  τ_zx = G γ_zx,  τ_xy = G γ_xy

  其中 λ = Eν/[(1+ν)(1-2ν)] 为 Lamé 第一参数
       G = E/[2(1+ν)]      为剪切模量（Lamé 第二参数 μ）

=== 弹性常数关系 ===

  已知 E, ν 可推导所有其他弹性常数：
    G  = E / [2(1+ν)]        剪切模量
    K  = E / [3(1-2ν)]       体积模量
    λ  = Eν / [(1+ν)(1-2ν)]  Lamé 第一参数

  约束：0 ≤ ν < 0.5（稳定材料），G > 0, K > 0

  互逆关系：
    E = 2G(1+ν) = 3K(1-2ν) = 9KG / (3K+G)
    ν = E/(2G) - 1 = (3K-E)/(6K) = (3K-2G)/(2(3K-G))

=== 经典均匀应力状态 ===

  1) 单轴拉伸（σ_xx = σ, 其余 = 0）：
     ε_xx = σ/E,  ε_yy = ε_zz = -νσ/E
     体积应变 θ = ε_xx + ε_yy + ε_zz = σ(1-2ν)/E
     体积模量验证：θ = σ_v/K, σ_v = σ/3（平均应力）

  2) 静水压缩（σ_xx = σ_yy = σ_zz = -p）：
     ε_xx = ε_yy = ε_zz = -p(1-2ν)/E = -p/(3K)
     体积应变 θ = -3p(1-2ν)/E = -p/K
     → K = p / |θ|（定义体积模量）

  3) 纯剪切（τ_xy = τ, 其余 = 0）：
     γ_xy = τ/G
     → G = τ / γ_xy（定义剪切模量）

=== 弹性应变能密度 ===

  U = ½ σ : ε = ½ (σ_xx ε_xx + σ_yy ε_yy + σ_zz ε_zz + τ_yz γ_yz + τ_zx γ_zx + τ_xy γ_xy)

  各状态：
    单轴：U = σ²/(2E)
    静水：U = 3p²/(2E) · (1-2ν) = p²/(2K)
    纯剪：U = τ²/(2G)

=== 弹性波速 ===

  纵波（P 波）：c_p = √[(λ+2G)/ρ] = √[E(1-ν) / (ρ(1+ν)(1-2ν))]
  横波（S 波）：c_s = √(G/ρ) = √[E / (2ρ(1+ν))]
  波速比：c_p/c_s = √[2(1-ν)/(1-2ν)] > √2

=== 与已有 MEC 模型的关系 ===

  MEC-050 梁 → MEC-053 弹性体：EI = E · bh³/12 中的 E 就是这里的一维特例
  MEC-051 板 → MEC-053 弹性体：D = Eh³/[12(1-ν²)] 是平面应变特例
  MEC-052 壳 → MEC-053 弹性体：壳弯曲刚度中的 D 同 MEC-051
  单轴拉伸 → MEC-010 弹簧：F = kΔx, k = EA/L, σ = Eε

  ν → 0 时各方向独立（无耦合）
  ν → 0.5 时不可压缩（体积不变），K → ∞

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(E=2.0e11, nu=0.3, rho=7850.0):
    """验证物理参数合法性。"""
    assert E > 0, f"杨氏模量 E 必须为正，当前 E={E}"
    assert 0 <= nu < 0.5, f"泊松比 ν 应在 [0, 0.5)，当前 ν={nu}"
    assert rho > 0, f"密度 rho 必须为正，当前 rho={rho}"


# ---------------------------------------------------------------------------
# 弹性常数换算
# ---------------------------------------------------------------------------

def shear_modulus(E, nu):
    """剪切模量 G = E / [2(1+ν)]。"""
    return E / (2.0 * (1.0 + nu))


def bulk_modulus(E, nu):
    """体积模量 K = E / [3(1-2ν)]。"""
    return E / (3.0 * (1.0 - 2.0 * nu))


def lame_first(E, nu):
    """Lamé 第一参数 λ = Eν / [(1+ν)(1-2ν)]。"""
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def lame_second(E, nu):
    """Lamé 第二参数 μ = G = E / [2(1+ν)]。"""
    return shear_modulus(E, nu)


def all_elastic_constants(E, nu):
    """返回所有弹性常数 (G, K, λ, μ)。"""
    G = shear_modulus(E, nu)
    K = bulk_modulus(E, nu)
    lam = lame_first(E, nu)
    mu = lame_second(E, nu)
    return G, K, lam, mu


def inverse_from_GK(G, K):
    """从 G, K 反推 E, ν。

    E = 9KG / (3K+G)
    ν = (3K - 2G) / (2(G + 3K))
    """
    E = 9.0 * K * G / (3.0 * K + G)
    nu = (3.0 * K - 2.0 * G) / (2.0 * (G + 3.0 * K))
    return E, nu


# ---------------------------------------------------------------------------
# 本构矩阵
# ---------------------------------------------------------------------------

def compliance_matrix(E, nu):
    """各向同性柔度矩阵 S (6×6)。

    ε = S σ, Voight 记号 [ε_xx, ε_yy, ε_zz, γ_yz, γ_zx, γ_xy]
    """
    G = shear_modulus(E, nu)
    S = np.zeros((6, 6))
    S[0, 0] = S[1, 1] = S[2, 2] = 1.0 / E
    S[0, 1] = S[0, 2] = S[1, 0] = S[1, 2] = S[2, 0] = S[2, 1] = -nu / E
    S[3, 3] = S[4, 4] = S[5, 5] = 1.0 / G
    return S


def stiffness_matrix(E, nu):
    """各向同性刚度矩阵 C (6×6)。

    σ = C ε
    """
    G = shear_modulus(E, nu)
    lam = lame_first(E, nu)
    C = np.zeros((6, 6))
    # 法向部分
    for i in range(3):
        C[i, i] = 2.0 * G + lam
        for j in range(3):
            if i != j:
                C[i, j] = lam
    # 剪切部分
    C[3, 3] = C[4, 4] = C[5, 5] = G
    return C


def stress_from_strain(epsilon, E, nu):
    """从应变计算应力（广义胡克定律）。

    epsilon: Voight [ε_xx, ε_yy, ε_zz, γ_yz, γ_zx, γ_xy]
    返回 Voight [σ_xx, σ_yy, σ_zz, τ_yz, τ_zx, τ_xy]
    """
    C = stiffness_matrix(E, nu)
    return C @ np.asarray(epsilon, dtype=float)


def strain_from_stress(sigma, E, nu):
    """从应力计算应变。

    sigma: Voight [σ_xx, σ_yy, σ_zz, τ_yz, τ_zx, τ_xy]
    返回 Voight [ε_xx, ε_yy, ε_zz, γ_yz, γ_zx, γ_xy]
    """
    S = compliance_matrix(E, nu)
    return S @ np.asarray(sigma, dtype=float)


# ---------------------------------------------------------------------------
# 经典均匀应力状态
# ---------------------------------------------------------------------------

def uniaxial_tension(sigma, E, nu):
    """单轴拉伸：σ_xx = sigma，其余应力为零。

    返回 (epsilon_voight, strain_energy_density)。
    """
    sigma_voight = np.array([sigma, 0, 0, 0, 0, 0])
    eps = strain_from_stress(sigma_voight, E, nu)
    U = 0.5 * sigma * eps[0]  # σ_xx · ε_xx / 2
    return eps, U


def hydrostatic_compression(p, E, nu):
    """静水压缩：σ_xx = σ_yy = σ_zz = -p。

    返回 (epsilon_voight, strain_energy_density)。
    """
    sigma_voight = np.array([-p, -p, -p, 0, 0, 0])
    eps = strain_from_stress(sigma_voight, E, nu)
    U = 0.5 * np.dot(sigma_voight, eps)
    return eps, U


def pure_shear(tau, E, nu):
    """纯剪切：τ_xy = tau，其余应力为零。

    返回 (epsilon_voight, strain_energy_density)。
    """
    sigma_voight = np.array([0, 0, 0, 0, 0, tau])
    eps = strain_from_stress(sigma_voight, E, nu)
    U = 0.5 * tau * eps[5]  # τ_xy · γ_xy / 2
    return eps, U


def volumetric_strain(epsilon_voight):
    """体积应变 θ = ε_xx + ε_yy + ε_zz。"""
    return epsilon_voight[0] + epsilon_voight[1] + epsilon_voight[2]


def mean_stress(sigma_voight):
    """平均应力 σ_m = (σ_xx + σ_yy + σ_zz) / 3。"""
    return (sigma_voight[0] + sigma_voight[1] + sigma_voight[2]) / 3.0


def deviatoric_stress(sigma_voight):
    """应力偏量张量 s = σ - σ_m δ_ij。"""
    s = np.copy(sigma_voight)
    s_m = mean_stress(sigma_voight)
    s[0] -= s_m
    s[1] -= s_m
    s[2] -= s_m
    return s


# ---------------------------------------------------------------------------
# 应变能密度
# ---------------------------------------------------------------------------

def strain_energy_density(sigma_voight, epsilon_voight):
    """弹性应变能密度 U = ½ σ : ε。"""
    return 0.5 * np.dot(sigma_voight, epsilon_voight)


# ---------------------------------------------------------------------------
# 弹性波速
# ---------------------------------------------------------------------------

def p_wave_speed(E, nu, rho):
    """纵波（P 波）速度 c_p = √[(λ+2G)/ρ]。"""
    lam = lame_first(E, nu)
    G = shear_modulus(E, nu)
    return np.sqrt((lam + 2 * G) / rho)


def s_wave_speed(E, nu, rho):
    """横波（S 波）速度 c_s = √(G/ρ)。"""
    G = shear_modulus(E, nu)
    return np.sqrt(G / rho)


def wave_speed_ratio(nu):
    """波速比 c_p/c_s = √[2(1-ν)/(1-2ν)]。"""
    return np.sqrt(2.0 * (1.0 - nu) / (1.0 - 2.0 * nu))


# ---------------------------------------------------------------------------
# 退化验证
# ---------------------------------------------------------------------------

def degradation_uncoupled(E, nu_zero=0.0):
    """ν→0 时各方向独立（无 Poisson 耦合）。

    验证：S[0,1] = -ν/E → 0
    """
    S = compliance_matrix(E, nu_zero)
    return S[0, 1]  # 应为 0


def incompressibility_limit():
    """ν→0.5 时不可压缩（K→∞）。

    验证：K = E/[3(1-2ν)] → ∞
    """
    E = 1.0
    nu_values = np.array([0.0, 0.25, 0.4, 0.45, 0.49, 0.499, 0.4999])
    K_values = E / (3 * (1 - 2 * nu_values))
    return nu_values, K_values


def check_incompressibility(nu, tol=0.01):
    """检查接近不可压缩时体积应变趋近于零。"""
    p = 1.0e6
    E = 2.0e11
    eps, _ = hydrostatic_compression(p, E, nu)
    theta = volumetric_strain(eps)
    # θ = -p/K = -p·3(1-2ν)/E
    K = bulk_modulus(E, nu)
    expected = -p / K
    return abs(theta - expected) / abs(expected)
