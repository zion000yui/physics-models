"""MEC-100-elastoplasticity — 模型定义（引擎无关）

弹塑性力学：塑性屈服准则与弹塑性本构关系。
在 MEC-053 三维弹性体基础上，引入塑性变形。

=== 物理系统 ===

  弹塑性材料：在小应变范围内先弹后塑。
  - 弹性阶段：σ = Eε（广义胡克定律，MEC-053）
  - 塑性阶段：永久变形，卸载后不回零

=== 屈服准则 ===

  1) Tresca 准则（最大剪应力准则）：
     τ_max = (σ₁ - σ₃)/2 ≤ σ_y/2
     等效：max(|σ₁-σ₂|, |σ₂-σ₃|, |σ₃-σ₁|) ≤ σ_y

  2) von Mises 准则（畸变能准则）：
     σ_eq = √(½[(σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²]) ≤ σ_y

     等效应力（张量形式）：
     σ_eq = √(3/2 · s_ij s_ij)
     其中 s_ij = σ_ij - σ_m δ_ij 为应力偏量

=== 单轴应力-应变关系 ===

  1) 理想弹塑性（perfectly plastic）：
     |σ| < σ_y: σ = Eε（弹性）
     |σ| = σ_y: σ = σ_y（塑性流动，应变继续增大但应力不变）

  2) 线性硬化弹塑性（linear hardening）：
     |σ| < σ_y: σ = Eε
     |σ| ≥ σ_y: σ = σ_y + H(ε - ε_y)
     H 为塑性硬化模量

  3) 幂律硬化（power-law hardening, Hollomon）：
     σ = σ_y + K ε_p^n  （ε_p 为塑性应变）

=== 等效（单轴）本构 ===

  弹性：σ_eq = E ε_eq
  塑性：σ_eq = σ_y + f(ε_p)

  卸载：沿弹性斜率 E 回弹

=== 能量 ===

  弹性应变能：U_e = σ²/(2E)
  塑性耗散：U_p = ∫ σ dε_p ≥ 0（不可逆）

=== 与已有 MEC 模型的关系 ===

  MEC-053 弹性体 → MEC-100 弹塑性体（弹性 → 弹塑）
  MEC-050 梁 → MEC-100 梁的塑性铰
  ν → 0.5 不可压缩 → MEC-100 塑性流动的体积守恒

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(E=2.0e11, nu=0.3, sigma_y=250e6,
                        H=0.0, K=0.0, n=0.0):
    """验证物理参数合法性。"""
    assert E > 0, f"杨氏模量 E 必须为正，当前 E={E}"
    assert 0 <= nu < 0.5, f"泊松比 ν 应在 [0, 0.5)，当前 ν={nu}"
    assert sigma_y > 0, f"屈服强度 σ_y 必须为正，当前 σ_y={sigma_y}"
    assert H >= 0, f"硬化模量 H 必须非负，当前 H={H}"
    assert K >= 0, f"幂律系数 K 必须非负，当前 K={K}"
    assert 0 <= n <= 1, f"幂律指数 n 应在 [0, 1]，当前 n={n}"


# ============================================================
# 屈服准则
# ============================================================

def tresca_yield(sigma_voight):
    """Tresca 等效应力 = max|σ_i - σ_j| / 2。

    对 6 分量 Voight，先计算主应力。
    """
    if len(sigma_voight) == 6:
        # 构造应力张量并求主应力
        sig = np.array([[sigma_voight[0], sigma_voight[5], sigma_voight[4]],
                        [sigma_voight[5], sigma_voight[1], sigma_voight[3]],
                        [sigma_voight[4], sigma_voight[3], sigma_voight[2]]])
        principals = np.sort(np.linalg.eigvalsh(sig))[::-1]
        diffs = [abs(principals[0] - principals[1]),
                 abs(principals[1] - principals[2]),
                 abs(principals[2] - principals[0])]
        return max(diffs) / 2.0
    s1, s2, s3 = sigma_voight[0], sigma_voight[1], sigma_voight[2]
    diffs = [abs(s1 - s2), abs(s2 - s3), abs(s3 - s1)]
    return max(diffs) / 2.0


def tresca_check(sigma_voight, sigma_y, tol=1e-6):
    """Tresca 屈服判断：τ_max ≤ σ_y/2。"""
    return tresca_yield(sigma_voight) <= sigma_y / 2.0 * (1 + tol)


def von_mises_stress(sigma_voight):
    """von Mises 等效应力。

    对 6 分量 Voight 向量：σ_eq = √(3/2 · s:s)
    s 为应力偏量。

    对仅 3 分量主应力：σ_eq = √(½[(σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²])
    """
    if len(sigma_voight) == 6:
        return von_mises_from_deviator(sigma_voight)
    s1, s2, s3 = sigma_voight[0], sigma_voight[1], sigma_voight[2]
    return np.sqrt(0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))


def von_mises_check(sigma_voight, sigma_y, tol=1e-6):
    """von Mises 屈服判断：σ_eq ≤ σ_y。"""
    return von_mises_stress(sigma_voight) <= sigma_y * (1 + tol)


def stress_deviator(sigma_voight):
    """应力偏量 s = σ - σ_m δ。"""
    s = np.copy(sigma_voight)
    s_m = (sigma_voight[0] + sigma_voight[1] + sigma_voight[2]) / 3.0
    s[0] -= s_m
    s[1] -= s_m
    s[2] -= s_m
    return s


def von_mises_from_deviator(sigma_voight):
    """从应力偏量计算 von Mises 应力：σ_eq = √(3/2 s:s)。"""
    s = stress_deviator(sigma_voight)
    s_dot = np.sum(s[:3]**2) + 2 * np.sum(sigma_voight[3:]**2)  # Voight: γ=2ε
    # 实际上对于 Voight [σ_xx, σ_yy, σ_zz, τ_yz, τ_zx, τ_xy]
    # s:s = s_xx² + s_yy² + s_zz² + 2(τ_yz² + τ_zx² + τ_xy²)
    return np.sqrt(1.5 * (s[0]**2 + s[1]**2 + s[2]**2
                          + 2 * (sigma_voight[3]**2 + sigma_voight[4]**2 + sigma_voight[5]**2)))


# ============================================================
# 单轴弹塑性本构
# ============================================================

def uniaxial_stress_strain(eps, E, sigma_y, H=0.0, K=0.0, n=0.0):
    """单轴弹塑性应力-应变关系。

    弹性：|ε| < ε_y → σ = Eε,  ε_y = σ_y/E
    塑性：
      线性硬化：σ = σ_y + H(ε - ε_y)
      幂律硬化：σ = σ_y + K ε_p^n  （ε_p = ε - σ/E）
      理想塑性：σ = σ_y（H=K=0）

    返回 (sigma, is_plastic)。
    """
    eps = np.asarray(eps, dtype=float)
    eps_y = sigma_y / E
    sigma = np.empty_like(eps)
    is_plastic = np.abs(eps) > eps_y

    # 弹性
    sigma[~is_plastic] = E * eps[~is_plastic]

    # 塑性
    eps_p = np.abs(eps[is_plastic]) - eps_y  # 塑性应变
    sign = np.sign(eps[is_plastic])

    if H > 0:
        # 线性硬化
        sigma[is_plastic] = sign * (sigma_y + H * eps_p)
    elif K > 0 and n > 0:
        # 幂律硬化
        sigma[is_plastic] = sign * (sigma_y + K * eps_p**n)
    else:
        # 理想塑性
        sigma[is_plastic] = sign * sigma_y

    return sigma, is_plastic


def yield_strain(E, sigma_y):
    """屈服应变 ε_y = σ_y/E。"""
    return sigma_y / E


def plastic_strain(eps, E, sigma_y):
    """塑性应变 ε_p = |ε| - σ_y/E（当 |ε| > ε_y）。"""
    eps = np.asarray(eps, dtype=float)
    eps_y = sigma_y / E
    eps_p = np.maximum(0, np.abs(eps) - eps_y)
    return eps_p


# ============================================================
# 弹塑性能量
# ============================================================

def elastic_energy(sigma, E):
    """弹性应变能密度 U_e = σ²/(2E)。"""
    return sigma**2 / (2 * E)


def plastic_dissipation(eps, E, sigma_y, H=0.0):
    """塑性耗散密度 U_p = ∫ σ dε_p。"""
    eps = np.asarray(eps, dtype=float)
    eps_y = sigma_y / E
    eps_p = np.maximum(0, np.abs(eps) - eps_y)

    if H > 0:
        # 线性硬化：∫₀^εp (σ_y + H s) ds = σ_y ε_p + ½ H ε_p²
        return sigma_y * eps_p + 0.5 * H * eps_p**2
    else:
        # 理想塑性：∫₀^εp σ_y ds = σ_y ε_p
        return sigma_y * eps_p


def total_energy(eps, E, sigma_y, H=0.0):
    """总能量 = 弹性 + 塑性耗散。"""
    sigma, _ = uniaxial_stress_strain(eps, E, sigma_y, H)
    U_e = elastic_energy(sigma, E)
    U_p = plastic_dissipation(eps, E, sigma_y, H)
    return U_e + U_p


# ============================================================
# 卸载与再加载
# ============================================================

def unload_stress(eps_max, E, sigma_y, H=0.0):
    """从最大应变 ε_max 卸载到零应力时的残余应变。

    卸载沿弹性斜率 E：
    σ_unload = σ(ε_max) - E · (ε_max - ε_residual)
    残余应变 ε_res = ε_max - σ(ε_max)/E
    """
    eps = np.array([eps_max])
    sigma, _ = uniaxial_stress_strain(eps, E, sigma_y, H)
    sigma_max = sigma[0]
    eps_res = eps_max - sigma_max / E
    return eps_res, sigma_max


def bauschinger_effect(eps_max, E, sigma_y, H=0.0):
    """Bauschinger 效应：反向加载屈服强度。

    理想塑性（H=0）：反向屈服 σ_y_reverse = -σ_y（各向同性硬化为零）
    线性硬化（H>0）：反向屈服 ≈ -(σ_y + 2H·ε_p)
    （运动硬化模型，简化）
    """
    eps_y = sigma_y / E
    eps_p = max(0, abs(eps_max) - eps_y)
    if H > 0:
        # 运动硬化：反向屈服 ≈ σ_y + H·ε_p - 2(σ_y + H·ε_p)
        # 简化：反向屈服应力 = -(σ_y + 2H·ε_p)
        return -(sigma_y + 2 * H * eps_p)
    else:
        return -sigma_y


# ============================================================
# 退化验证
# ============================================================

def degradation_to_elastic(E, sigma_y, H=0.0):
    """σ_y → ∞ 时退化为纯弹性（MEC-053 单轴）。"""
    # 当 σ_y 很大时，所有应变都是弹性的
    eps_test = np.array([0.001])
    sigma_large_y, is_pl = uniaxial_stress_strain(
        eps_test, E, sigma_y * 1e10, H)
    sigma_elastic = E * eps_test
    return abs(sigma_large_y[0] - sigma_elastic[0]) / sigma_elastic[0]


def check_pure_elastic_when_below_yield(E, sigma_y):
    """应力低于屈服时应与纯弹性一致。"""
    eps = np.array([sigma_y / (2 * E)])  # 一半屈服应变
    sigma, is_pl = uniaxial_stress_strain(eps, E, sigma_y)
    assert not is_pl[0], "低于屈服应变不应进入塑性"
    assert abs(sigma[0] - E * eps[0]) < 1e-15
