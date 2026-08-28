"""MEC-053 —— 一致性测试：三维弹性体。

验证：
- 弹性常数换算
- 本构矩阵 C@S = I
- 单轴/静水/剪切经典状态
- 应变能密度
- 弹性波速
- 退化与极限
- 反例验证
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC053_consistency.py
"""

import numpy as np

from model import (
    validate_parameters,
    shear_modulus,
    bulk_modulus,
    lame_first,
    lame_second,
    all_elastic_constants,
    inverse_from_GK,
    compliance_matrix,
    stiffness_matrix,
    stress_from_strain,
    strain_from_stress,
    uniaxial_tension,
    hydrostatic_compression,
    pure_shear,
    volumetric_strain,
    mean_stress,
    deviatoric_stress,
    strain_energy_density,
    p_wave_speed,
    s_wave_speed,
    wave_speed_ratio,
    degradation_uncoupled,
    check_incompressibility,
)

TOL = 1e-10

E = 2.0e11
NU = 0.3
RHO = 7850.0


def test_elastic_constant_relations():
    """弹性常数应满足所有换算关系。"""
    G = shear_modulus(E, NU)
    K = bulk_modulus(E, NU)
    lam = lame_first(E, NU)
    mu = lame_second(E, NU)

    # G = E / [2(1+ν)]
    assert abs(G - E / (2 * (1 + NU))) < TOL
    # K = E / [3(1-2ν)]
    assert abs(K - E / (3 * (1 - 2 * NU))) < TOL
    # λ = Eν / [(1+ν)(1-2ν)]
    assert abs(lam - E * NU / ((1 + NU) * (1 - 2 * NU))) < TOL
    # μ = G
    assert abs(mu - G) < TOL
    # E = 2G(1+ν)
    assert abs(E - 2 * G * (1 + NU)) < TOL * E
    # E = 3K(1-2ν)
    assert abs(E - 3 * K * (1 - 2 * NU)) < TOL * E


def test_inverse_from_GK():
    """从 G, K 反推 E, ν 应正确。"""
    G = shear_modulus(E, NU)
    K = bulk_modulus(E, NU)
    E_back, nu_back = inverse_from_GK(G, K)
    assert abs(E_back - E) / E < 1e-10
    assert abs(nu_back - NU) / NU < 1e-10


def test_constitutive_matrix_inverse():
    """刚度矩阵和柔度矩阵应互逆：C @ S = I。"""
    C = stiffness_matrix(E, NU)
    S = compliance_matrix(E, NU)
    I_check = C @ S
    err = np.max(np.abs(I_check - np.eye(6)))
    assert err < 1e-10, f"C @ S = I 误差 {err:.3e}"


def test_stress_strain_roundtrip():
    """应力→应变→应力应可逆。"""
    sigma = np.array([100e6, 50e6, -30e6, 20e6, -10e6, 40e6])
    eps = strain_from_stress(sigma, E, NU)
    sigma_back = stress_from_strain(eps, E, NU)
    err = np.max(np.abs(sigma_back - sigma)) / np.max(np.abs(sigma))
    assert err < 1e-10, f"应力应变往返误差 {err:.3e}"


def test_uniaxial_tension():
    """单轴拉伸应满足 σ=Eε, 横向应变 = -ν ε。"""
    sigma0 = 100e6
    eps, U = uniaxial_tension(sigma0, E, NU)
    assert abs(eps[0] - sigma0 / E) < TOL
    assert abs(eps[1] - (-NU * sigma0 / E)) < TOL
    assert abs(eps[2] - (-NU * sigma0 / E)) < TOL
    # 体积应变 θ = σ(1-2ν)/E
    theta = volumetric_strain(eps)
    assert abs(theta - sigma0 * (1 - 2 * NU) / E) < TOL
    # 应变能 = σ²/(2E)
    assert abs(U - sigma0**2 / (2 * E)) < TOL


def test_hydrostatic_compression():
    """静水压缩应满足 θ = -p/K。"""
    p = 100e6
    eps, U = hydrostatic_compression(p, E, NU)
    K = bulk_modulus(E, NU)
    theta = volumetric_strain(eps)
    assert abs(theta - (-p / K)) < TOL
    # 各方向应变相同
    assert abs(eps[0] - eps[1]) < TOL
    assert abs(eps[1] - eps[2]) < TOL
    # 应变能 = p²/(2K)
    assert abs(U - p**2 / (2 * K)) < TOL


def test_pure_shear():
    """纯剪切应满足 γ = τ/G。"""
    tau = 50e6
    eps, U = pure_shear(tau, E, NU)
    G = shear_modulus(E, NU)
    assert abs(eps[5] - tau / G) < TOL
    # 无正应变
    assert abs(eps[0]) < TOL
    assert abs(eps[1]) < TOL
    assert abs(eps[2]) < TOL
    # 应变能 = τ²/(2G)
    assert abs(U - tau**2 / (2 * G)) < TOL


def test_volumetric_strain_uniaxial():
    """单轴拉伸体积应变 θ = ε(1-2ν) > 0（ν<0.5）。"""
    sigma = 50e6
    eps, _ = uniaxial_tension(sigma, E, NU)
    theta = volumetric_strain(eps)
    assert theta > 0, "ν<0.5 时单轴拉伸应体积增大"
    expected = sigma / E * (1 - 2 * NU)
    assert abs(theta - expected) < TOL


def test_deviatoric_stress_hydrostatic():
    """静水压缩时应力偏量应为零。"""
    p = 100e6
    sigma = np.array([-p, -p, -p, 0, 0, 0])
    s = deviatoric_stress(sigma)
    assert np.max(np.abs(s)) < TOL


def test_wave_speeds():
    """波速应满足 c_p > c_s 且 c_p/c_s = √[2(1-ν)/(1-2ν)]。"""
    c_p = p_wave_speed(E, NU, RHO)
    c_s = s_wave_speed(E, NU, RHO)
    assert c_p > c_s, "P 波应快于 S 波"
    ratio = c_p / c_s
    expected = wave_speed_ratio(NU)
    assert abs(ratio - expected) < 1e-10
    assert ratio > np.sqrt(2), "c_p/c_s 应 > √2"


def test_p_wave_formula():
    """纵波速度应等于 √[(λ+2G)/ρ]。"""
    c_p = p_wave_speed(E, NU, RHO)
    lam = lame_first(E, NU)
    G = shear_modulus(E, NU)
    expected = np.sqrt((lam + 2 * G) / RHO)
    assert abs(c_p - expected) < 1e-6


def test_degradation_nu_zero():
    """ν→0 时柔度矩阵非对角项趋零。"""
    S = compliance_matrix(E, 0.0)
    assert abs(S[0, 1]) < 1e-15
    assert abs(S[0, 2]) < 1e-15


def test_incompressibility():
    """ν→0.5 时体积模量发散。"""
    nu_near = 0.4999
    K = bulk_modulus(E, nu_near)
    K_half = bulk_modulus(E, 0.5 - 1e-10)
    assert K < K_half, "ν 更接近 0.5 时 K 应更大"

    # 体积应变趋近零
    p = 1.0e6
    eps, _ = hydrostatic_compression(p, E, nu_near)
    theta = volumetric_strain(eps)
    eps_low, _ = hydrostatic_compression(p, E, 0.3)
    theta_low = volumetric_strain(eps_low)
    assert abs(theta) < abs(theta_low), "ν→0.5 时体积应变应更小"


def test_error_injection_wrong_E():
    """反例：E 翻倍应导致单轴应变减半。"""
    eps1, _ = uniaxial_tension(100e6, E, NU)
    eps2, _ = uniaxial_tension(100e6, 2 * E, NU)
    assert abs(eps2[0] - eps1[0] / 2) < TOL


def test_error_injection_wrong_nu():
    """反例：ν 翻倍应改变横向应变。"""
    eps1, _ = uniaxial_tension(100e6, E, 0.1)
    eps2, _ = uniaxial_tension(100e6, E, 0.2)
    # ε_yy = -νσ/E, 翻倍 ν 应翻倍 |ε_yy|
    assert abs(eps2[1] - 2 * eps1[1]) < TOL


def test_shear_decoupled_from_normal():
    """剪切分量不应与法向分量耦合。"""
    sigma = np.array([100e6, 0, 0, 0, 0, 50e6])
    eps = strain_from_stress(sigma, E, NU)
    # γ_xy 不受 σ_xx 影响
    G = shear_modulus(E, NU)
    assert abs(eps[5] - 50e6 / G) < TOL
    # ε_xx 不受 τ_xy 影响
    assert abs(eps[0] - 100e6 / E) < TOL


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    bad_cases = [
        ("E", {"E": -1}),
        ("ν", {"nu": 0.6}),
        ("ν", {"nu": 0.5}),
        ("ν", {"nu": -0.1}),
        ("rho", {"rho": -1}),
    ]
    for label, kwargs in bad_cases:
        try:
            validate_parameters(**kwargs)
            assert False, f"应拒绝 {kwargs}"
        except AssertionError as e:
            assert label in str(e), f"错误信息应包含 {label}: {e}"


if __name__ == "__main__":
    test_elastic_constant_relations()
    print("✓ 弹性常数换算")
    test_inverse_from_GK()
    print("✓ G→E,ν 逆推")
    test_constitutive_matrix_inverse()
    print("✓ C @ S = I")
    test_stress_strain_roundtrip()
    print("✓ 应力-应变往返")
    test_uniaxial_tension()
    print("✓ 单轴拉伸")
    test_hydrostatic_compression()
    print("✓ 静水压缩")
    test_pure_shear()
    print("✓ 纯剪切")
    test_volumetric_strain_uniaxial()
    print("✓ 体积应变")
    test_deviatoric_stress_hydrostatic()
    print("✓ 应力偏量")
    test_wave_speeds()
    print("✓ 弹性波速")
    test_p_wave_formula()
    print("✓ P 波公式")
    test_degradation_nu_zero()
    print("✓ ν→0 退化")
    test_incompressibility()
    print("✓ 不可压缩极限")
    test_error_injection_wrong_E()
    print("✓ 反例: E 翻倍")
    test_error_injection_wrong_nu()
    print("✓ 反例: ν 翻倍")
    test_shear_decoupled_from_normal()
    print("✓ 剪切-法向解耦")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-053 所有一致性测试通过")
