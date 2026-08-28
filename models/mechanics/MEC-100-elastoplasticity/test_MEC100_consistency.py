"""MEC-100 —— 一致性测试：弹塑性力学。

验证：
- 单轴弹塑性应力-应变
- 屈服准则（Tresca / von Mises）
- 静水压力不屈服
- 弹塑性能量
- 卸载残余应变
- 退化到纯弹性
- 反例验证
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC100_consistency.py
"""

import numpy as np

from model import (
    validate_parameters,
    tresca_yield,
    tresca_check,
    von_mises_stress,
    von_mises_check,
    von_mises_from_deviator,
    stress_deviator,
    uniaxial_stress_strain,
    yield_strain,
    plastic_strain,
    elastic_energy,
    plastic_dissipation,
    total_energy,
    unload_stress,
    bauschinger_effect,
    degradation_to_elastic,
    check_pure_elastic_when_below_yield,
)

TOL = 1e-6

E = 2.0e11
NU = 0.3
SIGMA_Y = 250e6
H = 5e9


def test_yield_strain():
    """屈服应变 ε_y = σ_y/E。"""
    eps_y = yield_strain(E, SIGMA_Y)
    assert abs(eps_y - SIGMA_Y / E) < 1e-15


def test_elastic_region():
    """弹性段应满足 σ = Eε。"""
    eps_y = yield_strain(E, SIGMA_Y)
    eps = np.array([eps_y * 0.5])
    sigma, is_pl = uniaxial_stress_strain(eps, E, SIGMA_Y, H)
    assert not is_pl[0]
    assert abs(sigma[0] - E * eps[0]) < 1e-15


def test_ideal_plastic():
    """理想塑性：塑性段应力恒为 σ_y。"""
    eps_y = yield_strain(E, SIGMA_Y)
    eps = np.array([3 * eps_y])
    sigma, is_pl = uniaxial_stress_strain(eps, E, SIGMA_Y, H=0.0)
    assert is_pl[0]
    assert abs(sigma[0] - SIGMA_Y) < 1e-15


def test_linear_hardening():
    """线性硬化：σ = σ_y + H·ε_p。"""
    eps_y = yield_strain(E, SIGMA_Y)
    eps = np.array([3 * eps_y])
    sigma, is_pl = uniaxial_stress_strain(eps, E, SIGMA_Y, H=H)
    eps_p = 3 * eps_y - eps_y
    expected = SIGMA_Y + H * eps_p
    assert abs(sigma[0] - expected) < 1e-6 * abs(expected)


def test_stress_continuity():
    """弹性-塑性交界处应力应连续。"""
    eps_y = yield_strain(E, SIGMA_Y)
    eps = np.array([eps_y * 0.999, eps_y * 1.001])
    sigma, _ = uniaxial_stress_strain(eps, E, SIGMA_Y, H)
    assert abs(sigma[0] - sigma[1]) < SIGMA_Y * 0.01


def test_von_mises_uniaxial():
    """单轴应力下 von Mises = |σ|。"""
    sigma_v = np.array([100e6, 0, 0, 0, 0, 0])
    vm = von_mises_stress(sigma_v)
    assert abs(vm - 100e6) < TOL


def test_von_mises_hydrostatic():
    """静水压力下 von Mises = 0（不屈服）。"""
    sigma_v = np.array([-100e6, -100e6, -100e6, 0, 0, 0])
    vm = von_mises_stress(sigma_v)
    assert abs(vm) < 1e-3


def test_tresca_hydrostatic():
    """静水压力下 Tresca = 0（不屈服）。"""
    sigma_v = np.array([-100e6, -100e6, -100e6, 0, 0, 0])
    tr = tresca_yield(sigma_v)
    assert abs(tr) < 1e-3


def test_von_mises_pure_shear():
    """纯剪切下 von Mises = √3·τ。"""
    tau = 100e6
    sigma_v = np.array([0, 0, 0, 0, 0, tau])
    vm = von_mises_stress(sigma_v)
    assert abs(vm - np.sqrt(3) * tau) < TOL * tau


def test_tresca_pure_shear():
    """纯剪切下 Tresca = τ。"""
    tau = 100e6
    sigma_v = np.array([0, 0, 0, 0, 0, tau])
    tr = tresca_yield(sigma_v)
    assert abs(tr - tau) < TOL * tau


def test_yield_check_below():
    """应力低于屈服强度应不屈服。"""
    sigma_v = np.array([100e6, 0, 0, 0, 0, 0])
    assert von_mises_check(sigma_v, SIGMA_Y)
    assert tresca_check(sigma_v, SIGMA_Y)


def test_yield_check_above():
    """应力超过屈服强度应屈服。"""
    sigma_v = np.array([300e6, 0, 0, 0, 0, 0])
    assert not von_mises_check(sigma_v, SIGMA_Y)
    assert not tresca_check(sigma_v, SIGMA_Y)


def test_von_mises_equals_tresca_uniaxial():
    """单轴应力下 von Mises 和 Tresca 给出相同屈服判断。"""
    for sigma in [50e6, 250e6, 500e6]:
        sigma_v = np.array([sigma, 0, 0, 0, 0, 0])
        vm = von_mises_stress(sigma_v)
        tr = tresca_yield(sigma_v)
        # von Mises = |σ|, Tresca = |σ|/2
        # 但屈服判断：VM: σ≤σ_y, Tr: σ/2≤σ_y/2 → σ≤σ_y
        # 所以两者一致
        assert von_mises_check(sigma_v, SIGMA_Y) == tresca_check(sigma_v, SIGMA_Y)


def test_plastic_dissipation_positive():
    """塑性耗散应非负。"""
    eps_y = yield_strain(E, SIGMA_Y)
    eps = np.array([3 * eps_y])
    U_p = plastic_dissipation(eps, E, SIGMA_Y, H)
    assert U_p[0] > 0


def test_elastic_dissipation_zero():
    """弹性段塑性耗散应为零。"""
    eps_y = yield_strain(E, SIGMA_Y)
    eps = np.array([0.5 * eps_y])
    U_p = plastic_dissipation(eps, E, SIGMA_Y, H)
    assert U_p[0] < 1e-15


def test_unload_residual_strain():
    """卸载后应有残余应变（塑性变形不可逆）。"""
    eps_y = yield_strain(E, SIGMA_Y)
    eps_max = 3 * eps_y
    eps_res, sigma_max = unload_stress(eps_max, E, SIGMA_Y, H)
    assert eps_res > 0, "应有残余应变"
    assert eps_res < eps_max, "残余应变应小于最大应变"


def test_unload_residual_zero_if_elastic():
    """弹性范围内卸载应无残余应变。"""
    eps_y = yield_strain(E, SIGMA_Y)
    eps_max = 0.5 * eps_y
    eps_res, _ = unload_stress(eps_max, E, SIGMA_Y, H)
    assert abs(eps_res) < 1e-15


def test_degradation_to_elastic():
    """σ_y→∞ 时退化为纯弹性。"""
    err = degradation_to_elastic(E, SIGMA_Y, H)
    assert err < 1e-6


def test_error_injection_wrong_E():
    """反例：E 翻倍应使屈服应变减半。"""
    eps_y1 = yield_strain(E, SIGMA_Y)
    eps_y2 = yield_strain(2 * E, SIGMA_Y)
    assert abs(eps_y2 - eps_y1 / 2) < 1e-15


def test_error_injection_wrong_sigma_y():
    """反例：σ_y 翻倍应使屈服应变翻倍。"""
    eps_y1 = yield_strain(E, SIGMA_Y)
    eps_y2 = yield_strain(E, 2 * SIGMA_Y)
    assert abs(eps_y2 - 2 * eps_y1) < 1e-15


def test_dynamics_interface():
    """本构函数应返回正确形状。"""
    eps = np.array([0.001, 0.002, 0.003])
    sigma, is_pl = uniaxial_stress_strain(eps, E, SIGMA_Y, H)
    assert sigma.shape == (3,)
    assert is_pl.shape == (3,)


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    bad_cases = [
        ("E", {"E": -1}),
        ("ν", {"nu": 0.6}),
        ("σ_y", {"sigma_y": -1}),
        ("H", {"H": -1}),
        ("n", {"n": 1.5}),
    ]
    for label, kwargs in bad_cases:
        try:
            validate_parameters(**kwargs)
            assert False, f"应拒绝 {kwargs}"
        except AssertionError as e:
            assert label in str(e)


if __name__ == "__main__":
    test_yield_strain()
    print("✓ 屈服应变公式")
    test_elastic_region()
    print("✓ 弹性段 σ=Eε")
    test_ideal_plastic()
    print("✓ 理想塑性")
    test_linear_hardening()
    print("✓ 线性硬化")
    test_stress_continuity()
    print("✓ 应力连续性")
    test_von_mises_uniaxial()
    print("✓ von Mises 单轴")
    test_von_mises_hydrostatic()
    print("✓ von Mises 静水=0")
    test_tresca_hydrostatic()
    print("✓ Tresca 静水=0")
    test_von_mises_pure_shear()
    print("✓ von Mises 纯剪=√3τ")
    test_tresca_pure_shear()
    print("✓ Tresca 纯剪=τ")
    test_yield_check_below()
    print("✓ 低于屈服不屈服")
    test_yield_check_above()
    print("✓ 超过屈服屈服")
    test_von_mises_equals_tresca_uniaxial()
    print("✓ 单轴 VM=Tr 判断一致")
    test_plastic_dissipation_positive()
    print("✓ 塑性耗散>0")
    test_elastic_dissipation_zero()
    print("✓ 弹性段耗散=0")
    test_unload_residual_strain()
    print("✓ 残余应变")
    test_unload_residual_zero_if_elastic()
    print("✓ 弹性范围无残余")
    test_degradation_to_elastic()
    print("✓ 退化到纯弹性")
    test_error_injection_wrong_E()
    print("✓ 反例: E 翻倍")
    test_error_injection_wrong_sigma_y()
    print("✓ 反例: σ_y 翻倍")
    test_dynamics_interface()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-100 所有一致性测试通过")
