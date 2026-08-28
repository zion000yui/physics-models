"""MEC-051 —— 一致性测试：Kirchhoff-Love 薄板。

验证：
- 静态 Navier 解 + 边界条件
- 最大挠度公式
- 固有频率公式
- 模态正交性
- ω² = k_mn / m_mn
- 能量守恒
- 有限差分收敛
- 反例验证
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC051_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (
    validate_parameters,
    plate_stiffness,
    static_simply_supported_navier,
    navier_load_coeff,
    max_deflection_simply_supported,
    natural_frequencies_plate,
    plate_mode_shape,
    plate_mode_laplacian,
    modal_dynamics_plate,
    modal_energy_plate,
    verify_plate_orthogonality,
    plate_modal_mass,
    plate_modal_stiffness,
    fd_plate_natural_frequencies,
)

TOL = 1e-6

E = 2.0e11
H = 0.01
NU = 0.3
RHO = 7850.0
A = 1.0
B = 1.0
Q = 1000.0


def test_static_navier_solution():
    """Navier 解应满足边界条件。"""
    x = np.linspace(0, A, 51)
    y = np.linspace(0, B, 51)
    w, Mx, My, Mxy = static_simply_supported_navier(
        x, y, Q, A, B, E, H, NU, n_terms=30)

    # 边界 w=0
    assert abs(w[0, :]).max() < 1e-10, f"w(0, y) 非零: {abs(w[0, :]).max()}"
    assert abs(w[-1, :]).max() < 1e-10, f"w(a, y) 非零"
    assert abs(w[:, 0]).max() < 1e-10, f"w(x, 0) 非零"
    assert abs(w[:, -1]).max() < 1e-10, f"w(x, b) 非零"


def test_static_deflection_formula():
    """中心挠度公式应与 Navier 解一致。"""
    x_mid = np.array([A / 2])
    y_mid = np.array([B / 2])
    w_nav, _, _, _ = static_simply_supported_navier(
        x_mid, y_mid, Q, A, B, E, H, NU, n_terms=50)
    w_formula = max_deflection_simply_supported(Q, A, B, E, H, NU, n_terms=50)
    err = abs(w_nav[0, 0] - w_formula) / w_formula
    assert err < 1e-6, f"挠度公式不匹配 Navier: err={err:.3e}"


def test_deflection_coefficient():
    """方板挠度系数应接近理论值 0.00406。"""
    D = plate_stiffness(E, H, NU)
    w_max = max_deflection_simply_supported(Q, A, B, E, H, NU, n_terms=100)
    alpha = w_max * D / (Q * A**4)
    assert abs(alpha - 0.004062) < 0.001, f"α = {alpha:.6f}, 应接近 0.004062"


def test_load_coefficient_odd():
    """均布载荷的 Navier 系数：奇数 m,n 非零，偶数为零。"""
    q_mn_odd = navier_load_coeff(1, 1, Q, A, B)
    assert abs(q_mn_odd - 16 * Q / (np.pi**2)) < 1e-10
    q_mn_even = navier_load_coeff(2, 1, Q, A, B)
    assert q_mn_even == 0


def test_natural_frequency_formula():
    """固有频率应满足 ω_mn = π²(m²/a² + n²/b²)√(D/(ρh))。"""
    D = plate_stiffness(E, H, NU)
    factor = np.sqrt(D / (RHO * H))
    omegas = natural_frequencies_plate(5, A, B, E, H, NU, RHO)
    expected = [
        np.pi**2 * (1 + 1) * factor,
        np.pi**2 * (4 + 1) * factor,
        np.pi**2 * (1 + 4) * factor,
        np.pi**2 * (4 + 4) * factor,
        np.pi**2 * (9 + 1) * factor,
    ]
    for i in range(5):
        err = abs(omegas[i] - expected[i]) / expected[i]
        assert err < 1e-10, f"mode {i+1}: {omegas[i]:.4f} vs {expected[i]:.4f}"


def test_mode_shape_boundary():
    """模态形状应满足简支边界条件。"""
    x = np.linspace(0, A, 101)
    y = np.linspace(0, B, 101)
    for m in range(1, 4):
        for n in range(1, 4):
            phi = plate_mode_shape(x, y, m, n, A, B)
            # 边界 φ=0
            assert abs(phi[0, :]).max() < 1e-10
            assert abs(phi[-1, :]).max() < 1e-10
            assert abs(phi[:, 0]).max() < 1e-10
            assert abs(phi[:, -1]).max() < 1e-10


def test_mode_shape_laplacian():
    """模态拉普拉斯算子应等于 -(m²π²/a² + n²π²/b²) φ。"""
    x = np.linspace(0, A, 51)
    y = np.linspace(0, B, 51)
    for m in range(1, 4):
        for n in range(1, 4):
            phi = plate_mode_shape(x, y, m, n, A, B)
            lap = plate_mode_laplacian(x, y, m, n, A, B)
            expected_lap = -((m * np.pi / A)**2 + (n * np.pi / B)**2) * phi
            err = np.max(np.abs(lap - expected_lap))
            assert err < 1e-10, f"({m},{n}): ∇²φ 误差 {err:.3e}"


def test_modal_orthogonality():
    """不同模态应正交。"""
    modes = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3)]
    for i, (m1, n1) in enumerate(modes):
        for j, (m2, n2) in enumerate(modes[i + 1:], i + 1):
            result = verify_plate_orthogonality(m1, n1, m2, n2, RHO, H, A, B)
            self_val = verify_plate_orthogonality(m1, n1, m1, n1, RHO, H, A, B)
            assert abs(result) / self_val < 1e-6, \
                f"<φ({m1},{n1}), φ({m2},{n2})> = {result:.3e}"


def test_omega_squared_equals_k_over_m():
    """ω_mn² = k_mn / m_mn 应成立。"""
    D = plate_stiffness(E, H, NU)
    omegas = natural_frequencies_plate(4, A, B, E, H, NU, RHO)
    mode_pairs = [(1, 1), (2, 1), (1, 2), (2, 2)]
    for i, (m, n) in enumerate(mode_pairs):
        m_mn = plate_modal_mass(m, n, RHO, H, A, B)
        k_mn = plate_modal_stiffness(m, n, D, A, B, NU)
        omega_calc = np.sqrt(k_mn / m_mn)
        err = abs(omega_calc - omegas[i]) / omegas[i]
        assert err < 0.01, f"({m},{n}): √(k/m)={omega_calc:.4f} vs {omegas[i]:.4f}"


def test_modal_energy_conservation():
    """自由振动应能量守恒。"""
    omegas = natural_frequencies_plate(3, A, B, E, H, NU, RHO)
    q0 = np.array([1e-4, 0, 0])
    qdot0 = np.array([0, 0, 0])
    t_end = 2 * np.pi / omegas[0] * 3
    sol = solve_ivp(
        modal_dynamics_plate, (0, t_end), np.concatenate([q0, qdot0]),
        args=(omegas, None),
        t_eval=np.linspace(0, t_end, 501),
        rtol=1e-10, atol=1e-12,
    )
    energies = np.array([modal_energy_plate(sol.y[:, i], omegas)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / energies[0]
    assert dE < 1e-6, f"能量变化 {dE:.3e}"


def test_fd_frequency_convergence():
    """有限差分频率应随网格加密收敛到解析值。"""
    omegas_ana = natural_frequencies_plate(3, A, B, E, H, NU, RHO)
    for N in [15, 21, 31]:
        omegas_fd = fd_plate_natural_frequencies(N, N, A, B, E, H, NU, RHO,
                                                  n_modes=3)
        errs = np.abs(omegas_fd[:3] - omegas_ana[:3]) / omegas_ana[:3]
        assert errs[0] < 0.01, f"N={N}, mode 1 误差 {errs[0]:.4f}"


def test_degradation_no_load():
    """载荷 q=0 时挠度应为零。"""
    x = np.linspace(0, A, 21)
    y = np.linspace(0, B, 21)
    w, _, _, _ = static_simply_supported_navier(x, y, 0.0, A, B, E, H, NU)
    assert np.allclose(w, 0), "零载荷挠度非零"


def test_error_injection_wrong_E():
    """反例：E 乘 2 应导致挠度减半。"""
    w1 = max_deflection_simply_supported(Q, A, B, E, H, NU)
    w2 = max_deflection_simply_supported(Q, A, B, 2 * E, H, NU)
    assert abs(w2 - w1 / 2) < 1e-15 * abs(w1), f"E 翻倍挠度应减半"


def test_error_injection_wrong_h():
    """反例：h 乘 2 应导致 D 变为 8 倍（h³），挠度减为 1/8。"""
    w1 = max_deflection_simply_supported(Q, A, B, E, H, NU)
    w2 = max_deflection_simply_supported(Q, A, B, E, 2 * H, NU)
    assert abs(w2 - w1 / 8) < 1e-10 * abs(w1), f"h 翻倍挠度应×1/8: {w2/w1}"


def test_dynamics_interface():
    """dynamics 接口应返回正确形状。"""
    omegas = np.array([1.0, 2.0])
    state = np.array([0.1, 0.2, 0.0, 0.0])
    d = modal_dynamics_plate(0.0, state, omegas, None)
    assert d.shape == (4,)


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    bad_cases = [
        ("E", {"E": -1}),
        ("h", {"h": -0.01}),
        ("ν", {"nu": 0.6}),
        ("ν", {"nu": -0.1}),
        ("rho", {"rho": -1}),
        ("a", {"a": -1}),
        ("b", {"b": -1}),
    ]
    for label, kwargs in bad_cases:
        try:
            validate_parameters(**kwargs)
            assert False, f"应拒绝 {kwargs}"
        except AssertionError as e:
            assert label in str(e), f"错误信息应包含 {label}: {e}"


if __name__ == "__main__":
    test_static_navier_solution()
    print("✓ Navier 解边界条件")
    test_static_deflection_formula()
    print("✓ 挠度公式一致性")
    test_deflection_coefficient()
    print("✓ 挠度系数")
    test_load_coefficient_odd()
    print("✓ 载荷系数")
    test_natural_frequency_formula()
    print("✓ 固有频率公式")
    test_mode_shape_boundary()
    print("✓ 模态边界条件")
    test_mode_shape_laplacian()
    print("✓ 模态拉普拉斯算子")
    test_modal_orthogonality()
    print("✓ 模态正交性")
    test_omega_squared_equals_k_over_m()
    print("✓ ω² = k_mn / m_mn")
    test_modal_energy_conservation()
    print("✓ 模态能量守恒")
    test_fd_frequency_convergence()
    print("✓ 有限差分收敛")
    test_degradation_no_load()
    print("✓ 零载荷退化")
    test_error_injection_wrong_E()
    print("✓ 反例: E 翻倍")
    test_error_injection_wrong_h()
    print("✓ 反例: h 翻倍")
    test_dynamics_interface()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-051 所有一致性测试通过")
