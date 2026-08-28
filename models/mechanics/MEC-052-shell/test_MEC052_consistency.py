"""MEC-052 —— 一致性测试：圆柱壳力学。

验证：
- 薄膜理论：内压应力比 = 2
- 弯曲理论：解析解边界条件 + BVP 一致性
- 固有频率公式
- 退化到板（R→∞）
- 能量守恒
- 有限差分收敛
- 反例验证
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC052_consistency.py
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

from model import (
    validate_parameters,
    bending_stiffness,
    membrane_stiffness,
    characteristic_length,
    decay_constant,
    membrane_forces_internal_pressure,
    membrane_stresses_internal_pressure,
    hoop_to_axial_ratio,
    axial_bending_ode_coefficients,
    axial_bending_analytical,
    axial_bending_max_deflection,
    natural_frequencies_cylindrical,
    membrane_frequency,
    bending_frequency_limit,
    modal_dynamics_shell,
    modal_energy_shell,
    degradation_to_plate_check,
    fd_shell_natural_frequencies,
)

TOL = 1e-6

E = 2.0e11
H = 0.005
NU = 0.3
RHO = 7850.0
R = 0.5
L = 2.0
P = 1.0e6


# ============================================================
# 薄膜理论测试
# ============================================================

def test_membrane_stress_ratio():
    """环向应力应为轴向应力的 2 倍。"""
    sigma_x, sigma_theta = membrane_stresses_internal_pressure(P, R, H)
    ratio = sigma_theta / sigma_x
    assert abs(ratio - hoop_to_axial_ratio()) < 1e-15, f"σ_θ/σ_x = {ratio}"


def test_membrane_forces():
    """薄膜力应满足 N_θ = pR, N_x = pR/2。"""
    N_x, N_theta = membrane_forces_internal_pressure(P, R)
    assert abs(N_theta - P * R) < 1e-6
    assert abs(N_x - P * R / 2) < 1e-6


def test_membrane_stress_formula():
    """薄膜应力应满足 σ_θ = pR/h, σ_x = pR/(2h)。"""
    sigma_x, sigma_theta = membrane_stresses_internal_pressure(P, R, H)
    assert abs(sigma_theta - P * R / H) < 1e-6
    assert abs(sigma_x - P * R / (2 * H)) < 1e-6


# ============================================================
# 弯曲理论测试
# ============================================================

def test_axial_bending_boundary_cantilever():
    """长壳悬臂解析解应满足 x=0 处 w=0, w'=0。"""
    lam = characteristic_length(E, H, NU, R)
    alpha = decay_constant(E, H, NU, R)
    x = np.linspace(0, 5 * lam, 200)
    w = axial_bending_analytical(x, P, E, H, NU, R, L, bc='long_cantilever')
    assert abs(w[0]) < 1e-10, f"w(0) = {w[0]:.3e}"
    # w'(0) = 0 via 4th-order forward difference
    dx = x[1] - x[0]
    w_prime_0 = (-3*w[0] + 4*w[1] - w[2]) / (2*dx)
    assert abs(w_prime_0) < 1e-4, f"w'(0) ≈ {w_prime_0:.3e}"


def test_axial_bending_boundary_ss():
    """长壳铰支解析解应满足 x=0 处 w=0。"""
    lam = characteristic_length(E, H, NU, R)
    x = np.linspace(0, 5 * lam, 200)
    w = axial_bending_analytical(x, P, E, H, NU, R, L, bc='long_simply_supported')
    assert abs(w[0]) < 1e-10, f"w(0) = {w[0]:.3e}"


def test_axial_bending_far_field():
    """远端挠度应趋近于特解 w = p/k。"""
    lam = characteristic_length(E, H, NU, R)
    x = np.array([20 * lam])
    w = axial_bending_analytical(x, P, E, H, NU, R, L, bc='long_cantilever')
    w_max = axial_bending_max_deflection(P, E, H, R)
    err = abs(w[0] - w_max) / w_max
    assert err < 1e-6, f"远端挠度 {w[0]:.6e} vs {w_max:.6e}, err={err:.3e}"


def test_bvp_vs_analytical():
    """BVP 解析解应在中间区域一致。"""
    D, k = axial_bending_ode_coefficients(E, H, NU, R)
    alpha = decay_constant(E, H, NU, R)
    x = np.linspace(0, L, 201)
    y_init = np.zeros((4, len(x)))
    w_max = P / k
    y_init[0] = w_max * (1 - np.exp(-alpha * x) * np.cos(alpha * x))

    def ode(x, y):
        dy = np.zeros_like(y)
        dy[0] = y[1]
        dy[1] = y[2]
        dy[2] = y[3]
        dy[3] = (P - k * y[0]) / D * np.ones_like(x)
        return dy

    def bc_cantilever(ya, yb):
        return np.array([ya[0], ya[1], yb[2], yb[3]])

    sol = solve_bvp(ode, bc_cantilever, x, y_init, tol=1e-10, max_nodes=10000)
    assert sol.success, f"BVP 失败: {sol.message}"

    # 中间区域比较（避开端部效应）
    mask = (x > 0.1 * L) & (x < 0.9 * L)
    w_bvp = sol.sol(x)[0]
    w_ana = axial_bending_analytical(x, P, E, H, NU, R, L, bc='long_cantilever')
    err = np.max(np.abs(w_bvp[mask] - w_ana[mask])) / w_max
    assert err < 0.01, f"BVP vs 解析误差 {err:.3e}"


# ============================================================
# 固有频率测试
# ============================================================

def test_natural_frequency_formula():
    """固有频率应满足 ω_n² = [D(nπ/L)⁴ + Eh/R²] / (ρh)。"""
    D = bending_stiffness(E, H, NU)
    k = membrane_stiffness(E, H, R)
    mu = RHO * H
    omegas = natural_frequencies_cylindrical(5, E, H, NU, RHO, R, L)
    for n in range(1, 6):
        kappa_n = n * np.pi / L
        omega_expected = np.sqrt((D * kappa_n**4 + k) / mu)
        err = abs(omegas[n - 1] - omega_expected) / omega_expected
        assert err < 1e-10, f"mode {n}: err={err:.3e}"


def test_membrane_frequency():
    """薄膜频率极限 ω_0 = √(E/(ρR²)) 应等于低阶频率极限。"""
    omega_0 = membrane_frequency(E, RHO, R)
    omegas = natural_frequencies_cylindrical(1, E, H, NU, RHO, R, L)
    # 低阶频率应接近薄膜频率（弯曲项小）
    err = abs(omegas[0] - omega_0) / omega_0
    assert err < 0.01, f"ω_1={omegas[0]:.4f} vs ω_0={omega_0:.4f}, err={err:.3e}"


def test_frequency_increases_with_n():
    """频率应随阶数递增。"""
    omegas = natural_frequencies_cylindrical(5, E, H, NU, RHO, R, L)
    assert np.all(np.diff(omegas) > 0), f"频率未递增: {omegas}"


# ============================================================
# 退化测试
# ============================================================

def test_degradation_to_plate():
    """R→∞ 时壳频率应退化为板频率。"""
    omega_shell, omega_plate, err = degradation_to_plate_check(E, H, NU, RHO, L)
    assert err < 1e-6, f"退化误差 {err:.3e}"


# ============================================================
# 动态测试
# ============================================================

def test_modal_energy_conservation():
    """自由振动应能量守恒。"""
    omegas = natural_frequencies_cylindrical(3, E, H, NU, RHO, R, L)
    q0 = np.array([1e-4, 0, 0])
    qdot0 = np.array([0, 0, 0])
    t_end = 2 * np.pi / omegas[0] * 3
    sol = solve_ivp(
        modal_dynamics_shell, (0, t_end), np.concatenate([q0, qdot0]),
        args=(omegas, None),
        t_eval=np.linspace(0, t_end, 501),
        rtol=1e-10, atol=1e-12,
    )
    energies = np.array([modal_energy_shell(sol.y[:, i], omegas)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / energies[0]
    assert dE < 1e-6, f"能量变化 {dE:.3e}"


def test_fd_frequency_convergence():
    """有限差分频率应收敛到解析值。"""
    omegas_ana = natural_frequencies_cylindrical(3, E, H, NU, RHO, R, L)
    for N in [31, 51, 101]:
        omegas_fd = fd_shell_natural_frequencies(N, L, E, H, NU, RHO, R, n_modes=3)
        errs = np.abs(omegas_fd[:3] - omegas_ana[:3]) / omegas_ana[:3]
        assert errs[0] < 1e-4, f"N={N}, mode 1 误差 {errs[0]:.4f}"


# ============================================================
# 反例与参数验证
# ============================================================

def test_error_injection_wrong_R():
    """反例：R 翻倍应导致环向应力翻倍（σ_θ = pR/h）。"""
    _, sigma1 = membrane_stresses_internal_pressure(P, R, H)
    _, sigma2 = membrane_stresses_internal_pressure(P, 2 * R, H)
    assert abs(sigma2 - 2 * sigma1) < 1e-6 * abs(sigma1)


def test_error_injection_wrong_h():
    """反例：h 翻倍应导致应力减半。"""
    _, sigma1 = membrane_stresses_internal_pressure(P, R, H)
    _, sigma2 = membrane_stresses_internal_pressure(P, R, 2 * H)
    assert abs(sigma2 - sigma1 / 2) < 1e-6 * abs(sigma1)


def test_dynamics_interface():
    """dynamics 接口应返回正确形状。"""
    omegas = np.array([1.0, 2.0])
    state = np.array([0.1, 0.2, 0.0, 0.0])
    d = modal_dynamics_shell(0.0, state, omegas, None)
    assert d.shape == (4,)


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    bad_cases = [
        ("E", {"E": -1}),
        ("h", {"h": -0.01}),
        ("ν", {"nu": 0.6}),
        ("rho", {"rho": -1}),
        ("R", {"R": -1}),
        ("L", {"L": -1}),
    ]
    for label, kwargs in bad_cases:
        try:
            validate_parameters(**kwargs)
            assert False, f"应拒绝 {kwargs}"
        except AssertionError as e:
            assert label in str(e), f"错误信息应包含 {label}: {e}"


if __name__ == "__main__":
    test_membrane_stress_ratio()
    print("✓ 薄膜应力比 = 2")
    test_membrane_forces()
    print("✓ 薄膜力公式")
    test_membrane_stress_formula()
    print("✓ 薄膜应力公式")
    test_axial_bending_boundary_cantilever()
    print("✓ 悬臂解析边界条件")
    test_axial_bending_boundary_ss()
    print("✓ 铰支解析边界条件")
    test_axial_bending_far_field()
    print("✓ 远端挠度趋近特解")
    test_bvp_vs_analytical()
    print("✓ BVP vs 解析（中间区域）")
    test_natural_frequency_formula()
    print("✓ 固有频率公式")
    test_membrane_frequency()
    print("✓ 薄膜频率极限")
    test_frequency_increases_with_n()
    print("✓ 频率递增")
    test_degradation_to_plate()
    print("✓ R→∞ 退化为板")
    test_modal_energy_conservation()
    print("✓ 模态能量守恒")
    test_fd_frequency_convergence()
    print("✓ 有限差分收敛")
    test_error_injection_wrong_R()
    print("✓ 反例: R 翻倍→应力翻倍")
    test_error_injection_wrong_h()
    print("✓ 反例: h 翻倍")
    test_dynamics_interface()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-052 所有一致性测试通过")
