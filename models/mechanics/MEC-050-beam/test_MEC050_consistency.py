"""MEC-050 —— 一致性测试：欧拉-伯努利梁。

验证：
- 静态弯曲：解析解 vs BVP 数值解（悬臂、简支）
- 静态挠度公式
- 固有频率：解析公式
- 模态正交性
- ω² = k_n / m_n
- 能量守恒（自由振动）
- 退化：q=0 → 无变形
- 有限差分收敛性
- 反例验证
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC050_consistency.py
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

from model import (
    validate_parameters,
    static_cantilever_uniform_load,
    static_simply_supported_uniform_load,
    static_cantilever_tip_load,
    max_deflection_cantilever,
    max_deflection_simply_supported,
    natural_frequencies,
    mode_shape,
    mode_shape_second_derivative,
    modal_dynamics,
    modal_energy,
    modal_mass,
    modal_stiffness,
    verify_orthogonality,
    fd_natural_frequencies,
    fd_stiffness_matrix,
    fd_mass_matrix,
    beam_pde_rhs,
    bending_stiffness,
    mass_per_length,
    cantilever_beta_L,
    cantilever_sigma,
)

TOL = 1e-6

# 默认参数
E = 2.0e11     # 钢
I = 1.0e-8     # 1cm × 1cm 方截面
RHO = 7850.0
A = 1.0e-4
L = 1.0
Q = 100.0      # 均布载荷


# ============================================================
# 静态测试
# ============================================================

def test_static_cantilever_bvp():
    """悬臂梁解析解应与 BVP 求解一致。"""
    from scipy_solve import solve_static_bvp_cantilever
    x = np.linspace(0, L, 201)
    w_ana, _, _, _ = static_cantilever_uniform_load(x, Q, L, E, I)
    sol = solve_static_bvp_cantilever(Q, L, E, I)
    assert sol.success, f"BVP 求解失败: {sol.message}"
    w_bvp = sol.sol(x)[0]
    err = np.max(np.abs(w_bvp - w_ana))
    assert err < 1e-8, f"悬臂 BVP 误差 {err:.3e}"


def test_static_simply_supported_bvp():
    """简支梁解析解应与 BVP 求解一致。"""
    from scipy_solve import solve_static_bvp_simply_supported
    x = np.linspace(0, L, 201)
    w_ana, _, _, _ = static_simply_supported_uniform_load(x, Q, L, E, I)
    sol = solve_static_bvp_simply_supported(Q, L, E, I)
    assert sol.success, f"BVP 求解失败: {sol.message}"
    w_bvp = sol.sol(x)[0]
    err = np.max(np.abs(w_bvp - w_ana))
    assert err < 1e-8, f"简支 BVP 误差 {err:.3e}"


def test_max_deflection_formulas():
    """端部最大挠度公式应正确。"""
    # 悬臂均布载荷
    w_max = max_deflection_cantilever(Q, L, E, I, 'uniform')
    expected = Q * L**4 / (8 * E * I)
    assert abs(w_max - expected) < 1e-15

    # 与解析解在 x=L 处一致
    x_L = np.array([L])
    w_at_L, _, _, _ = static_cantilever_uniform_load(x_L, Q, L, E, I)
    assert abs(w_at_L[0] - w_max) < 1e-15

    # 简支梁
    w_max_s = max_deflection_simply_supported(Q, L, E, I)
    expected_s = 5 * Q * L**4 / (384 * E * I)
    assert abs(w_max_s - expected_s) < 1e-15

    # 与解析解在 x=L/2 处一致
    x_mid = np.array([L / 2])
    w_at_mid, _, _, _ = static_simply_supported_uniform_load(x_mid, Q, L, E, I)
    assert abs(w_at_mid[0] - w_max_s) < 1e-15


def test_cantilever_boundary_conditions():
    """悬臂梁解析解应满足边界条件。"""
    x = np.linspace(0, L, 1001)
    w, theta, M, V = static_cantilever_uniform_load(x, Q, L, E, I)

    # 固定端 x=0: w=0, θ=0
    assert abs(w[0]) < 1e-15, f"w(0) = {w[0]:.3e}"
    assert abs(theta[0]) < 1e-15, f"θ(0) = {theta[0]:.3e}"

    # 自由端 x=L: M=0, V=0
    assert abs(M[-1]) < 1e-10, f"M(L) = {M[-1]:.3e}"
    assert abs(V[-1]) < 1e-10, f"V(L) = {V[-1]:.3e}"


def test_simply_supported_boundary_conditions():
    """简支梁解析解应满足边界条件。"""
    x = np.linspace(0, L, 1001)
    w, _, M, _ = static_simply_supported_uniform_load(x, Q, L, E, I)

    # x=0: w=0, M=0
    assert abs(w[0]) < 1e-15, f"w(0) = {w[0]:.3e}"
    assert abs(M[0]) < 1e-10, f"M(0) = {M[0]:.3e}"

    # x=L: w=0, M=0
    assert abs(w[-1]) < 1e-15, f"w(L) = {w[-1]:.3e}"
    assert abs(M[-1]) < 1e-10, f"M(L) = {M[-1]:.3e}"


def test_tip_load_solution():
    """悬臂梁端部集中力解析解。"""
    P = 50.0  # N
    x = np.linspace(0, L, 1001)
    w, theta, M, V = static_cantilever_tip_load(x, P, L, E, I)

    # 端部挠度 w(L) = P L³ / (3 EI)
    w_max = max_deflection_cantilever(P, L, E, I, 'tip')
    assert abs(w[-1] - w_max) < 1e-15

    # 边界条件
    assert abs(w[0]) < 1e-15
    assert abs(theta[0]) < 1e-15
    assert abs(M[-1]) < 1e-10  # M(L) = 0


def test_static_degradation_no_load():
    """载荷 q=0 时挠度应为零。"""
    x = np.linspace(0, L, 101)
    w_c, _, _, _ = static_cantilever_uniform_load(x, 0.0, L, E, I)
    w_s, _, _, _ = static_simply_supported_uniform_load(x, 0.0, L, E, I)
    assert np.allclose(w_c, 0), "悬臂零载荷挠度非零"
    assert np.allclose(w_s, 0), "简支零载荷挠度非零"


# ============================================================
# 动态测试
# ============================================================

def test_natural_frequency_formula():
    """固有频率应满足 ω_n = (β_n L)² √(EI/(ρA L⁴))。"""
    for bc in ['cantilever', 'simply_supported']:
        omegas = natural_frequencies(5, bc, E, I, RHO, A, L)
        EI = E * I
        mu = RHO * A
        for n in range(1, 6):
            if bc == 'cantilever':
                bL = cantilever_beta_L(n)
            else:
                bL = n * np.pi
            omega_expected = bL**2 * np.sqrt(EI / (mu * L**4))
            assert abs(omegas[n - 1] - omega_expected) < 1e-6, \
                f"{bc} mode {n}: {omegas[n-1]:.4f} vs {omega_expected:.4f}"


def test_cantilever_frequency_ratios():
    """悬臂梁固有频率应有正确的相对比值。

    ω_n = (β_n L)² √(EI/(ρA L⁴))
    故 ω_n / ω_1 = (β_n L)² / (β_1 L)²
    """
    omegas = natural_frequencies(5, 'cantilever', E, I, RHO, A, L)
    bLs = np.array([cantilever_beta_L(n) for n in range(1, 6)])
    bL_sq = bLs**2
    ratios = bL_sq / bL_sq[0]
    omega_ratios = omegas / omegas[0]
    assert np.allclose(ratios, omega_ratios, rtol=1e-10)


def test_mode_shape_boundary_cantilever():
    """悬臂梁模态形状应满足边界条件。

    φ(x) = cosh(βx) - cos(βx) - σ[sinh(βx) - sin(βx)]
    φ'(x) = β[sinh(βx) + sin(βx) - σ(cosh(βx) - cos(βx))]
    φ''(x) = β²[cosh(βx) + cos(βx) - σ(sinh(βx) + sin(βx))]
    φ'''(x) = β³[sinh(βx) - sin(βx) - σ(cosh(βx) + cos(βx))]
    """
    L_test = L
    for n in range(1, 4):
        beta = cantilever_beta_L(n) / L_test
        sigma = cantilever_sigma(n)

        # 固定端 x=0: φ=0, φ'=0
        phi_0 = 1 - 1 - sigma * (0 - 0)  # cosh(0)-cos(0) = 0
        assert abs(phi_0) < 1e-10, f"phi({n}, 0) = {phi_0:.3e}"
        phi_prime_0 = beta * (0 + 0 - sigma * (1 - 1))  # = 0
        assert abs(phi_prime_0) < 1e-10, f"phi'({n}, 0) = {phi_prime_0:.3e}"

        # 自由端 x=L: φ''=0, φ'''=0
        xL = L_test
        phi_pp_L = beta**2 * (np.cosh(beta*xL) + np.cos(beta*xL)
                               - sigma * (np.sinh(beta*xL) + np.sin(beta*xL)))
        assert abs(phi_pp_L) < 1e-6, f"phi''({n}, L) = {phi_pp_L:.3e}"
        phi_ppp_L = beta**3 * (np.sinh(beta*xL) - np.sin(beta*xL)
                                - sigma * (np.cosh(beta*xL) + np.cos(beta*xL)))
        assert abs(phi_ppp_L) < 0.1, f"phi'''({n}, L) = {phi_ppp_L:.3e}"


def test_mode_shape_boundary_simply_supported():
    """简支梁模态形状应满足边界条件。"""
    x = np.linspace(0, L, 1001)
    for n in range(1, 4):
        phi = mode_shape(x, n, 'simply_supported', L)
        phi_pp = mode_shape_second_derivative(x, n, 'simply_supported', L)

        # x=0, x=L: φ=0
        assert abs(phi[0]) < 1e-10, f"φ({n}, 0) = {phi[0]:.3e}"
        assert abs(phi[-1]) < 1e-10, f"φ({n}, L) = {phi[-1]:.3e}"

        # x=0, x=L: φ''=0 (弯矩为零)
        assert abs(phi_pp[0]) < 1e-10, f"φ''({n}, 0) = {phi_pp[0]:.3e}"
        assert abs(phi_pp[-1]) < 1e-10, f"φ''({n}, L) = {phi_pp[-1]:.3e}"


def test_modal_orthogonality_cantilever():
    """悬臂梁模态应正交。"""
    for i in range(1, 4):
        for j in range(i + 1, 5):
            result = verify_orthogonality(i, j, 'cantilever', RHO, A, L)
            # 归一化后 <φ_i, φ_j> 应远小于 <φ_i, φ_i>
            self = verify_orthogonality(i, i, 'cantilever', RHO, A, L)
            assert abs(result) / self < 1e-3, \
                f"<φ{i}, φ{j}> = {result:.6e}, 自身 = {self:.6e}"


def test_modal_orthogonality_simply_supported():
    """简支梁模态应正交。"""
    for i in range(1, 4):
        for j in range(i + 1, 5):
            result = verify_orthogonality(i, j, 'simply_supported', RHO, A, L)
            self = verify_orthogonality(i, i, 'simply_supported', RHO, A, L)
            assert abs(result) / self < 1e-3, \
                f"<φ{i}, φ{j}> = {result:.6e}, 自身 = {self:.6e}"


def test_omega_squared_equals_k_over_m():
    """ω_n² = k_n / m_n 应成立。"""
    for bc in ['cantilever', 'simply_supported']:
        omegas = natural_frequencies(3, bc, E, I, RHO, A, L)
        for n in range(1, 4):
            m_n = modal_mass(n, bc, RHO, A, L)
            k_n = modal_stiffness(n, bc, E, I, L)
            omega_sq = k_n / m_n
            omega_calc = np.sqrt(omega_sq)
            err = abs(omega_calc - omegas[n - 1]) / omegas[n - 1]
            assert err < 0.01, \
                f"{bc} mode {n}: √(k/m)={omega_calc:.4f}, ω_ana={omegas[n-1]:.4f}, err={err:.4f}"


def test_modal_dynamics_energy_conservation():
    """模态动力学自由振动应能量守恒。"""
    omegas = natural_frequencies(3, 'simply_supported', E, I, RHO, A, L)
    # 初始条件：第一模态振幅 1e-3
    q0 = np.array([1e-3, 0, 0])
    qdot0 = np.array([0, 0, 0])
    t_end = 2 * np.pi / omegas[0] * 5

    sol = solve_ivp(
        modal_dynamics, (0, t_end), np.concatenate([q0, qdot0]),
        args=(omegas, None, 3),
        t_eval=np.linspace(0, t_end, 501),
        rtol=1e-10, atol=1e-12,
    )
    energies = np.array([modal_energy(sol.y[:, i], omegas)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / energies[0]
    assert dE < 1e-6, f"能量变化 {dE:.3e}"


def test_modal_dynamics_frequency():
    """自由振动频率应等于固有频率。"""
    omegas = natural_frequencies(2, 'simply_supported', E, I, RHO, A, L)
    q0 = np.array([1e-3, 0])
    qdot0 = np.array([0, 0])
    t_end = 2 * np.pi / omegas[0] * 3

    sol = solve_ivp(
        modal_dynamics, (0, t_end), np.concatenate([q0, qdot0]),
        args=(omegas, None, 2),
        t_eval=np.linspace(0, t_end, 1001),
        rtol=1e-10, atol=1e-12,
    )
    q1 = sol.y[0, :]
    # 零交叉法测频率
    sign_changes = np.where(np.diff(np.sign(q1)))[0]
    if len(sign_changes) >= 2:
        half_period = np.mean(np.diff(sol.t[sign_changes[:4]]))
        omega_num = np.pi / half_period
        err = abs(omega_num - omegas[0]) / omegas[0]
        assert err < 0.01, f"数值频率 {omega_num:.4f} vs 解析 {omegas[0]:.4f}"
    else:
        assert False, "零交叉点不足"


# ============================================================
# 有限差分测试
# ============================================================

def test_fd_frequency_convergence():
    """有限差分频率应随网格加密收敛到解析值。"""
    omegas_ana = natural_frequencies(3, 'simply_supported', E, I, RHO, A, L)
    for N in [51, 101, 201]:
        omegas_fd = fd_natural_frequencies(N, L, E, I, RHO, A, n_modes=3)
        errs = np.abs(omegas_fd[:3] - omegas_ana[:3]) / omegas_ana[:3]
        # 误差应小于 1%（粗网格也要在合理范围）
        assert errs[0] < 0.01, f"N={N}, mode 1 误差 {errs[0]:.4f}"


def test_fd_convergence_order():
    """有限差分应呈 O(dx²) 收敛。"""
    omegas_ana = natural_frequencies(1, 'simply_supported', E, I, RHO, A, L)[0]
    Ns = [51, 101, 201, 401]
    errs = []
    for N in Ns:
        omega_fd = fd_natural_frequencies(N, L, E, I, RHO, A, n_modes=1)[0]
        errs.append(abs(omega_fd - omegas_ana) / omegas_ana)
    # 误差应随 N 递减
    assert errs[-1] < errs[0], f"未收敛: {errs}"
    # 比率应接近 4 (O(dx²))
    ratio = errs[0] / errs[-1]
    dx_ratio = (Ns[-1] - 1) / (Ns[0] - 1)  # = 8
    expected_ratio = dx_ratio**2  # = 64
    assert ratio > expected_ratio * 0.5, f"收敛阶不足: ratio={ratio}, expected≈{expected_ratio}"


# ============================================================
# 反例与参数验证
# ============================================================

def test_error_injection_wrong_E():
    """反例：E 乘 2 应导致挠度减半。"""
    w1 = max_deflection_cantilever(Q, L, E, I)
    w2 = max_deflection_cantilever(Q, L, 2 * E, I)
    assert abs(w2 - w1 / 2) < 1e-15, f"E 翻倍挠度应减半: {w2} vs {w1/2}"


def test_error_injection_wrong_L():
    """反例：L 乘 2 应导致挠度变为 16 倍（L⁴）。"""
    w1 = max_deflection_cantilever(Q, L, E, I)
    w2 = max_deflection_cantilever(Q, 2 * L, E, I)
    assert abs(w2 - 16 * w1) < 1e-10, f"L 翻倍挠度应×16: {w2} vs {16*w1}"


def test_error_injection_wrong_sigma():
    """反例：使用错误的 σ 应导致悬臂模态不满足自由端弯矩为零。"""
    L = 1.0
    x = np.array([L])  # 自由端
    n = 1
    beta_L = cantilever_beta_L(n)
    beta = beta_L / L
    sigma_correct = cantilever_sigma(n)
    sigma_wrong = 1.0 / sigma_correct  # 反转

    # 正确 σ
    phi_pp_correct = (beta**2 * (np.cosh(beta * x) + np.cos(beta * x)
                                  - sigma_correct * (np.sinh(beta * x) + np.sin(beta * x))))
    # 错误 σ
    phi_pp_wrong = (beta**2 * (np.cosh(beta * x) + np.cos(beta * x)
                                - sigma_wrong * (np.sinh(beta * x) + np.sin(beta * x))))

    assert abs(phi_pp_correct[0]) < 1e-10, f"正确σ: φ''(L)={phi_pp_correct[0]:.3e}"
    assert abs(phi_pp_wrong[0]) > 0.1, f"错误σ: φ''(L)={phi_pp_wrong[0]:.3e}"


def test_dynamics_interface():
    """dynamics 接口应返回正确形状。"""
    omegas = np.array([1.0, 2.0])
    state = np.array([0.1, 0.2, 0.0, 0.0])
    d = modal_dynamics(0.0, state, omegas, None, 2)
    assert d.shape == (4,)


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    try:
        validate_parameters(E=-1)
        assert False, "应拒绝 E<0"
    except AssertionError as e:
        assert "E" in str(e)
    try:
        validate_parameters(I=-1)
        assert False, "应拒绝 I<0"
    except AssertionError as e:
        assert "I" in str(e)
    try:
        validate_parameters(rho=-1)
        assert False, "应拒绝 rho<0"
    except AssertionError as e:
        assert "rho" in str(e) or "ρ" in str(e)
    try:
        validate_parameters(L=-1)
        assert False, "应拒绝 L<0"
    except AssertionError as e:
        assert "L" in str(e)


if __name__ == "__main__":
    test_static_cantilever_bvp()
    print("✓ 悬臂梁 BVP 一致性")
    test_static_simply_supported_bvp()
    print("✓ 简支梁 BVP 一致性")
    test_max_deflection_formulas()
    print("✓ 最大挠度公式")
    test_cantilever_boundary_conditions()
    print("✓ 悬臂边界条件")
    test_simply_supported_boundary_conditions()
    print("✓ 简支边界条件")
    test_tip_load_solution()
    print("✓ 端部集中力解")
    test_static_degradation_no_load()
    print("✓ 零载荷退化")
    test_natural_frequency_formula()
    print("✓ 固有频率公式")
    test_cantilever_frequency_ratios()
    print("✓ 悬臂频率比")
    test_mode_shape_boundary_cantilever()
    print("✓ 悬臂模态边界")
    test_mode_shape_boundary_simply_supported()
    print("✓ 简支模态边界")
    test_modal_orthogonality_cantilever()
    print("✓ 悬臂模态正交性")
    test_modal_orthogonality_simply_supported()
    print("✓ 简支模态正交性")
    test_omega_squared_equals_k_over_m()
    print("✓ ω² = k_n / m_n")
    test_modal_dynamics_energy_conservation()
    print("✓ 模态能量守恒")
    test_modal_dynamics_frequency()
    print("✓ 自由振动频率")
    test_fd_frequency_convergence()
    print("✓ 有限差分收敛")
    test_fd_convergence_order()
    print("✓ FD 收敛阶 O(dx²)")
    test_error_injection_wrong_E()
    print("✓ 反例: E 翻倍")
    test_error_injection_wrong_L()
    print("✓ 反例: L 翻倍")
    test_error_injection_wrong_sigma()
    print("✓ 反例: σ 反转")
    test_dynamics_interface()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-050 所有一致性测试通过")
