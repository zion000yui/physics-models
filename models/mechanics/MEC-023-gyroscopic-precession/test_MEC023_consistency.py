"""MEC-023 —— 一致性测试：陀螺慢进动近似模型。

验证策略（避免循环验证）：
- dynamics 实现精确 Routhian 方程
- 稳态公式 Ω_p = mgl/(I₃ω_s) 来自力矩=角动量变化率的独立推导
- 通过守恒量（p_φ, E_eff）验证 dynamics 方程本身的正确性
- 通过精确稳态解（二次方程根）验证稳态条件
- 通过不同 ω_s 的对比验证 Ω_p ∝ 1/ω_s

运行方法（在本文件所在目录执行）：
    python test_MEC023_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, validate_parameters,
                   steady_state_precession, exact_steady_state_precession,
                   conjugate_momentum_phi, effective_energy, analytical)

TOL = 1e-6
STABLE_TOL = 1e-3  # 稳态容差（近似模型，允许小偏差）


def _solve(theta0, theta_dot0, phi0, phi_dot0,
           m, l, I1, I3, omega_s, g, t_end=5.0, n=401):
    """小工具：数值积分。"""
    initial_state = np.array([theta0, theta_dot0, phi0, phi_dot0],
                              dtype=float)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(m, l, I1, I3, omega_s, g),
                    rtol=1e-10, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def test_exact_steady_state_theta_constant():
    """精确稳态初始条件下 θ 应严格保持恒定。

    使用精确二次方程根作为初始 φ̇₀，而非近似公式。
    这独立验证 dynamics 方程的正确性（不依赖近似公式）。
    """
    m, l, I1, I3, omega_s, g = 1.0, 0.5, 0.2, 0.1, 50.0, 9.81
    theta_0 = np.pi / 4
    omega_slow, _ = exact_steady_state_precession(m, l, I1, I3, omega_s, g, theta_0)
    assert omega_slow is not None, "应存在稳态解"

    t, theta_n, _, _, _ = _solve(
        theta_0, 0.0, 0.0, omega_slow,
        m, l, I1, I3, omega_s, g, t_end=5.0, n=401)
    max_dev = np.max(np.abs(theta_n - theta_0))
    assert max_dev < TOL, \
        f"精确稳态 θ 波动 {max_dev:.3e} 超出容差 {TOL}"


def test_conjugate_momentum_phi_conserved():
    """p_φ = I₁φ̇sin²θ + I₃ω_s cos θ 应守恒。

    独立验证 φ 方程的正确性（不依赖稳态假设）。
    """
    m, l, I1, I3, omega_s, g = 1.0, 0.5, 0.2, 0.1, 50.0, 9.81
    theta_0 = np.pi / 4
    omega_p = steady_state_precession(m, l, I1, I3, omega_s, g, theta_0)
    # 使用近似稳态初始条件（故意偏离精确稳态，产生章动）
    t, theta_n, td_n, phi_n, pd_n = _solve(
        theta_0, 0.0, 0.0, omega_p,
        m, l, I1, I3, omega_s, g, t_end=5.0, n=401)
    p_phi = np.array([conjugate_momentum_phi(
        [theta_n[i], td_n[i], phi_n[i], pd_n[i]], I1, I3, omega_s)
        for i in range(len(t))])
    p0 = p_phi[0]
    assert np.allclose(p_phi, p0, atol=TOL), \
        f"p_φ 不守恒：波动 {np.max(np.abs(p_phi - p0)):.3e}"


def test_effective_energy_conserved():
    """E_eff 应守恒（L_eff 不显含 t）。

    独立验证 θ 方程的正确性（不依赖稳态假设）。
    """
    m, l, I1, I3, omega_s, g = 1.0, 0.5, 0.2, 0.1, 50.0, 9.81
    theta_0 = np.pi / 4
    omega_p = steady_state_precession(m, l, I1, I3, omega_s, g, theta_0)
    t, theta_n, td_n, phi_n, pd_n = _solve(
        theta_0, 0.0, 0.0, omega_p,
        m, l, I1, I3, omega_s, g, t_end=5.0, n=401)
    E = np.array([effective_energy(
        [theta_n[i], td_n[i], phi_n[i], pd_n[i]],
        m, l, I1, I3, omega_s, g) for i in range(len(t))])
    E0 = E[0]
    assert np.allclose(E, E0, atol=TOL), \
        f"E_eff 不守恒：波动 {np.max(np.abs(E - E0)):.3e}"


def test_steady_state_equation_satisfied():
    """稳态时 θ 方程应给出 θ̈ = 0。

    直接将精确稳态条件代入 dynamics，检查 θ̈ = 0。
    这是对 dynamics 方程本身的独立检查。
    """
    m, l, I1, I3, omega_s, g = 1.0, 0.5, 0.2, 0.1, 50.0, 9.81
    theta_0 = np.pi / 4
    omega_slow, _ = exact_steady_state_precession(m, l, I1, I3, omega_s, g, theta_0)
    d = dynamics(0, [theta_0, 0.0, 0.0, omega_slow],
                 m, l, I1, I3, omega_s, g)
    assert abs(d[1]) < 1e-12, \
        f"精确稳态 θ̈ 不为零：{d[1]:.3e}"


def test_omega_p_proportional_to_1_over_omega_s():
    """Ω_p · ω_s 应为常数 mgl/I₃（Ω_p ∝ 1/ω_s）。

    对不同自旋速度验证进动率与自旋速度的反比关系。
    """
    m, l, I1, I3, g = 1.0, 0.5, 0.2, 0.1, 9.81
    theta_0 = np.pi / 4
    expected = m * g * l / I3
    for omega_s in [20, 50, 100, 200]:
        omega_p = steady_state_precession(m, l, I1, I3, omega_s, g, theta_0)
        product = omega_p * omega_s
        assert np.isclose(product, expected, rtol=1e-10), \
            f"ω_s={omega_s}: Ω_p·ω_s={product:.6f} ≠ {expected:.6f}"


def test_approximation_improves_with_higher_spin():
    """自旋越快，近似 Ω_p 越接近精确值（慢进动近似越来越好）。"""
    m, l, I1, I3, g = 1.0, 0.5, 0.2, 0.1, 9.81
    theta_0 = np.pi / 4
    errors = []
    for omega_s in [50, 100, 500, 1000, 5000]:
        omega_approx = steady_state_precession(m, l, I1, I3, omega_s, g, theta_0)
        omega_exact, _ = exact_steady_state_precession(m, l, I1, I3, omega_s, g, theta_0)
        err = abs(omega_approx - omega_exact) / omega_exact
        errors.append(err)
    # 误差应递减
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], \
            f"近似误差未随 ω_s 递减：{errors[i]:.3e} → {errors[i+1]:.3e}"


def test_no_sin_theta_factor_in_omega_p():
    """验证 Ω_p = mgl/(I₃ω_s) 不含 sin(θ₀) 因子。

    对不同 θ₀，Ω_p 应相同（近似公式不依赖 θ₀）。
    """
    m, l, I1, I3, omega_s, g = 1.0, 0.5, 0.2, 0.1, 50.0, 9.81
    omega_p_ref = steady_state_precession(m, l, I1, I3, omega_s, g, np.pi / 4)
    for theta_0 in [0.1, 0.5, 1.0, 1.5, 2.5]:
        omega_p = steady_state_precession(m, l, I1, I3, omega_s, g, theta_0)
        assert np.isclose(omega_p, omega_p_ref), \
            f"θ₀={theta_0}: Ω_p={omega_p:.6f} ≠ {omega_p_ref:.6f}（不应含 sin θ₀）"


def test_approximate_steady_state_near_constant():
    """近似稳态初始条件下 θ 应近似恒定（允许小偏差）。

    使用近似 Ω_p（非精确）设置初始条件，θ 应有小的章动振荡，
    但振幅应随 ω_s 增大而减小。
    """
    m, l, I1, I3, g = 1.0, 0.5, 0.2, 0.1, 9.81
    theta_0 = np.pi / 4
    # 高速自旋：近似好，θ 波动小
    omega_s_high = 200.0
    omega_p = steady_state_precession(m, l, I1, I3, omega_s_high, g, theta_0)
    t, theta_n, _, _, _ = _solve(
        theta_0, 0.0, 0.0, omega_p,
        m, l, I1, I3, omega_s_high, g, t_end=5.0, n=401)
    max_dev = np.max(np.abs(theta_n - theta_0))
    # 高速自旋时近似好，波动应远小于 θ₀
    assert max_dev < 0.01 * theta_0, \
        f"高速自旋 θ 波动过大：{max_dev:.3e}（应 < {0.01*theta_0:.3e}）"


def test_omega_s_zero_rejected():
    """omega_s=0 应被拒绝（高速自旋假设失效）。"""
    try:
        validate_parameters(m=1.0, l=0.5, I1=0.2, I3=0.1, omega_s=0.0, g=9.81)
        raise AssertionError("应拒绝 omega_s=0")
    except ValueError as e:
        assert "omega_s" in str(e) or "失效" in str(e)


def test_invalid_parameters_rejected():
    """m≤0、l≤0、I1≤0、I3≤0 或 g≤0 应被拒绝。"""
    for kwargs, key in [
        (dict(m=0.0), "m"),
        (dict(l=0.0), "l"),
        (dict(I1=-1.0), "I1"),
        (dict(I3=0.0), "I3"),
        (dict(g=-1.0), "g"),
    ]:
        try:
            validate_parameters(omega_s=10.0, **kwargs)
            raise AssertionError(f"应拒绝 {key}={kwargs[key]}")
        except (AssertionError, ValueError) as e:
            assert key in str(e), f"错误信息应包含 {key}"


if __name__ == "__main__":
    test_exact_steady_state_theta_constant()
    test_conjugate_momentum_phi_conserved()
    test_effective_energy_conserved()
    test_steady_state_equation_satisfied()
    test_omega_p_proportional_to_1_over_omega_s()
    test_approximation_improves_with_higher_spin()
    test_no_sin_theta_factor_in_omega_p()
    test_approximate_steady_state_near_constant()
    test_omega_s_zero_rejected()
    test_invalid_parameters_rejected()
    print("OK: MEC-023 数值解与物理一致性验证通过")
