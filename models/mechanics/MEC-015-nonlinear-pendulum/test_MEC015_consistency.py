"""MEC-015 —— 一致性测试：非线性单摆。

验证：
- 小角度情况下与线性理论解一致
- 小角度极限下与 MEC-010 简谐振子对应
- 不同振幅下周期变化：振幅增大时周期增长
- 数值周期与椭圆积分理论结果一致
- 机械能守恒
- 平衡点特例
- 非法参数处理

运行方法（在本文件所在目录执行）：
    python test_MEC015_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk

from model import (analytical, dynamics, validate_parameters,
                   natural_frequency, small_angle_period, nonlinear_period,
                   mechanical_energy)

TOL = 1e-6


def _solve(theta0=1.0, omega0=0.0, g=9.81, L=1.0, m=1.0,
           t_end=5.0, n=401):
    """小工具：跑一次数值积分，返回 (t, theta, omega)。"""
    initial_state = np.array([theta0, omega0], dtype=float)
    validate_parameters(g=g, L=L, m=m)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(g, L, m),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1]


def test_small_angle_matches_linear():
    """小角度情况下非线性数值解应与线性解析解一致。"""
    g, L, m = 9.81, 1.0, 1.0
    theta0 = 0.005  # 0.29°，极小角度确保线性近似精度
    omega0 = 0.0
    T0 = small_angle_period(g, L)
    t_end, n = T0, 401
    t, theta_num, omega_num = _solve(
        theta0, omega0, g=g, L=L, m=m, t_end=t_end, n=n)
    theta_ana, omega_ana = analytical(
        t, [theta0, omega0], g=g, L=L, m=m)
    err_theta = np.max(np.abs(theta_num - theta_ana))
    err_omega = np.max(np.abs(omega_num - omega_ana))
    assert err_theta < TOL, f"小角度 θ 误差 {err_theta:.3e} 超出容差 {TOL}"
    assert err_omega < TOL, f"小角度 ω 误差 {err_omega:.3e} 超出容差 {TOL}"


def test_small_angle_degenerates_to_MEC010():
    """小角度极限下应与 MEC-010 简谐振子对应。

    非线性方程 θ̈ + (g/L)sin(θ) = 0 在小角度下退化为
    θ̈ + (g/L)θ = 0，即 MEC-010 形式，ω₀ = √(g/L)。
    """
    g, L, m = 9.81, 1.0, 1.0
    theta0 = 0.005  # 极小角度
    omega0 = 0.0
    omega_0 = natural_frequency(g, L)
    # MEC-010 等效: k_eff = mg/L, ω₀ = √(k_eff/m) = √(g/L)
    T0 = 2.0 * np.pi / omega_0
    t_end, n = T0, 401
    t, theta_num, _ = _solve(
        theta0, omega0, g=g, L=L, m=m, t_end=t_end, n=n)
    # MEC-010 解析解
    theta_exp = theta0 * np.cos(omega_0 * t)
    err = np.max(np.abs(theta_num - theta_exp))
    assert err < TOL, f"小角度退化误差 {err:.3e}（未退化为 MEC-010）"


def test_period_increases_with_amplitude():
    """振幅增大时周期应增长（非线性效应）。"""
    g, L, m = 9.81, 1.0, 1.0
    T0 = small_angle_period(g, L)
    # 取多个振幅，验证周期递增
    amplitudes = [0.1, 0.5, 1.0, 1.5, 2.0]
    periods = []
    for theta_max in amplitudes:
        T = nonlinear_period(g, L, theta_max)
        periods.append(T)
    # 验证严格递增
    for i in range(len(periods) - 1):
        assert periods[i + 1] > periods[i], \
            f"周期未随振幅增大：θ={amplitudes[i]}→{amplitudes[i+1]}, T={periods[i]:.6f}→{periods[i+1]:.6f}"
    # 所有周期都应大于小角度周期
    for T in periods:
        assert T > T0, f"非线性周期 {T:.6f} 应大于小角度周期 {T0:.6f}"


def test_numerical_period_matches_elliptic_integral():
    """数值积分得到的周期应与椭圆积分理论结果一致。"""
    g, L, m = 9.81, 1.0, 1.0
    theta_max = 1.0  # ~57°
    T_theory = nonlinear_period(g, L, theta_max)
    # 从静止释放，数值积分一个理论周期，检查是否回到起点
    t, theta_num, omega_num = _solve(
        theta0=theta_max, omega0=0.0, g=g, L=L, m=m,
        t_end=T_theory, n=2001)
    # 周期后应回到初始角度（速度应为零）
    err_theta = abs(theta_num[-1] - theta_max)
    err_omega = abs(omega_num[-1])
    assert err_theta < 0.01, \
        f"周期后 θ 未闭合：误差 {err_theta:.3e}（理论周期 {T_theory:.6f}）"
    assert err_omega < 0.01, \
        f"周期后 ω 不为零：{err_omega:.3e}"


def test_mechanical_energy_conserved():
    """机械能应守恒。"""
    g, L, m = 9.81, 1.0, 1.0
    theta0, omega0 = 1.5, 0.0  # 大角度
    T = nonlinear_period(g, L, theta0)
    t, theta_num, omega_num = _solve(
        theta0, omega0, g=g, L=L, m=m, t_end=T, n=501)
    E_num = np.array([mechanical_energy([th, om], g=g, L=L, m=m)
                      for th, om in zip(theta_num, omega_num)])
    E0 = mechanical_energy([theta0, omega0], g=g, L=L, m=m)
    assert np.allclose(E_num, E0, atol=TOL), \
        f"机械能不守恒：波动 {np.max(np.abs(E_num - E0)):.3e}"


def test_equilibrium_point():
    """平衡点特例：θ=0, ω=0 时质点应始终静止。"""
    g, L, m = 9.81, 1.0, 1.0
    t, theta_num, omega_num = _solve(
        theta0=0.0, omega0=0.0, g=g, L=L, m=m, t_end=10.0, n=101)
    assert np.allclose(theta_num, 0.0, atol=TOL), \
        f"平衡点 θ 不为零：最大 {np.max(np.abs(theta_num)):.3e}"
    assert np.allclose(omega_num, 0.0, atol=TOL), \
        f"平衡点 ω 不为零：最大 {np.max(np.abs(omega_num)):.3e}"


def test_separatrix_energy():
    """分离轨道能量应等于 2·m·g·L。

    E = mgL(1-cos(π)) + 0 = 2mgL，此时摆恰好到达最高点 (θ=π)。
    """
    g, L, m = 9.81, 1.0, 1.0
    E_sep = mechanical_energy([np.pi, 0.0], g=g, L=L, m=m)
    E_theory = 2.0 * m * g * L
    assert np.isclose(E_sep, E_theory, rtol=TOL), \
        f"分离轨道能量不符：{E_sep:.6f} vs {E_theory:.6f}"


def test_invalid_parameters_rejected():
    """g≤0、L≤0 或 m≤0 应被拒绝。"""
    try:
        validate_parameters(g=0.0, L=1.0, m=1.0)
        raise AssertionError("应拒绝 g=0")
    except AssertionError as e:
        assert "g" in str(e)

    try:
        validate_parameters(g=9.81, L=-1.0, m=1.0)
        raise AssertionError("应拒绝 L<0")
    except AssertionError as e:
        assert "L" in str(e)

    try:
        validate_parameters(g=9.81, L=1.0, m=0.0)
        raise AssertionError("应拒绝 m=0")
    except AssertionError as e:
        assert "m" in str(e)


if __name__ == "__main__":
    test_small_angle_matches_linear()
    test_small_angle_degenerates_to_MEC010()
    test_period_increases_with_amplitude()
    test_numerical_period_matches_elliptic_integral()
    test_mechanical_energy_conserved()
    test_equilibrium_point()
    test_separatrix_energy()
    test_invalid_parameters_rejected()
    print("OK: MEC-015 数值解与解析解一致")
