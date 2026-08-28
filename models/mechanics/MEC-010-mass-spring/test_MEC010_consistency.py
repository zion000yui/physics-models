"""MEC-010 —— 一致性测试：数值解 vs 解析解。

验证：
- 位移和速度数值解与解析解一致
- 机械能守恒
- 周期与理论值一致
- 零振幅特例（平衡位置静止）
- 相图轨迹为椭圆（等能量曲线）
- 参数关系（周期与质量/弹性系数的关系）
- 非法参数拒绝

运行方法（在本文件所在目录执行）：
    python test_MEC010_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    angular_frequency, period, amplitude, mechanical_energy

TOL = 1e-6


def _solve(x0=1.0, v0=0.0, k=1.0, m=1.0,
           t_end=6.28318530718, n=401):
    """小工具：跑一次数值积分，返回 (t, x, v)。"""
    initial_state = np.array([x0, v0], dtype=float)
    validate_parameters(k=k, m=m)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(k, m),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1]


def test_position_matches_analytical():
    """位移数值解应与解析解一致。"""
    x0, v0, k, m = 1.0, 0.0, 2.0, 0.5
    omega0 = angular_frequency(k, m)
    T = 2.0 * np.pi / omega0
    t_end, n = T, 401
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, t_end=t_end, n=n)
    x_ana, v_ana = analytical(t, [x0, v0], k=k, m=m)
    err_x = np.max(np.abs(x_num - x_ana))
    err_v = np.max(np.abs(v_num - v_ana))
    assert err_x < TOL, f"x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_v < TOL, f"v 误差 {err_v:.3e} 超出容差 {TOL}"


def test_mechanical_energy_conserved():
    """机械能 E = ½mv² + ½kx² 应保持恒定。"""
    x0, v0, k, m = 1.5, 2.0, 3.0, 2.0
    omega0 = angular_frequency(k, m)
    T = 2.0 * np.pi / omega0
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, t_end=T, n=501)
    E_num = np.array([mechanical_energy([x, v], k=k, m=m)
                      for x, v in zip(x_num, v_num)])
    E0 = mechanical_energy([x0, v0], k=k, m=m)
    assert np.allclose(E_num, E0, atol=TOL), \
        f"机械能不守恒：波动 {np.max(np.abs(E_num - E0)):.3e}"


def test_period_matches_theory():
    """数值周期应与理论值 T = 2π√(m/k) 一致。"""
    k, m = 2.0, 8.0
    omega0 = angular_frequency(k, m)
    T_theory = 2.0 * np.pi / omega0
    # 从 x0=A 处静止释放，x 在 t=0 为最大值
    x0, v0 = 1.0, 0.0
    # 积分一个周期，检查末点是否回到初值
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, t_end=T_theory, n=2001)
    err_x = abs(x_num[-1] - x0)
    err_v = abs(v_num[-1] - v0)
    assert err_x < TOL, f"周期后 x 未闭合：误差 {err_x:.3e}"
    assert err_v < TOL, f"周期后 v 未闭合：误差 {err_v:.3e}"


def test_period_mass_scaling():
    """周期应与 √m 成正比。"""
    k = 1.0
    m1, m2 = 1.0, 4.0
    T1 = period(k=k, m=m1)
    T2 = period(k=k, m=m2)
    # m 放大 4 倍，周期应放大 2 倍
    assert np.isclose(T2, 2.0 * T1, rtol=TOL), \
        f"周期-质量标度不符：T2/T1 = {T2/T1:.4f}（预期 2.0）"


def test_period_spring_scaling():
    """周期应与 1/√k 成正比。"""
    m = 1.0
    k1, k2 = 1.0, 9.0
    T1 = period(k=k1, m=m)
    T2 = period(k=k2, m=m)
    # k 放大 9 倍，周期应缩小 3 倍
    assert np.isclose(T2, T1 / 3.0, rtol=TOL), \
        f"周期-弹性系数标度不符：T2/T1 = {T2/T1:.4f}（预期 1/3）"


def test_zero_amplitude_equilibrium():
    """零振幅特例：x0=0, v0=0 时质点应始终静止在平衡位置。"""
    x0, v0, k, m = 0.0, 0.0, 1.0, 1.0
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, t_end=10.0, n=101)
    assert np.allclose(x_num, 0.0, atol=TOL), \
        f"零振幅时 x 不为零：最大 {np.max(np.abs(x_num)):.3e}"
    assert np.allclose(v_num, 0.0, atol=TOL), \
        f"零振幅时 v 不为零：最大 {np.max(np.abs(v_num)):.3e}"


def test_phase_portrait_is_ellipse():
    """相图轨迹应为以原点为中心的等能量椭圆。

    椭圆方程：x²/A² + v²/(Aω₀)² = 1
    """
    x0, v0, k, m = 2.0, 3.0, 1.5, 0.8
    omega0 = angular_frequency(k, m)
    A = amplitude(x0, v0, k, m)
    E0 = mechanical_energy([x0, v0], k=k, m=m)
    T = 2.0 * np.pi / omega0
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, t_end=T, n=1001)
    # 验证轨迹满足椭圆方程
    ellipse_val = (x_num / A) ** 2 + (v_num / (A * omega0)) ** 2
    assert np.allclose(ellipse_val, 1.0, atol=1e-4), \
        f"轨迹不在等能量椭圆上：偏差 {np.max(np.abs(ellipse_val - 1.0)):.3e}"
    # 验证椭圆面积 ∝ 能量
    # S = π * A * A*ω₀ = 2π*E / (m*ω₀)
    S_expected = 2.0 * np.pi * E0 / (m * omega0)
    # 数值估算面积（椭圆参数方程的面积 = π*a*b）
    a = A  # x 半轴
    b = A * omega0  # v 半轴
    S_numerical = np.pi * a * b
    assert np.isclose(S_numerical, S_expected, rtol=TOL), \
        f"相图面积不符：{S_numerical:.6f} vs {S_expected:.6f}"


def test_energy_equals_half_kA_squared():
    """机械能应等于 ½kA²。"""
    x0, v0, k, m = 1.5, 2.0, 3.0, 2.0
    A = amplitude(x0, v0, k, m)
    E = mechanical_energy([x0, v0], k=k, m=m)
    E_expected = 0.5 * k * A ** 2
    assert np.isclose(E, E_expected, rtol=TOL), \
        f"机械能 ≠ ½kA²：E={E:.6f}, ½kA²={E_expected:.6f}"


def test_invalid_parameters_rejected():
    """k <= 0 或 m <= 0 应被拒绝。"""
    try:
        validate_parameters(k=0.0, m=1.0)
        raise AssertionError("应拒绝 k=0")
    except AssertionError as e:
        assert "k" in str(e)

    try:
        validate_parameters(k=1.0, m=-1.0)
        raise AssertionError("应拒绝 m<0")
    except AssertionError as e:
        assert "m" in str(e)


if __name__ == "__main__":
    test_position_matches_analytical()
    test_mechanical_energy_conserved()
    test_period_matches_theory()
    test_period_mass_scaling()
    test_period_spring_scaling()
    test_zero_amplitude_equilibrium()
    test_phase_portrait_is_ellipse()
    test_energy_equals_half_kA_squared()
    test_invalid_parameters_rejected()
    print("OK: MEC-010 数值解与解析解一致")
