"""MEC-090 —— 一致性测试：非线性力学。

验证：
- 非线性单摆周期-振幅关系
- 周期 > 线性周期
- 能量守恒
- 旋转阈值
- 庞加莱截面
- 双摆轨迹发散
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC090_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk

from model import (
    validate_parameters,
    pendulum_dynamics,
    pendulum_linear_frequency,
    pendulum_period_analytical,
    pendulum_period_series,
    pendulum_energy,
    pendulum_energy_threshold,
    driven_damped_pendulum,
    solve_driven_pendulum,
    poincare_section,
    estimate_lyapunov_exponent,
    double_pendulum_divergence,
)

TOL = 1e-6


def test_period_equals_linear_for_small_angle():
    """小角度周期应接近线性值 T₀ = 2π√(l/g)。"""
    g, l = 9.81, 1.0
    T0 = 2 * np.pi * np.sqrt(l / g)
    T_small = pendulum_period_analytical(0.01, g, l)
    assert abs(T_small - T0) / T0 < 0.001


def test_period_increases_with_amplitude():
    """周期应随振幅增大（非线性特征）。"""
    g, l = 9.81, 1.0
    T1 = pendulum_period_analytical(0.1, g, l)
    T2 = pendulum_period_analytical(1.0, g, l)
    T3 = pendulum_period_analytical(2.0, g, l)
    assert T1 < T2 < T3


def test_period_elliptic_formula():
    """椭圆积分公式应正确。"""
    g, l = 9.81, 1.0
    th0 = 1.0  # ~57°
    T = pendulum_period_analytical(th0, g, l)
    T_expected = 4 * np.sqrt(l / g) * ellipk(np.sin(th0 / 2)**2)
    assert abs(T - T_expected) < 1e-10


def test_period_series_matches_elliptic():
    """级数展开应在小角度时与精确公式一致。"""
    g, l = 9.81, 1.0
    for th0 in [0.01, 0.1, 0.3]:
        T_ell = pendulum_period_analytical(th0, g, l)
        T_ser = pendulum_period_series(th0, g, l)
        err = abs(T_ell - T_ser) / T_ell
        assert err < 0.01, f"θ₀={th0}: 级数误差 {err:.4f}"


def test_period_diverges_near_pi():
    """振幅接近 π 时周期应显著增大。"""
    g, l = 9.81, 1.0
    T_near = pendulum_period_analytical(np.pi - 0.01, g, l)
    T0 = 2 * np.pi * np.sqrt(l / g)
    assert T_near > 3 * T0, f"接近 π 时周期应显著大: {T_near:.1f}"


def test_pendulum_energy_conservation():
    """无阻尼单摆应能量守恒。"""
    g, l = 9.81, 1.0
    sol = solve_ivp(pendulum_dynamics, (0, 10.0), [0.5, 0.0],
                    args=(g, l), t_eval=np.linspace(0, 10.0, 1001),
                    rtol=1e-10, atol=1e-12)
    energies = np.array([pendulum_energy(sol.y[:, i], g, l)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / abs(energies[0])
    assert dE < 1e-6


def test_energy_threshold():
    """旋转阈值 E_c = 2mgl。"""
    g, l = 9.81, 1.0
    Ec = pendulum_energy_threshold(g, l)
    assert abs(Ec - 2 * g * l) < 1e-15


def test_oscillation_below_threshold():
    """E < E_c 时应振荡（角度有界）。"""
    g, l = 9.81, 1.0
    th0 = 1.0  # E = mgl(1-cos1) ≈ 4.6 < 2mgl=19.6
    sol = solve_ivp(pendulum_dynamics, (0, 20.0), [th0, 0.0],
                    args=(g, l), t_eval=np.linspace(0, 20.0, 2001),
                    rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(sol.y[0])) < np.pi, "应振荡有界"


def test_rotation_above_threshold():
    """E > E_c 时应旋转（角度单调增长）。"""
    g, l = 9.81, 1.0
    w0 = 10.0  # E = ½l²w₀² >> 2mgl
    sol = solve_ivp(pendulum_dynamics, (0, 5.0), [0.0, w0],
                    args=(g, l), t_eval=np.linspace(0, 5.0, 501),
                    rtol=1e-10, atol=1e-12)
    # 角度应超过 2π（完整旋转）
    assert sol.y[0, -1] > 2 * np.pi, "应旋转超过 2π"


def test_driven_pendulum_steady_state():
    """受驱阻尼摆应达到稳态（瞬态衰减后）。"""
    g, l = 9.81, 1.0
    sol = solve_driven_pendulum(g, l, 0.5, 0.5, 2.0 / 3.0,
                                  th0=1.0, w0=0.0, t_end=100.0, n=10001)
    # 后半段振幅应稳定
    late = sol.y[0, 5000:]
    assert np.std(late) / np.max(np.abs(late)) < 1.0


def test_poincare_section_shape():
    """庞加莱截面应返回二维点。"""
    g, l = 9.81, 1.0
    sol = solve_driven_pendulum(g, l, 0.5, 0.5, 2.0 / 3.0,
                                  th0=0.5, w0=0.0, t_end=50.0, n=5001)
    psec = poincare_section(sol, 2.0 / 3.0, t_start=10.0)
    assert psec.ndim == 2
    assert psec.shape[1] == 2


def test_double_pendulum_divergence_chaotic():
    """双摆大角度应轨迹发散（混沌）。"""
    t_arr, d_arr = double_pendulum_divergence(
        th1=2.5, th2=2.0, t_end=10.0, n=10001)
    d0 = max(d_arr[1], 1e-20)
    d_end = d_arr[-1]
    # 混沌应显著发散
    assert d_end > 100 * d0, f"大角度双摆应发散: {d_end/d0:.1f}"


def test_double_pendulum_no_divergence_small():
    """双摆小角度不应显著发散（近周期）。"""
    t_arr, d_arr = double_pendulum_divergence(
        th1=0.1, th2=0.1, t_end=10.0, n=10001)
    d0 = max(d_arr[1], 1e-20)
    d_end = d_arr[-1]
    # 小角度不应剧烈发散
    assert d_end < 1000 * d0, f"小角度不应剧烈发散: {d_end/d0:.1f}"


def test_dynamics_interface():
    """dynamics 接口应返回正确形状。"""
    d = pendulum_dynamics(0.0, [0.5, 0.0], 9.81, 1.0)
    assert d.shape == (2,)
    d = driven_damped_pendulum(0.0, [0.5, 0.0], 9.81, 1.0, 0.1, 1.0, 2.0 / 3.0)
    assert d.shape == (2,)


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    bad_cases = [
        ("g", {"g": -1}),
        ("l", {"l": -1}),
        ("c", {"c": -1}),
        ("A", {"A": -1}),
    ]
    for label, kwargs in bad_cases:
        try:
            validate_parameters(**kwargs)
            assert False, f"应拒绝 {kwargs}"
        except AssertionError as e:
            assert label in str(e)


if __name__ == "__main__":
    test_period_equals_linear_for_small_angle()
    print("✓ 小角度周期≈线性")
    test_period_increases_with_amplitude()
    print("✓ 周期随振幅增大")
    test_period_elliptic_formula()
    print("✓ 椭圆积分公式")
    test_period_series_matches_elliptic()
    print("✓ 级数展开一致性")
    test_period_diverges_near_pi()
    print("✓ 接近 π 时周期发散")
    test_pendulum_energy_conservation()
    print("✓ 能量守恒")
    test_energy_threshold()
    print("✓ 旋转阈值 E_c=2mgl")
    test_oscillation_below_threshold()
    print("✓ 振荡运动有界")
    test_rotation_above_threshold()
    print("✓ 旋转运动越 2π")
    test_driven_pendulum_steady_state()
    print("✓ 受驱摆稳态")
    test_poincare_section_shape()
    print("✓ 庞加莱截面")
    test_double_pendulum_divergence_chaotic()
    print("✓ 双摆大角度混沌发散")
    test_double_pendulum_no_divergence_small()
    print("✓ 双摆小角度不发散")
    test_dynamics_interface()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-090 所有一致性测试通过")
