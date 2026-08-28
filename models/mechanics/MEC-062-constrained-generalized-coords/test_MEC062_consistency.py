"""MEC-062 —— 一致性测试：广义坐标与约束系统。

验证：
- 单摆：能量守恒 + 小角度频率 + 约束
- 阿特伍德机：加速度 + 张力 + 解析解
- 斜面纯滚动：加速度 + 能量 + 摩擦力
- 约束验证
- 反例验证
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC062_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (
    validate_parameters,
    pendulum_dynamics,
    pendulum_small_angle_frequency,
    pendulum_energy,
    pendulum_lagrangian,
    atwood_dynamics,
    atwood_acceleration,
    atwood_tension,
    rolling_incline_dynamics,
    rolling_incline_acceleration,
    rolling_incline_energy,
    rolling_incline_lagrangian,
    lagrange_multiplier_pendulum,
    static_friction_required,
    max_incline_angle_for_pure_rolling,
    verify_pendulum_constraint,
    verify_rolling_constraint,
)

TOL = 1e-6


def test_pendulum_energy():
    """单摆应能量守恒。"""
    m, g, l = 1.0, 9.81, 1.0
    sol = solve_ivp(pendulum_dynamics, (0, 10.0), [0.5, 0.0],
                    args=(m, g, l), t_eval=np.linspace(0, 10.0, 1001),
                    rtol=1e-10, atol=1e-12)
    energies = np.array([pendulum_energy(sol.y[:, i], m, g, l)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / abs(energies[0])
    assert dE < 1e-6, f"能量变化 {dE:.3e}"


def test_pendulum_small_angle():
    """单摆小角度应匹配 ω = √(g/l)。"""
    g, l = 9.81, 1.0
    omega = pendulum_small_angle_frequency(g, l)
    assert abs(omega - np.sqrt(g / l)) < 1e-15

    # 小角度数值积分
    eps = 0.01
    sol = solve_ivp(pendulum_dynamics, (0, 5.0), [eps, 0.0],
                    args=(1.0, g, l), t_eval=np.linspace(0, 5.0, 501),
                    rtol=1e-10, atol=1e-12)
    x_ana = eps * np.cos(omega * sol.t)
    assert np.max(np.abs(sol.y[0] - x_ana)) < 1e-4


def test_pendulum_constraint():
    """单摆约束 x²+y²=l² 应满足。"""
    assert verify_pendulum_constraint([0.5, 0.0], 1.0)
    assert verify_pendulum_constraint([np.pi / 4, 0.0], 2.0)


def test_atwood_acceleration():
    """阿特伍德机加速度公式。"""
    m1, m2, g = 2.0, 1.0, 9.81
    a = atwood_acceleration(m1, m2, g)
    assert abs(a - (m1 - m2) * g / (m1 + m2)) < 1e-15


def test_atwood_tension():
    """阿特伍德机张力公式。"""
    m1, m2, g = 2.0, 1.0, 9.81
    T = atwood_tension(m1, m2, g)
    assert abs(T - 2 * m1 * m2 * g / (m1 + m2)) < 1e-15


def test_atwood_dynamics():
    """阿特伍德机数值解应匹配解析解。"""
    m1, m2, g = 2.0, 1.0, 9.81
    a = atwood_acceleration(m1, m2, g)
    sol = solve_ivp(atwood_dynamics, (0, 5.0), [0.0, 0.0],
                    args=(m1, m2, g, 2.0), t_eval=np.linspace(0, 5.0, 501),
                    rtol=1e-10, atol=1e-12)
    x_ana = 0.5 * a * sol.t**2
    v_ana = a * sol.t
    assert np.max(np.abs(sol.y[0] - x_ana)) < TOL
    assert np.max(np.abs(sol.y[1] - v_ana)) < TOL


def test_rolling_acceleration():
    """纯滚动加速度 a = g·sinθ/(1+k)。"""
    g, k, theta = 9.81, 0.4, 30.0
    a = rolling_incline_acceleration(g, k, theta)
    expected = g * np.sin(np.radians(theta)) / (1 + k)
    assert abs(a - expected) < 1e-15


def test_rolling_energy():
    """纯滚动应能量守恒。"""
    m, g, R, k, theta = 1.0, 9.81, 0.5, 0.4, 30.0
    # 给初始位移以避免零能量
    sol = solve_ivp(rolling_incline_dynamics, (0, 5.0), [1.0, 0.0],
                    args=(m, g, R, k, theta),
                    t_eval=np.linspace(0, 5.0, 501),
                    rtol=1e-10, atol=1e-12)
    energies = np.array([rolling_incline_energy(sol.y[:, i], m, g, R, k, theta)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0]))
    assert dE < 1e-6, f"能量变化 {dE:.3e}"


def test_rolling_vs_sliding():
    """纯滚动加速度应小于纯滑动加速度。"""
    g, k, theta = 9.81, 0.4, 30.0
    a_roll = rolling_incline_acceleration(g, k, theta)
    a_slide = g * np.sin(np.radians(theta))
    assert a_roll < a_slide, "纯滚动应慢于纯滑动"
    assert abs(a_roll / a_slide - 1 / (1 + k)) < 1e-15


def test_friction_force():
    """静摩擦力应满足 f_s = k·m·g·sinθ/(1+k)。"""
    m, g, k, theta = 1.0, 9.81, 0.4, 30.0
    f_s = static_friction_required(m, g, k, theta)
    expected = k * m * g * np.sin(np.radians(theta)) / (1 + k)
    assert abs(f_s - expected) < 1e-15


def test_max_incline_angle():
    """最大角度应满足 tan(θ_max) = μ_s(1+k)/k。"""
    mu_s, k = 0.5, 0.4
    theta_max = max_incline_angle_for_pure_rolling(mu_s, k)
    expected = np.degrees(np.arctan(mu_s * (1 + k) / k))
    assert abs(theta_max - expected) < 1e-10


def test_constraint_force_pendulum():
    """单摆静止时绳张力 = mg。"""
    m, g, l = 1.0, 9.81, 1.0
    T = lagrange_multiplier_pendulum([0.0, 0.0], m, g, l)
    assert abs(T - m * g) < 1e-15


def test_rolling_constraint_verification():
    """纯滚动约束验证。"""
    R = 0.5
    ok_x, ok_v = verify_rolling_constraint([0.3, 1.0], R)
    assert ok_x and ok_v


def test_error_injection_wrong_k():
    """反例：k=0（无转动惯量）时应退化为纯滑动。"""
    g, theta = 9.81, 30.0
    a_roll = rolling_incline_acceleration(g, 0.001, theta)
    a_slide = g * np.sin(np.radians(theta))
    err = abs(a_roll - a_slide) / a_slide
    assert err < 0.01, f"k→0 时应退化为纯滑动: err={err:.4f}"


def test_error_injection_wrong_atwood():
    """反例：m1=m2 时加速度应为零。"""
    g = 9.81
    a = atwood_acceleration(1.0, 1.0, g)
    assert abs(a) < 1e-15


def test_dynamics_interface():
    """各 dynamics 函数应返回正确形状。"""
    d = pendulum_dynamics(0.0, [0.5, 0.0], 1.0, 9.81, 1.0)
    assert d.shape == (2,)
    d = atwood_dynamics(0.0, [0.0, 0.0], 2.0, 1.0, 9.81, 2.0)
    assert d.shape == (2,)
    d = rolling_incline_dynamics(0.0, [0.0, 0.0], 1.0, 9.81, 0.5, 0.4, 30.0)
    assert d.shape == (2,)


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    bad_cases = [
        ("m", {"m": -1}),
        ("g", {"g": -1}),
        ("l", {"l": -1}),
        ("k", {"k": 1.5}),
        ("R", {"R": -1}),
    ]
    for label, kwargs in bad_cases:
        try:
            validate_parameters(**kwargs)
            assert False, f"应拒绝 {kwargs}"
        except AssertionError as e:
            assert label in str(e)


if __name__ == "__main__":
    test_pendulum_energy()
    print("✓ 单摆能量守恒")
    test_pendulum_small_angle()
    print("✓ 单摆小角度频率")
    test_pendulum_constraint()
    print("✓ 单摆约束验证")
    test_atwood_acceleration()
    print("✓ 阿特伍德加速度")
    test_atwood_tension()
    print("✓ 阿特伍德张力")
    test_atwood_dynamics()
    print("✓ 阿特伍德数值解")
    test_rolling_acceleration()
    print("✓ 纯滚动加速度")
    test_rolling_energy()
    print("✓ 纯滚动能量守恒")
    test_rolling_vs_sliding()
    print("✓ 滚动<滑动")
    test_friction_force()
    print("✓ 静摩擦力公式")
    test_max_incline_angle()
    print("✓ 最大角度")
    test_constraint_force_pendulum()
    print("✓ 单摆静止张力=mg")
    test_rolling_constraint_verification()
    print("✓ 纯滚动约束验证")
    test_error_injection_wrong_k()
    print("✓ 反例: k→0→纯滑动")
    test_error_injection_wrong_atwood()
    print("✓ 反例: m1=m2→a=0")
    test_dynamics_interface()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-062 所有一致性测试通过")
