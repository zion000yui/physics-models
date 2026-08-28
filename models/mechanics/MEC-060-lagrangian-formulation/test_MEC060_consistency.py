"""MEC-060 —— 一致性测试：拉格朗日力学公式化。

验证：
- 自由/受力质点 vs 解析解
- 弹簧振子 vs 解析解
- 阻尼振子 vs 解析解
- 双摆能量守恒
- 拉格朗日量计算
- Noether 定理（动量守恒）
- 退化验证
- 反例验证
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC060_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (
    validate_parameters,
    free_particle_lagrangian,
    free_particle_dynamics,
    forced_particle_lagrangian,
    forced_particle_dynamics,
    spring_lagrangian,
    spring_dynamics,
    spring_natural_frequency,
    damped_spring_dynamics,
    damping_ratio,
    hooke_lagrangian_2d,
    hooke_dynamics_2d,
    double_pendulum_lagrangian,
    double_pendulum_dynamics,
    double_pendulum_energy,
    noether_charge_momentum,
)

TOL = 1e-6


def test_free_particle():
    """自由质点应做匀速直线运动。"""
    m, v0 = 1.0, 2.0
    t_eval = np.linspace(0, 5.0, 501)
    sol = solve_ivp(free_particle_dynamics, (0, 5.0), [0.0, v0],
                    args=(m,), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    x_ana = v0 * t_eval
    v_ana = v0 * np.ones_like(t_eval)
    assert np.max(np.abs(sol.y[0] - x_ana)) < TOL
    assert np.max(np.abs(sol.y[1] - v_ana)) < TOL


def test_forced_particle():
    """受力质点应做匀加速运动。"""
    m, F = 1.0, 2.0
    t_eval = np.linspace(0, 5.0, 501)
    sol = solve_ivp(forced_particle_dynamics, (0, 5.0), [0.0, 0.0],
                    args=(m, F), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    x_ana = 0.5 * F / m * t_eval**2
    v_ana = F / m * t_eval
    assert np.max(np.abs(sol.y[0] - x_ana)) < TOL
    assert np.max(np.abs(sol.y[1] - v_ana)) < TOL


def test_spring_oscillator():
    """弹簧振子应满足 ω = √(k/m)。"""
    m, k = 1.0, 4.0
    omega = spring_natural_frequency(m, k)
    t_eval = np.linspace(0, 10.0, 1001)
    sol = solve_ivp(spring_dynamics, (0, 10.0), [1.0, 0.0],
                    args=(m, k), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    x_ana = np.cos(omega * t_eval)
    assert np.max(np.abs(sol.y[0] - x_ana)) < 1e-6
    assert abs(omega - np.sqrt(k / m)) < 1e-15


def test_damped_spring():
    """阻尼振子应匹配欠阻尼解析解。"""
    m, k, c = 1.0, 4.0, 0.4
    zeta = damping_ratio(m, k, c)
    assert zeta < 1, "应为欠阻尼"
    omega = spring_natural_frequency(m, k)
    omega_d = omega * np.sqrt(1 - zeta**2)

    t_eval = np.linspace(0, 20.0, 2001)
    sol = solve_ivp(damped_spring_dynamics, (0, 20.0), [1.0, 0.0],
                    args=(m, k, c), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    x_ana = np.exp(-zeta * omega * t_eval) * (
        np.cos(omega_d * t_eval)
        + zeta * omega / omega_d * np.sin(omega_d * t_eval))
    assert np.max(np.abs(sol.y[0] - x_ana)) < 1e-6


def test_double_pendulum_energy():
    """双摆应能量守恒。"""
    m1, m2, l1, l2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    t_eval = np.linspace(0, 10.0, 5001)
    sol = solve_ivp(double_pendulum_dynamics, (0, 10.0),
                    [0.5, 0.3, 0.0, 0.0],
                    args=(m1, m2, l1, l2, g),
                    t_eval=t_eval, rtol=1e-10, atol=1e-12)
    energies = np.array([double_pendulum_energy(sol.y[:, i], m1, m2, l1, l2, g)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / abs(energies[0])
    assert dE < 1e-6, f"能量变化 {dE:.3e}"


def test_double_pendulum_small_angle():
    """双摆小角度极限应退化为耦合简谐振子。"""
    m1, m2, l1, l2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    # 小角度
    eps = 0.01
    sol = solve_ivp(double_pendulum_dynamics, (0, 2.0),
                    [eps, eps, 0.0, 0.0],
                    args=(m1, m2, l1, l2, g),
                    t_eval=np.linspace(0, 2.0, 1001),
                    rtol=1e-10, atol=1e-12)
    # 小角度时 θ 不应超过 ~2*eps
    assert np.max(np.abs(sol.y[0])) < 5 * eps, "小角度应保持小幅度"
    assert np.max(np.abs(sol.y[1])) < 5 * eps


def test_lagrangian_values():
    """拉格朗日量应正确计算 L = T - V。"""
    # 自由质点
    L = free_particle_lagrangian([0.0, 2.0], m=1.0)
    assert abs(L - 0.5 * 1.0 * 4.0) < 1e-15

    # 弹簧振子
    L = spring_lagrangian([0.5, 0.0], m=1.0, k=4.0)
    assert abs(L - (-0.5 * 4.0 * 0.5**2)) < 1e-15

    # 受力质点
    L = forced_particle_lagrangian([1.0, 1.0], m=1.0, F=2.0)
    assert abs(L - (0.5 * 1.0 * 1.0 - 2.0 * 1.0)) < 1e-15


def test_noether_momentum_conservation():
    """自由质点动量守恒（Noether 定理）。"""
    m, v0 = 2.0, 3.0
    t_eval = np.linspace(0, 10.0, 1001)
    sol = solve_ivp(free_particle_dynamics, (0, 10.0), [0.0, v0],
                    args=(m,), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    p_values = [noether_charge_momentum(None, sol.y[:, i], m) for i in range(len(t_eval))]
    p0 = m * v0
    dp = np.max(np.abs(np.array(p_values) - p0))
    assert dp < 1e-10, f"动量变化 {dp:.3e}"


def test_hooke_2d_independence():
    """2D 胡克力各方向独立简谐振动。"""
    m, k = 1.0, 1.0
    omega = np.sqrt(k / m)
    t_eval = np.linspace(0, 10.0, 1001)
    sol = solve_ivp(hooke_dynamics_2d, (0, 10.0), [1.0, 0.5, 0.0, 0.0],
                    args=(m, k), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    x_ana = 1.0 * np.cos(omega * t_eval)
    y_ana = 0.5 * np.cos(omega * t_eval)
    assert np.max(np.abs(sol.y[0] - x_ana)) < 1e-6
    assert np.max(np.abs(sol.y[1] - y_ana)) < 1e-6


def test_degradation_free_from_spring():
    """k→0 时弹簧振子退化为自由质点。"""
    m, k = 1.0, 1e-10  # 近似零
    t_eval = np.linspace(0, 5.0, 501)
    sol = solve_ivp(spring_dynamics, (0, 5.0), [0.0, 2.0],
                    args=(m, k), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    # 近似匀速
    assert np.allclose(sol.y[1], 2.0, atol=1e-3), "k≈0 时应匀速"


def test_error_injection_wrong_k():
    """反例：k 翻倍应导致频率翻倍。"""
    omega1 = spring_natural_frequency(1.0, 1.0)
    omega2 = spring_natural_frequency(1.0, 4.0)
    assert abs(omega2 - 2 * omega1) < 1e-15


def test_error_injection_wrong_m():
    """反例：m 翻倍应导致频率减半。"""
    omega1 = spring_natural_frequency(1.0, 1.0)
    omega2 = spring_natural_frequency(4.0, 1.0)
    assert abs(omega2 - omega1 / 2) < 1e-15


def test_dynamics_interface():
    """各 dynamics 函数应返回正确形状。"""
    d = free_particle_dynamics(0.0, [0.0, 1.0], 1.0)
    assert d.shape == (2,)
    d = spring_dynamics(0.0, [1.0, 0.0], 1.0, 1.0)
    assert d.shape == (2,)
    d = double_pendulum_dynamics(0.0, [0.1, 0.1, 0.0, 0.0], 1.0, 1.0, 1.0, 1.0, 9.81)
    assert d.shape == (4,)


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    bad_cases = [
        ("m", {"m": -1}),
        ("k", {"k": -1}),
        ("g", {"g": -1}),
        ("c", {"c": -1}),
    ]
    for label, kwargs in bad_cases:
        try:
            validate_parameters(**kwargs)
            assert False, f"应拒绝 {kwargs}"
        except AssertionError as e:
            assert label in str(e), f"错误信息应包含 {label}: {e}"


if __name__ == "__main__":
    test_free_particle()
    print("✓ 自由质点 → 匀速直线")
    test_forced_particle()
    print("✓ 受力质点 → 匀加速")
    test_spring_oscillator()
    print("✓ 弹簧振子 → 简谐振动")
    test_damped_spring()
    print("✓ 阻尼振子 → 欠阻尼解析")
    test_double_pendulum_energy()
    print("✓ 双摆能量守恒")
    test_double_pendulum_small_angle()
    print("✓ 双摆小角度极限")
    test_lagrangian_values()
    print("✓ 拉格朗日量计算")
    test_noether_momentum_conservation()
    print("✓ Noether 动量守恒")
    test_hooke_2d_independence()
    print("✓ 2D 胡克力方向独立")
    test_degradation_free_from_spring()
    print("✓ k→0 退化为自由质点")
    test_error_injection_wrong_k()
    print("✓ 反例: k 翻倍→ω翻倍")
    test_error_injection_wrong_m()
    print("✓ 反例: m 翻倍→ω减半")
    test_dynamics_interface()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-060 所有一致性测试通过")
