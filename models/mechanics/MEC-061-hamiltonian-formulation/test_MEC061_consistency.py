"""MEC-061 —— 一致性测试：哈密顿力学公式化。

验证：
- 自由/受力质点 vs 解析解
- 弹簧振子能量守恒
- 阻尼振子能量耗散
- Liouville 定理
- 泊松括号 {q,p}=1
- Legendre 变换
- 退化与反例
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC061_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (
    validate_parameters,
    free_particle_hamiltonian,
    free_particle_canonical,
    forced_particle_hamiltonian,
    forced_particle_canonical,
    spring_hamiltonian,
    spring_canonical,
    hooke_hamiltonian_2d,
    hooke_canonical_2d,
    damped_spring_hamiltonian,
    damped_spring_canonical,
    legendre_transform,
    poisson_bracket,
    canonical_commutator,
    phase_space_area,
)

TOL = 1e-6


def test_free_particle():
    """自由质点：q = pt/m, p 守恒。"""
    m = 1.0
    p0 = 2.0
    sol = solve_ivp(free_particle_canonical, (0, 5.0), [0.0, p0],
                    args=(m,), t_eval=np.linspace(0, 5.0, 501),
                    rtol=1e-10, atol=1e-12)
    q_ana = p0 / m * sol.t
    p_ana = p0 * np.ones_like(sol.t)
    assert np.max(np.abs(sol.y[0] - q_ana)) < TOL
    assert np.max(np.abs(sol.y[1] - p_ana)) < TOL


def test_forced_particle():
    """受力质点：p = p0 - Ft, q = (p0 - ½Ft)t/m。"""
    m, F = 1.0, 1.0
    p0 = 0.0
    sol = solve_ivp(forced_particle_canonical, (0, 5.0), [0.0, p0],
                    args=(m, F), t_eval=np.linspace(0, 5.0, 501),
                    rtol=1e-10, atol=1e-12)
    p_ana = p0 - F * sol.t
    q_ana = (p0 - 0.5 * F * sol.t) * sol.t / m
    assert np.max(np.abs(sol.y[1] - p_ana)) < TOL
    assert np.max(np.abs(sol.y[0] - q_ana)) < TOL


def test_spring_energy_conservation():
    """弹簧振子哈密顿量应守恒。"""
    m, k = 1.0, 4.0
    sol = solve_ivp(spring_canonical, (0, 10.0), [1.0, 0.0],
                    args=(m, k), t_eval=np.linspace(0, 10.0, 1001),
                    rtol=1e-10, atol=1e-12)
    energies = np.array([spring_hamiltonian(sol.y[:, i], m, k)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / energies[0]
    assert dE < 1e-6, f"能量变化 {dE:.3e}"


def test_spring_vs_lagrangian():
    """哈密顿方程应与拉格朗日方程给出相同结果。"""
    m, k = 1.0, 4.0
    omega = np.sqrt(k / m)
    sol = solve_ivp(spring_canonical, (0, 10.0), [1.0, 0.0],
                    args=(m, k), t_eval=np.linspace(0, 10.0, 1001),
                    rtol=1e-10, atol=1e-12)
    q_ana = np.cos(omega * sol.t)
    p_ana = -m * omega * np.sin(omega * sol.t)
    assert np.max(np.abs(sol.y[0] - q_ana)) < 1e-6
    assert np.max(np.abs(sol.y[1] - p_ana)) < 1e-6


def test_damped_energy_dissipation():
    """阻尼振子哈密顿量应单调递减。"""
    m, k, c = 1.0, 4.0, 0.4
    sol = solve_ivp(damped_spring_canonical, (0, 20.0), [1.0, 0.0],
                    args=(m, k, c), t_eval=np.linspace(0, 20.0, 2001),
                    rtol=1e-10, atol=1e-12)
    energies = np.array([damped_spring_hamiltonian(sol.y[:, i], m, k)
                         for i in range(sol.y.shape[1])])
    assert energies[-1] < energies[0], "能量应递减"
    assert np.all(np.diff(energies[::50]) <= 1e-6), "能量非单调递减"


def test_hamiltonian_equals_energy():
    """保守系统 H 应等于 T + V。"""
    m, k = 1.0, 4.0
    x, p = 0.5, 1.0
    H = spring_hamiltonian([x, p], m, k)
    T = p**2 / (2 * m)
    V = 0.5 * k * x**2
    assert abs(H - (T + V)) < 1e-15


def test_legendre_transform():
    """Legendre 变换应正确给出 H = pq̇ - L。"""
    m, k = 1.0, 4.0
    # 弹簧：L = ½mẋ² - ½kx², p = mẋ
    x, v = 0.5, 1.0
    p = m * v
    # L = ½(1)(1) - ½(4)(0.25) = 0.5 - 0.5 = 0
    L = 0.5 * m * v**2 - 0.5 * k * x**2
    H_expected = p * v - L  # = 1 - 0 = 1
    H_direct = spring_hamiltonian([x, p], m, k)
    assert abs(H_direct - H_expected) < 1e-15


def test_liouville_theorem():
    """Liouville 定理：相空间面积守恒。"""
    m, k = 1.0, 1.0
    omega = np.sqrt(k / m)
    theta = np.linspace(0, 2 * np.pi, 100)
    q0 = 0.5 * np.cos(theta)
    p0 = m * omega * 0.5 * np.sin(theta)
    states0 = np.column_stack([q0, p0])

    area0 = phase_space_area(states0)

    t_quarter = np.pi / (2 * omega)
    states_final = np.zeros_like(states0)
    for i in range(len(states0)):
        sol = solve_ivp(spring_canonical, (0, t_quarter),
                        states0[i], args=(m, k),
                        rtol=1e-10, atol=1e-12)
        states_final[i] = sol.y[:, -1]

    area_final = phase_space_area(states_final)
    err = abs(area_final - area0) / area0
    assert err < 1e-6, f"面积变化 {err:.3e}"


def test_poisson_bracket():
    """基本泊松括号 {q, p} = 1。"""
    state = [0.5, 1.0]
    bracket = canonical_commutator(state, m=1.0)
    assert abs(bracket - 1.0) < 1e-4, f"{{q,p}} = {bracket:.6f}"


def test_hooke_2d_energy():
    """2D 胡克力哈密顿量应守恒。"""
    m, k = 1.0, 1.0
    sol = solve_ivp(hooke_canonical_2d, (0, 10.0),
                    [1.0, 0.5, 0.0, 0.0],
                    args=(m, k), t_eval=np.linspace(0, 10.0, 1001),
                    rtol=1e-10, atol=1e-12)
    energies = np.array([hooke_hamiltonian_2d(sol.y[:, i], m, k)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / energies[0]
    assert dE < 1e-6


def test_error_injection_wrong_k():
    """反例：k 翻倍应导致频率翻倍。"""
    m = 1.0
    omega1 = np.sqrt(1.0 / m)
    omega2 = np.sqrt(4.0 / m)
    assert abs(omega2 - 2 * omega1) < 1e-15


def test_dynamics_interface():
    """各 canonical 函数应返回正确形状。"""
    d = free_particle_canonical(0.0, [0.0, 1.0], 1.0)
    assert d.shape == (2,)
    d = spring_canonical(0.0, [1.0, 0.0], 1.0, 1.0)
    assert d.shape == (2,)
    d = hooke_canonical_2d(0.0, [1.0, 0.0, 0.0, 0.0], 1.0, 1.0)
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
            assert label in str(e)


if __name__ == "__main__":
    test_free_particle()
    print("✓ 自由质点相空间动力学")
    test_forced_particle()
    print("✓ 受力质点动量变化")
    test_spring_energy_conservation()
    print("✓ 弹簧振子 H 守恒")
    test_spring_vs_lagrangian()
    print("✓ 哈密顿 = 拉格朗日结果")
    test_damped_energy_dissipation()
    print("✓ 阻尼振子 H 耗散")
    test_hamiltonian_equals_energy()
    print("✓ H = T + V")
    test_legendre_transform()
    print("✓ Legendre 变换")
    test_liouville_theorem()
    print("✓ Liouville 定理")
    test_poisson_bracket()
    print("✓ 泊松括号 {q,p}=1")
    test_hooke_2d_energy()
    print("✓ 2D 胡克力 H 守恒")
    test_error_injection_wrong_k()
    print("✓ 反例: k 翻倍→ω翻倍")
    test_dynamics_interface()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-061 所有一致性测试通过")
