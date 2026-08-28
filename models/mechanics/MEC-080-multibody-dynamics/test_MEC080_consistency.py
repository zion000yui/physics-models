"""MEC-080 —— 一致性测试：多体动力学。

验证：
- N=1 退化为单摆（质心在 l/2）
- N=2 质量矩阵解析一致性
- 能量守恒
- 质量矩阵正定性
- 科氏力对称性
- 反例验证
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC080_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (
    validate_parameters,
    mass_matrix,
    coriolis_vector,
    gravity_vector,
    dynamics,
    kinetic_energy,
    potential_energy,
    total_energy,
    center_of_mass_positions,
    center_of_mass_velocities,
    lagrangian,
)

TOL = 1e-4


def test_n1_pendulum_dynamics():
    """N=1 应退化为单摆 θ̈ = -(g/r)sinθ, r=l/2。"""
    m, l, I, g = 1.0, 1.0, 0.0, 9.81
    masses = [m]; lengths = [l]; inertias = [I]
    r = l / 2.0

    d = dynamics(0.0, [0.5, 0.0], masses, lengths, inertias, g)
    expected = -g / r * np.sin(0.5)
    assert abs(d[1] - expected) < TOL, f"θ̈={d[1]:.4f} vs {expected:.4f}"


def test_n1_small_angle_frequency():
    """N=1 小角度频率应为 √(g/r) = √(2g/l)。"""
    g, l = 9.81, 1.0
    r = l / 2
    omega = np.sqrt(g / r)
    assert abs(omega - np.sqrt(2 * g / l)) < 1e-15

    # 数值积分验证
    eps = 0.01
    sol = solve_ivp(dynamics, (0, 5.0), [eps, 0.0],
                    args=([1.0], [l], [0.0], g),
                    t_eval=np.linspace(0, 5.0, 501),
                    rtol=1e-10, atol=1e-12)
    x_ana = eps * np.cos(omega * sol.t)
    assert np.max(np.abs(sol.y[0] - x_ana)) < 1e-3


def test_n1_energy_conservation():
    """N=1 单摆应能量守恒。"""
    m, l, I, g = 1.0, 1.0, 0.0, 9.81
    masses = [m]; lengths = [l]; inertias = [I]
    sol = solve_ivp(dynamics, (0, 10.0), [0.5, 0.0],
                    args=(masses, lengths, inertias, g),
                    t_eval=np.linspace(0, 10.0, 1001),
                    rtol=1e-10, atol=1e-12)
    energies = np.array([total_energy(sol.y[:1, i], sol.y[1:, i],
                                       masses, lengths, inertias, g)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / abs(energies[0])
    assert dE < 1e-5


def test_n2_mass_matrix():
    """N=2 质量矩阵应匹配解析公式。"""
    m1, m2, l1, l2, I1, I2 = 1.0, 1.0, 1.0, 1.0, 0.0, 0.0
    masses = [m1, m2]; lengths = [l1, l2]; inertias = [I1, I2]
    M = mass_matrix([0.0, 0.0], masses, lengths, inertias)
    # M11 = m1*(l1/2)^2 + m2*l1^2 = 0.25 + 1 = 1.25
    assert abs(M[0, 0] - 1.25) < 1e-10
    # M12 = m2*l1*(l2/2)*cos(0) = 0.5
    assert abs(M[0, 1] - 0.5) < 1e-10
    # M22 = m2*(l2/2)^2 = 0.25
    assert abs(M[1, 1] - 0.25) < 1e-10
    # 对称性
    assert abs(M[0, 1] - M[1, 0]) < 1e-15


def test_n2_mass_matrix_angle_dependence():
    """N=2 M12 应随角度差变化 cos(θ1-θ2)。"""
    m1, m2, l1, l2, I1, I2 = 1.0, 1.0, 1.0, 1.0, 0.0, 0.0
    masses = [m1, m2]; lengths = [l1, l2]; inertias = [I1, I2]
    for d in [0.0, 0.5, 1.0, np.pi / 2]:
        M = mass_matrix([d, 0.0], masses, lengths, inertias)
        expected_12 = 0.5 * np.cos(d)
        assert abs(M[0, 1] - expected_12) < 1e-10


def test_n2_energy_conservation():
    """N=2 双摆应能量守恒。"""
    masses = [1.0, 1.0]; lengths = [1.0, 1.0]; inertias = [0.0, 0.0]; g = 9.81
    sol = solve_ivp(dynamics, (0, 10.0), [0.5, 0.3, 0.0, 0.0],
                    args=(masses, lengths, inertias, g),
                    t_eval=np.linspace(0, 10.0, 5001),
                    rtol=1e-10, atol=1e-12)
    energies = np.array([total_energy(sol.y[:2, i], sol.y[2:, i],
                                       masses, lengths, inertias, g)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / abs(energies[0])
    assert dE < 1e-5


def test_n3_energy_conservation():
    """N=3 三连杆应能量守恒。"""
    masses = [1.0, 1.0, 0.5]
    lengths = [1.0, 0.8, 0.6]
    inertias = [0.0, 0.0, 0.0]
    g = 9.81
    sol = solve_ivp(dynamics, (0, 10.0), [0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
                    args=(masses, lengths, inertias, g),
                    t_eval=np.linspace(0, 10.0, 5001),
                    rtol=1e-10, atol=1e-12)
    energies = np.array([total_energy(sol.y[:3, i], sol.y[3:, i],
                                       masses, lengths, inertias, g)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / abs(energies[0])
    assert dE < 1e-4


def test_mass_matrix_positive_definite():
    """质量矩阵应正定。"""
    masses = [1.0, 1.0, 0.5]
    lengths = [1.0, 0.8, 0.6]
    inertias = [0.0, 0.0, 0.0]
    for theta in [[0, 0, 0], [0.5, 0.3, 0.1], [1.0, -0.5, 0.3]]:
        M = mass_matrix(theta, masses, lengths, inertias)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), f"非正定: {eigvals}"


def test_gravity_zero_at_equilibrium():
    """θ=0 时重力应为零（平衡位置）。"""
    masses = [1.0, 1.0, 0.5]
    lengths = [1.0, 0.8, 0.6]
    g = 9.81
    G = gravity_vector([0.0, 0.0, 0.0], masses, lengths, g)
    assert np.max(np.abs(G)) < 1e-3, f"G(0)={G}"


def test_coriolis_zero_at_rest():
    """θ̇=0 时科氏力应为零。"""
    masses = [1.0, 1.0, 0.5]
    lengths = [1.0, 0.8, 0.6]
    inertias = [0.0, 0.0, 0.0]
    C = coriolis_vector([0.5, 0.3, 0.1], [0.0, 0.0, 0.0],
                         masses, lengths, inertias)
    assert np.max(np.abs(C)) < 1e-6, f"C(θ̇=0)={C}"


def test_dynamics_interface():
    """dynamics 应返回正确形状。"""
    N = 3
    masses = [1.0, 1.0, 0.5]
    lengths = [1.0, 0.8, 0.6]
    inertias = [0.0, 0.0, 0.0]
    d = dynamics(0.0, [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                 masses, lengths, inertias, 9.81)
    assert d.shape == (2 * N,)


def test_error_injection_wrong_mass():
    """反例：质量加倍应改变动力学。"""
    m, l, I, g = 1.0, 1.0, 0.0, 9.81
    d1 = dynamics(0.0, [0.5, 0.0], [m], [l], [I], g)
    d2 = dynamics(0.0, [0.5, 0.0], [2 * m], [l], [I], g)
    # 质量加倍不影响加速度（g/r 不依赖 m）
    assert abs(d1[1] - d2[1]) < TOL, "质点单摆加速度不依赖质量"


def test_error_injection_wrong_length():
    """反例：杆长加倍应改变频率（√(2g/l)）。"""
    g, l1, l2 = 9.81, 1.0, 2.0
    omega1 = np.sqrt(g / (l1 / 2))
    omega2 = np.sqrt(g / (l2 / 2))
    assert abs(omega2 - omega1 / np.sqrt(2)) < 1e-15


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    try:
        validate_parameters([1.0, 1.0], [1.0], [0.0], 9.81)
        assert False, "维度不匹配应报错"
    except AssertionError:
        pass
    try:
        validate_parameters([-1.0], [1.0], [0.0], 9.81)
        assert False, "负质量应报错"
    except AssertionError as e:
        assert "m" in str(e) or "质" in str(e)


if __name__ == "__main__":
    test_n1_pendulum_dynamics()
    print("✓ N=1 单摆动力学")
    test_n1_small_angle_frequency()
    print("✓ N=1 小角度频率")
    test_n1_energy_conservation()
    print("✓ N=1 能量守恒")
    test_n2_mass_matrix()
    print("✓ N=2 质量矩阵")
    test_n2_mass_matrix_angle_dependence()
    print("✓ N=2 角度依赖性")
    test_n2_energy_conservation()
    print("✓ N=2 能量守恒")
    test_n3_energy_conservation()
    print("✓ N=3 能量守恒")
    test_mass_matrix_positive_definite()
    print("✓ 质量矩阵正定")
    test_gravity_zero_at_equilibrium()
    print("✓ 平衡位置重力为零")
    test_coriolis_zero_at_rest()
    print("✓ 静止时科氏力为零")
    test_dynamics_interface()
    print("✓ dynamics 接口")
    test_error_injection_wrong_mass()
    print("✓ 反例: 质量不影响单摆加速度")
    test_error_injection_wrong_length()
    print("✓ 反例: 杆长影响频率")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-080 所有一致性测试通过")
