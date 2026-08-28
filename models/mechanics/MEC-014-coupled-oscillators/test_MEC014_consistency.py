"""MEC-014 —— 一致性测试：数值解 vs 解析解。

验证：
- 主求解方法（简正模态分解）与 SciPy 数值解的一致性
- 简正频率与理论值一致
- 对称系统中的同相模态和反相模态
- 单一简正模态下模态形状保持不变
- 多模态叠加情况下的运动一致性
- 无耦合极限退化为两个独立的 MEC-010 简谐振子
- 总机械能守恒
- 非法参数处理

运行方法（在本文件所在目录执行）：
    python test_MEC014_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    normal_modes, mechanical_energy, stiffness_matrix, mass_matrix

TOL = 1e-6


def _solve(x1_0=1.0, x2_0=0.0, v1_0=0.0, v2_0=0.0,
           m1=1.0, m2=1.0, k1=1.0, k2=1.0, kc=0.5,
           t_end=10.0, n=401):
    """小工具：跑一次数值积分，返回 (t, x1, x2, v1, v2)。"""
    initial_state = np.array([x1_0, x2_0, v1_0, v2_0], dtype=float)
    validate_parameters(m1=m1, m2=m2, k1=k1, k2=k2, kc=kc)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(m1, m2, k1, k2, kc),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def test_matches_analytical():
    """数值解应与简正模态分解解析解一致。"""
    m1, m2, k1, k2, kc = 1.0, 1.0, 1.0, 1.0, 0.5
    x1_0, x2_0, v1_0, v2_0 = 1.0, 0.5, 0.3, -0.2
    t_end, n = 10.0, 801
    t, x1_n, x2_n, v1_n, v2_n = _solve(
        x1_0, x2_0, v1_0, v2_0, m1, m2, k1, k2, kc, t_end=t_end, n=n)
    x1_a, x2_a, v1_a, v2_a = analytical(
        t, [x1_0, x2_0, v1_0, v2_0], m1, m2, k1, k2, kc)
    err_x1 = np.max(np.abs(x1_n - x1_a))
    err_x2 = np.max(np.abs(x2_n - x2_a))
    err_v1 = np.max(np.abs(v1_n - v1_a))
    err_v2 = np.max(np.abs(v2_n - v2_a))
    assert err_x1 < TOL, f"x1 误差 {err_x1:.3e} 超出容差 {TOL}"
    assert err_x2 < TOL, f"x2 误差 {err_x2:.3e} 超出容差 {TOL}"
    assert err_v1 < TOL, f"v1 误差 {err_v1:.3e} 超出容差 {TOL}"
    assert err_v2 < TOL, f"v2 误差 {err_v2:.3e} 超出容差 {TOL}"


def test_normal_frequencies_symmetric():
    """对称系统简正频率应与理论值一致。"""
    m, k, kc = 1.0, 1.0, 0.5
    modes = normal_modes(m, m, k, k, kc)
    # 同相模态：ω₁ = √(k/m)
    omega1_theory = np.sqrt(k / m)
    # 反相模态：ω₂ = √((k+2kc)/m)
    omega2_theory = np.sqrt((k + 2 * kc) / m)
    assert np.isclose(modes[0]['omega'], omega1_theory, rtol=TOL), \
        f"同相频率不符: {modes[0]['omega']:.6f} vs {omega1_theory:.6f}"
    assert np.isclose(modes[1]['omega'], omega2_theory, rtol=TOL), \
        f"反相频率不符: {modes[1]['omega']:.6f} vs {omega2_theory:.6f}"


def test_normal_frequencies_asymmetric():
    """非对称系统简正频率应满足特征方程 det(K - ω²M) = 0。"""
    m1, m2, k1, k2, kc = 2.0, 1.5, 3.0, 2.0, 0.8
    modes = normal_modes(m1, m2, k1, k2, kc)
    K = stiffness_matrix(k1, k2, kc)
    M = mass_matrix(m1, m2)
    for mode in modes:
        omega = mode['omega']
        # 验证 det(K - ω²M) = 0
        det_val = np.linalg.det(K - omega ** 2 * M)
        assert abs(det_val) < 1e-6, \
            f"频率 {omega:.6f} 不满足特征方程: det={det_val:.3e}"


def test_in_phase_mode_shape():
    """对称系统同相模态应为 [1, 1]。"""
    m, k, kc = 1.0, 1.0, 0.5
    modes = normal_modes(m, m, k, k, kc)
    mode1 = modes[0]['mode']
    assert np.allclose(mode1, [1.0, 1.0], atol=1e-10), \
        f"同相模态形状不符: {mode1}"


def test_anti_phase_mode_shape():
    """对称系统反相模态应为 [1, -1]。"""
    m, k, kc = 1.0, 1.0, 0.5
    modes = normal_modes(m, m, k, k, kc)
    mode2 = modes[1]['mode']
    assert np.allclose(mode2, [1.0, -1.0], atol=1e-10), \
        f"反相模态形状不符: {mode2}"


def test_single_mode_preserves_shape():
    """单一简正模态激励下，位移比例应保持不变。"""
    m, k, kc = 1.0, 1.0, 0.5
    modes = normal_modes(m, m, k, k, kc)
    # 反相模态 [1, -1]，初始位移按此比例
    phi = modes[1]['mode']
    A = 1.0
    x1_0, x2_0 = A * phi[0], A * phi[1]
    t_end, n = 5.0, 401
    t, x1_n, x2_n, _, _ = _solve(
        x1_0, x2_0, 0.0, 0.0, m, m, k, k, kc, t_end=t_end, n=n)
    # 在运动过程中，x2/x1 应始终保持 phi[1]/phi[0] = -1
    for i in range(0, len(t), 20):
        if abs(x1_n[i]) > 1e-6:
            ratio = x2_n[i] / x1_n[i]
            assert np.isclose(ratio, phi[1] / phi[0], atol=1e-4), \
                f"模态形状变化: t={t[i]:.3f}, ratio={ratio:.6f} vs {phi[1]/phi[0]:.6f}"


def test_multi_mode_superposition():
    """多模态叠加：初始条件为两个模态的线性组合，运动应正确。"""
    m, k, kc = 1.0, 1.0, 0.5
    # 初始条件 = 模态1(幅度A1) + 模态2(幅度A2)
    A1, A2 = 0.7, 0.3
    modes = normal_modes(m, m, k, k, kc)
    phi1, phi2 = modes[0]['mode'], modes[1]['mode']
    x1_0 = A1 * phi1[0] + A2 * phi2[0]
    x2_0 = A1 * phi1[1] + A2 * phi2[1]
    t_end, n = 10.0, 801
    t, x1_n, x2_n, v1_n, v2_n = _solve(
        x1_0, x2_0, 0.0, 0.0, m, m, k, k, kc, t_end=t_end, n=n)
    x1_a, x2_a, v1_a, v2_a = analytical(
        t, [x1_0, x2_0, 0.0, 0.0], m, m, k, k, kc)
    err = np.max(np.abs(x1_n - x1_a)) + np.max(np.abs(x2_n - x2_a))
    assert err < TOL, f"多模态叠加误差 {err:.3e}"


def test_no_coupling_degenerates_to_MEC010():
    """无耦合极限（kc=0）退化为两个独立的 MEC-010 简谐振子。"""
    m1, m2, k1, k2, kc = 1.0, 2.0, 3.0, 5.0, 0.0
    x1_0, x2_0, v1_0, v2_0 = 1.0, 0.5, 0.0, 0.0
    t_end, n = 5.0, 401
    t, x1_n, x2_n, _, _ = _solve(
        x1_0, x2_0, v1_0, v2_0, m1, m2, k1, k2, kc, t_end=t_end, n=n)
    # MEC-010 解析解
    omega1 = np.sqrt(k1 / m1)
    omega2 = np.sqrt(k2 / m2)
    x1_exp = x1_0 * np.cos(omega1 * t) + (v1_0 / omega1) * np.sin(omega1 * t)
    x2_exp = x2_0 * np.cos(omega2 * t) + (v2_0 / omega2) * np.sin(omega2 * t)
    err_x1 = np.max(np.abs(x1_n - x1_exp))
    err_x2 = np.max(np.abs(x2_n - x2_exp))
    assert err_x1 < TOL, f"无耦合 x1 误差 {err_x1:.3e}（未退化为 MEC-010）"
    assert err_x2 < TOL, f"无耦合 x2 误差 {err_x2:.3e}（未退化为 MEC-010）"


def test_energy_conserved():
    """总机械能应守恒。"""
    m1, m2, k1, k2, kc = 2.0, 1.5, 3.0, 2.0, 0.8
    x1_0, x2_0, v1_0, v2_0 = 1.5, -0.5, 0.3, 0.4
    t_end, n = 10.0, 801
    t, x1_n, x2_n, v1_n, v2_n = _solve(
        x1_0, x2_0, v1_0, v2_0, m1, m2, k1, k2, kc, t_end=t_end, n=n)
    E_num = np.array([mechanical_energy(
        [x1_n[i], x2_n[i], v1_n[i], v2_n[i]], m1, m2, k1, k2, kc)
        for i in range(len(t))])
    E0 = mechanical_energy([x1_0, x2_0, v1_0, v2_0], m1, m2, k1, k2, kc)
    assert np.allclose(E_num, E0, atol=TOL), \
        f"机械能不守恒：波动 {np.max(np.abs(E_num - E0)):.3e}"


def test_invalid_parameters_rejected():
    """m≤0、k≤0 或 kc<0 应被拒绝。"""
    try:
        validate_parameters(m1=0.0, m2=1.0, k1=1.0, k2=1.0, kc=0.5)
        raise AssertionError("应拒绝 m1=0")
    except AssertionError as e:
        assert "m1" in str(e)

    try:
        validate_parameters(m1=1.0, m2=-1.0, k1=1.0, k2=1.0, kc=0.5)
        raise AssertionError("应拒绝 m2<0")
    except AssertionError as e:
        assert "m2" in str(e)

    try:
        validate_parameters(m1=1.0, m2=1.0, k1=0.0, k2=1.0, kc=0.5)
        raise AssertionError("应拒绝 k1=0")
    except AssertionError as e:
        assert "k1" in str(e)

    try:
        validate_parameters(m1=1.0, m2=1.0, k1=1.0, k2=1.0, kc=-0.5)
        raise AssertionError("应拒绝 kc<0")
    except AssertionError as e:
        assert "kc" in str(e)


if __name__ == "__main__":
    test_matches_analytical()
    test_normal_frequencies_symmetric()
    test_normal_frequencies_asymmetric()
    test_in_phase_mode_shape()
    test_anti_phase_mode_shape()
    test_single_mode_preserves_shape()
    test_multi_mode_superposition()
    test_no_coupling_degenerates_to_MEC010()
    test_energy_conserved()
    test_invalid_parameters_rejected()
    print("OK: MEC-014 数值解与解析解一致")
