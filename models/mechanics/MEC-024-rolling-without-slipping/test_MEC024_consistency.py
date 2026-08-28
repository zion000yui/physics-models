"""MEC-024 —— 一致性测试：数值解 vs 解析解。

验证：
- 恒加速度解析解与数值积分的一致性
- 纯滚动约束 v_cm = R·ω 在整个轨迹中成立
- 机械能守恒（理想静摩擦不做功）
- 不同转动惯量 I 对加速度的影响
- 实心球、实心圆柱、薄壁圆筒的典型 I/(mR²) 情况
- 加速度公式 a = g·sin(α)/(1+I/(mR²)) 的正确性
- 非法参数处理

运行方法（在本文件所在目录执行）：
    python test_MEC024_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (analytical, dynamics, validate_parameters,
                   acceleration, effective_mass, mechanical_energy)

TOL = 1e-6


def _solve(x0=0.0, theta0=0.0, v0=0.0, omega0=0.0,
           m=1.0, I=1.0, R=1.0, g=9.81, alpha=0.5236,
           t_end=3.0, n=401):
    """小工具：跑一次数值积分，返回 (t, x, theta, v, omega)。"""
    initial_state = np.array([x0, theta0, v0, omega0], dtype=float)
    validate_parameters(m=m, I=I, R=R, g=g, alpha=alpha)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(m, I, R, g, alpha),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def test_matches_analytical():
    """数值解应与恒加速度解析解一致。"""
    g, alpha = 9.81, np.radians(30)
    m, R = 1.0, 0.5
    I = 0.4 * m * R ** 2  # 实心球
    x0, theta0, v0, omega0 = 0.5, 0.1, 1.0, 2.0
    t_end, n = 3.0, 401
    t, x_n, th_n, v_n, w_n = _solve(
        x0, theta0, v0, omega0, m, I, R, g, alpha, t_end, n)
    x_a, th_a, v_a, w_a = analytical(
        t, [x0, theta0, v0, omega0], m, I, R, g, alpha)
    err_x = np.max(np.abs(x_n - x_a))
    err_th = np.max(np.abs(th_n - th_a))
    err_v = np.max(np.abs(v_n - v_a))
    err_w = np.max(np.abs(w_n - w_a))
    assert err_x < TOL, f"x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_th < TOL, f"θ 误差 {err_th:.3e} 超出容差 {TOL}"
    assert err_v < TOL, f"v 误差 {err_v:.3e} 超出容差 {TOL}"
    assert err_w < TOL, f"ω 误差 {err_w:.3e} 超出容差 {TOL}"


def test_rolling_constraint():
    """纯滚动约束 v_cm = R·ω 应在整个轨迹中成立。"""
    g, alpha = 9.81, np.radians(25)
    m, R = 2.0, 0.3
    I = 0.5 * m * R ** 2  # 实心圆柱
    v0 = 0.8
    omega0 = v0 / R  # 满足约束 v = R*omega
    t, _, _, v_n, w_n = _solve(
        0, 0, v0, omega0, m, I, R, g, alpha, t_end=5.0, n=501)
    constraint_err = np.max(np.abs(v_n - R * w_n))
    assert constraint_err < TOL, \
        f"纯滚动约束不满足：|v-Rω| = {constraint_err:.3e}"


def test_energy_conserved():
    """机械能应守恒（理想静摩擦不做功）。"""
    g, alpha = 9.81, np.radians(30)
    m, R = 1.0, 0.5
    I = 0.4 * m * R ** 2  # 实心球
    t, x_n, th_n, v_n, w_n = _solve(
        0, 0, 1.0, 2.0, m, I, R, g, alpha, t_end=5.0, n=501)
    E_num = np.array([mechanical_energy(
        [x_n[i], th_n[i], v_n[i], w_n[i]], m, I, R, g, alpha)
        for i in range(len(t))])
    E0 = mechanical_energy([0, 0, 1.0, 2.0], m, I, R, g, alpha)
    assert np.allclose(E_num, E0, atol=TOL), \
        f"机械能不守恒：波动 {np.max(np.abs(E_num - E0)):.3e}"


def test_acceleration_formula():
    """加速度公式 a = g·sin(α)/(1+I/(mR²)) 应与理论值一致。"""
    g, alpha = 9.81, np.radians(30)
    m, R = 1.0, 0.5
    # 实心球 I=2/5 mR² → a = 5/7 g sinα
    I = 0.4 * m * R ** 2
    a = acceleration(g, m, I, R, alpha)
    a_theory = 5.0 / 7.0 * g * np.sin(alpha)
    assert np.isclose(a, a_theory, rtol=TOL), \
        f"实心球加速度不符：{a:.6f} vs {a_theory:.6f}"
    # 实心圆柱 I=1/2 mR² → a = 2/3 g sinα
    I = 0.5 * m * R ** 2
    a = acceleration(g, m, I, R, alpha)
    a_theory = 2.0 / 3.0 * g * np.sin(alpha)
    assert np.isclose(a, a_theory, rtol=TOL), \
        f"实心圆柱加速度不符：{a:.6f} vs {a_theory:.6f}"
    # 薄壁圆筒 I=mR² → a = 1/2 g sinα
    I = m * R ** 2
    a = acceleration(g, m, I, R, alpha)
    a_theory = 0.5 * g * np.sin(alpha)
    assert np.isclose(a, a_theory, rtol=TOL), \
        f"薄壁圆筒加速度不符：{a:.6f} vs {a_theory:.6f}"


def test_inertia_effect_on_acceleration():
    """I 增大时加速度应减小（转动惯性增大使平动减慢）。"""
    g, alpha = 9.81, np.radians(30)
    m, R = 1.0, 0.5
    I_ratios = [0.2, 0.4, 0.5, 1.0, 2.0]
    accels = []
    for ratio in I_ratios:
        I = ratio * m * R ** 2
        a = acceleration(g, m, I, R, alpha)
        accels.append(a)
    # 验证严格递减
    for i in range(len(accels) - 1):
        assert accels[i + 1] < accels[i], \
            f"加速度未随 I 递减：I_ratio={I_ratios[i]}→{I_ratios[i+1]}, a={accels[i]:.6f}→{accels[i+1]:.6f}"
    # 所有加速度都应小于 g*sin(α)（纯滑动加速度）
    g_sin = g * np.sin(alpha)
    for a in accels:
        assert a < g_sin, \
            f"纯滚动加速度 {a:.6f} 应小于纯滑动 {g_sin:.6f}"


def test_constraint_satisfied_from_nonzero_initial():
    """初始条件满足 v0=R·ω0 时，约束应在整个轨迹中成立。"""
    g, alpha = 9.81, np.radians(20)
    m, R = 1.5, 0.4
    I = 0.5 * m * R ** 2
    v0 = 0.8
    omega0 = v0 / R  # 满足约束
    t, _, _, v_n, w_n = _solve(
        0, 0, v0, omega0, m, I, R, g, alpha, t_end=5.0, n=501)
    constraint_err = np.max(np.abs(v_n - R * w_n))
    assert constraint_err < TOL, \
        f"约束不满足：{constraint_err:.3e}"


def test_invalid_parameters_rejected():
    """m≤0、I≤0、R≤0、g≤0 或 alpha<0 应被拒绝。"""
    try:
        validate_parameters(m=0.0, I=1.0, R=1.0, g=9.81, alpha=0.0)
        raise AssertionError("应拒绝 m=0")
    except AssertionError as e:
        assert "m" in str(e)

    try:
        validate_parameters(m=1.0, I=-1.0, R=1.0, g=9.81, alpha=0.0)
        raise AssertionError("应拒绝 I<0")
    except AssertionError as e:
        assert "I" in str(e)

    try:
        validate_parameters(m=1.0, I=1.0, R=0.0, g=9.81, alpha=0.0)
        raise AssertionError("应拒绝 R=0")
    except AssertionError as e:
        assert "R" in str(e)

    try:
        validate_parameters(m=1.0, I=1.0, R=1.0, g=-1.0, alpha=0.0)
        raise AssertionError("应拒绝 g<0")
    except AssertionError as e:
        assert "g" in str(e)

    try:
        validate_parameters(m=1.0, I=1.0, R=1.0, g=9.81, alpha=-0.1)
        raise AssertionError("应拒绝 alpha<0")
    except AssertionError as e:
        assert "alpha" in str(e)


if __name__ == "__main__":
    test_matches_analytical()
    test_rolling_constraint()
    test_energy_conserved()
    test_acceleration_formula()
    test_inertia_effect_on_acceleration()
    test_constraint_satisfied_from_nonzero_initial()
    test_invalid_parameters_rejected()
    print("OK: MEC-024 数值解与解析解一致")
