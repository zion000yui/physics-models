"""MEC-022 —— 一致性测试：数值解 vs 解析解。

验证：
- 恒力+恒力臂解析解与数值积分的一致性
- 力矩公式 tau = rx*Fy - ry*Fx 的正确性
- 力通过质心时 tau=0，退化为 MEC-020
- 无外力+v_cm0=0 时退化为 MEC-021
- 同一外力同时产生质心加速度和角加速度
- 无外力时动量守恒
- 无外力矩时角动量守恒
- 保守力时机械能守恒
- 非法参数处理

运行方法（在本文件所在目录执行）：
    python test_MEC022_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (analytical, dynamics, validate_parameters,
                   torque_from_force, momentum, angular_momentum,
                   mechanical_energy)

TOL = 1e-6


def _solve(x0=0.0, y0=0.0, vx0=0.0, vy0=0.0,
           theta0=0.0, omega0=0.0,
           m=1.0, I=1.0, Fx=0.0, Fy=4.0, rx=0.5, ry=0.0,
           t_end=3.0, n=401):
    """小工具：跑一次数值积分。"""
    initial_state = np.array([x0, y0, vx0, vy0, theta0, omega0],
                              dtype=float)
    validate_parameters(m=m, I=I)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(m, I, Fx, Fy, rx, ry),
                    rtol=1e-9, atol=1e-12)
    return (t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3],
            sol.y[4], sol.y[5])


def test_matches_analytical():
    """恒力+恒力臂下数值解应与解析解一致。"""
    m, I = 1.0, 2.0
    Fx, Fy, rx, ry = 3.0, 4.0, 0.5, 0.2
    x0, y0, vx0, vy0 = 1.0, 2.0, 0.5, -0.3
    theta0, omega0 = 0.3, 0.8
    t_end, n = 3.0, 401
    (t, x_n, y_n, vx_n, vy_n, th_n, w_n) = _solve(
        x0, y0, vx0, vy0, theta0, omega0,
        m, I, Fx, Fy, rx, ry, t_end, n)
    x_a, y_a, vx_a, vy_a, th_a, w_a = analytical(
        t, [x0, y0, vx0, vy0, theta0, omega0],
        m, I, Fx, Fy, rx, ry)
    err_x = np.max(np.abs(x_n - x_a))
    err_y = np.max(np.abs(y_n - y_a))
    err_th = np.max(np.abs(th_n - th_a))
    err_w = np.max(np.abs(w_n - w_a))
    assert err_x < TOL, f"x 误差 {err_x:.3e}"
    assert err_y < TOL, f"y 误差 {err_y:.3e}"
    assert err_th < TOL, f"θ 误差 {err_th:.3e}"
    assert err_w < TOL, f"ω 误差 {err_w:.3e}"


def test_torque_formula():
    """力矩公式 tau = rx*Fy - ry*Fx 应正确。"""
    # 一般情况
    tau = torque_from_force(Fx=3.0, Fy=4.0, rx=0.5, ry=0.2)
    assert np.isclose(tau, 0.5 * 4 - 0.2 * 3), f"力矩计算错误: {tau}"
    # 力通过质心
    tau0 = torque_from_force(Fx=3.0, Fy=4.0, rx=0.0, ry=0.0)
    assert tau0 == 0.0, f"力通过质心时力矩应为零: {tau0}"
    # 纯 y 方向力，x 偏移
    tau_pure = torque_from_force(Fx=0.0, Fy=4.0, rx=0.5, ry=0.0)
    assert np.isclose(tau_pure, 2.0), f"纯 y 力力矩错误: {tau_pure}"


def test_force_through_cm_degenerates_to_MEC020():
    """力通过质心（rx=ry=0）时 tau=0，无旋转，退化为 MEC-020。"""
    m, I = 2.0, 3.0
    Fx, Fy, rx, ry = 5.0, -9.81 * 2, 0.0, 0.0
    x0, y0, vx0, vy0 = 1.0, 2.0, 3.0, 4.0
    theta0, omega0 = 0.5, 0.0
    t_end, n = 3.0, 201
    (t, x_n, y_n, vx_n, vy_n, th_n, w_n) = _solve(
        x0, y0, vx0, vy0, theta0, omega0,
        m, I, Fx, Fy, rx, ry, t_end, n)
    # MEC-020 解析解
    ax = Fx / m
    ay = Fy / m
    x_exp = x0 + vx0 * t + 0.5 * ax * t ** 2
    y_exp = y0 + vy0 * t + 0.5 * ay * t ** 2
    err_x = np.max(np.abs(x_n - x_exp))
    err_y = np.max(np.abs(y_n - y_exp))
    err_th = np.max(np.abs(th_n - theta0))
    err_w = np.max(np.abs(w_n))
    assert err_x < TOL, f"力通过质心 x 误差 {err_x:.3e}（未退化为 MEC-020）"
    assert err_y < TOL, f"力通过质心 y 误差 {err_y:.3e}（未退化为 MEC-020）"
    assert err_th < TOL, f"theta 变化（应无旋转）"
    assert err_w < TOL, f"omega 不为零（应无旋转）"


def test_no_force_degenerates_to_MEC021():
    """无外力且 v_cm0=0 时质心不动，退化为 MEC-021 纯转动。"""
    m, I = 1.0, 2.0
    Fx, Fy, rx, ry = 0.0, 0.0, 0.0, 0.0
    theta0, omega0 = 0.5, 3.0
    t_end, n = 5.0, 201
    (t, x_n, y_n, vx_n, vy_n, th_n, w_n) = _solve(
        0, 0, 0, 0, theta0, omega0,
        m, I, Fx, Fy, rx, ry, t_end, n)
    # MEC-021: theta = theta0 + omega0*t, omega = omega0
    theta_exp = theta0 + omega0 * t
    err_th = np.max(np.abs(th_n - theta_exp))
    err_w = np.max(np.abs(w_n - omega0))
    err_x = np.max(np.abs(x_n))
    err_y = np.max(np.abs(y_n))
    assert err_th < TOL, f"无外力 θ 误差 {err_th:.3e}（未退化为 MEC-021）"
    assert err_w < TOL, f"无外力 ω 误差 {err_w:.3e}"
    assert err_x < TOL, f"质心不应移动（x）: {err_x:.3e}"
    assert err_y < TOL, f"质心不应移动（y）: {err_y:.3e}"


def test_same_force_produces_translation_and_rotation():
    """同一外力应同时产生质心加速度和角加速度。"""
    m, I = 1.0, 1.0
    # F=(0, 4), r=(0.5, 0) → a_cm=(0, 4), tau=0.5*4=2, alpha=2/1=2
    Fx, Fy, rx, ry = 0.0, 4.0, 0.5, 0.0
    d = dynamics(0, [0, 0, 0, 0, 0, 0], m, I, Fx, Fy, rx, ry)
    assert np.isclose(d[3], Fy / m), \
        f"质心加速度 y 不对: {d[3]:.4f} vs {Fy/m:.4f}"
    tau = torque_from_force(Fx, Fy, rx, ry)
    assert np.isclose(d[5], tau / I), \
        f"角加速度不对: {d[5]:.4f} vs {tau/I:.4f}"
    # 两者都非零
    assert d[3] != 0, "质心加速度为零（应有外力作用）"
    assert d[5] != 0, "角加速度为零（应有力矩）"


def test_momentum_conserved():
    """无外力时动量应守恒。"""
    m, I = 2.5, 1.5
    Fx, Fy, rx, ry = 0.0, 0.0, 0.0, 0.0
    x0, y0, vx0, vy0 = 1.0, 2.0, 3.0, -4.0
    theta0, omega0 = 0.5, 2.0
    t_end, n = 5.0, 201
    (t, x_n, y_n, vx_n, vy_n, th_n, w_n) = _solve(
        x0, y0, vx0, vy0, theta0, omega0,
        m, I, Fx, Fy, rx, ry, t_end, n)
    P_num = np.array([momentum(
        [x_n[i], y_n[i], vx_n[i], vy_n[i], th_n[i], w_n[i]], m)
        for i in range(len(t))])
    P0 = momentum([x0, y0, vx0, vy0, theta0, omega0], m)
    assert np.allclose(P_num, P0, atol=TOL), \
        f"动量不守恒：波动 {np.max(np.abs(P_num - P0)):.3e}"


def test_angular_momentum_conserved():
    """无外力矩时角动量应守恒。"""
    m, I = 2.5, 1.5
    # 无外力 → 无力矩
    Fx, Fy, rx, ry = 0.0, 0.0, 0.0, 0.0
    x0, y0, vx0, vy0 = 1.0, 2.0, 3.0, -4.0
    theta0, omega0 = 0.5, 2.0
    t_end, n = 5.0, 201
    (t, x_n, y_n, vx_n, vy_n, th_n, w_n) = _solve(
        x0, y0, vx0, vy0, theta0, omega0,
        m, I, Fx, Fy, rx, ry, t_end, n)
    L_num = np.array([angular_momentum(
        [x_n[i], y_n[i], vx_n[i], vy_n[i], th_n[i], w_n[i]], I)
        for i in range(len(t))])
    L0 = angular_momentum([x0, y0, vx0, vy0, theta0, omega0], I)
    assert np.allclose(L_num, L0, atol=TOL), \
        f"角动量不守恒：波动 {np.max(np.abs(L_num - L0)):.3e}"


def test_mechanical_energy_conserved():
    """无外力时机械能应守恒。"""
    m, I = 2.0, 3.0
    Fx, Fy, rx, ry = 0.0, 0.0, 0.0, 0.0
    x0, y0, vx0, vy0 = 1.0, 2.0, 0.5, -0.3
    theta0, omega0 = 0.3, 0.8
    t_end, n = 5.0, 201
    (t, x_n, y_n, vx_n, vy_n, th_n, w_n) = _solve(
        x0, y0, vx0, vy0, theta0, omega0,
        m, I, Fx, Fy, rx, ry, t_end, n)
    E_num = np.array([mechanical_energy(
        [x_n[i], y_n[i], vx_n[i], vy_n[i], th_n[i], w_n[i]],
        m, I, Fx, Fy, rx, ry) for i in range(len(t))])
    E0 = mechanical_energy(
        [x0, y0, vx0, vy0, theta0, omega0], m, I, Fx, Fy, rx, ry)
    assert np.allclose(E_num, E0, atol=TOL), \
        f"机械能不守恒：波动 {np.max(np.abs(E_num - E0)):.3e}"


def test_invalid_parameters_rejected():
    """m≤0 或 I≤0 应被拒绝。"""
    try:
        validate_parameters(m=0.0, I=1.0)
        raise AssertionError("应拒绝 m=0")
    except AssertionError as e:
        assert "m" in str(e)

    try:
        validate_parameters(m=1.0, I=-1.0)
        raise AssertionError("应拒绝 I<0")
    except AssertionError as e:
        assert "I" in str(e)


if __name__ == "__main__":
    test_matches_analytical()
    test_torque_formula()
    test_force_through_cm_degenerates_to_MEC020()
    test_no_force_degenerates_to_MEC021()
    test_same_force_produces_translation_and_rotation()
    test_momentum_conserved()
    test_angular_momentum_conserved()
    test_mechanical_energy_conserved()
    test_invalid_parameters_rejected()
    print("OK: MEC-022 数值解与解析解一致")
