"""MEC-020 —— 一致性测试：数值解 vs 解析解。

验证：
- 恒力解析解与数值积分的一致性
- 无外力退化为 MEC-001（自由质点）
- 恒力退化为 MEC-002（受力质点，2D 形式）
- 重力场退化为 MEC-003（抛体运动）
- 无外力时动量守恒
- 非法参数处理

运行方法（在本文件所在目录执行）：
    python test_MEC020_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, momentum

TOL = 1e-6


def _solve(x0=0.0, y0=0.0, vx0=10.0, vy0=15.0,
           m=1.0, Fx=0.0, Fy=0.0,
           t_end=5.0, n=401):
    """小工具：跑一次数值积分，返回 (t, x, y, vx, vy)。"""
    initial_state = np.array([x0, y0, vx0, vy0], dtype=float)
    validate_parameters(m=m, Fx=Fx, Fy=Fy)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(m, Fx, Fy),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def test_constant_force_matches_analytical():
    """恒力下数值解应与解析解一致。"""
    m, Fx, Fy = 2.0, 5.0, -9.81 * 2
    x0, y0, vx0, vy0 = 1.0, 2.0, 3.0, 4.0
    t_end, n = 5.0, 401
    t, x_n, y_n, vx_n, vy_n = _solve(
        x0, y0, vx0, vy0, m=m, Fx=Fx, Fy=Fy, t_end=t_end, n=n)
    x_a, y_a, vx_a, vy_a = analytical(
        t, [x0, y0, vx0, vy0], m=m, Fx=Fx, Fy=Fy)
    err_x = np.max(np.abs(x_n - x_a))
    err_y = np.max(np.abs(y_n - y_a))
    err_vx = np.max(np.abs(vx_n - vx_a))
    err_vy = np.max(np.abs(vy_n - vy_a))
    assert err_x < TOL, f"x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_y < TOL, f"y 误差 {err_y:.3e} 超出容差 {TOL}"
    assert err_vx < TOL, f"vx 误差 {err_vx:.3e} 超出容差 {TOL}"
    assert err_vy < TOL, f"vy 误差 {err_vy:.3e} 超出容差 {TOL}"


def test_no_force_degenerates_to_MEC001():
    """无外力（Fx=0, Fy=0）应退化为 MEC-001（自由质点，匀速直线运动）。"""
    m = 1.0
    x0, y0, vx0, vy0 = 1.0, 2.0, 3.0, 4.0
    t_end, n = 5.0, 201
    t, x_n, y_n, vx_n, vy_n = _solve(
        x0, y0, vx0, vy0, m=m, Fx=0.0, Fy=0.0, t_end=t_end, n=n)
    # MEC-001 解析解：x = x0 + vx0*t, y = y0 + vy0*t, v = const
    x_exp = x0 + vx0 * t
    y_exp = y0 + vy0 * t
    err_x = np.max(np.abs(x_n - x_exp))
    err_y = np.max(np.abs(y_n - y_exp))
    err_vx = np.max(np.abs(vx_n - vx0))
    err_vy = np.max(np.abs(vy_n - vy0))
    assert err_x < TOL, f"无外力 x 误差 {err_x:.3e}（未退化为 MEC-001）"
    assert err_y < TOL, f"无外力 y 误差 {err_y:.3e}（未退化为 MEC-001）"
    assert err_vx < TOL, f"无外力 vx 误差 {err_vx:.3e}（未退化为 MEC-001）"
    assert err_vy < TOL, f"无外力 vy 误差 {err_vy:.3e}（未退化为 MEC-001）"


def test_constant_force_degenerates_to_MEC002():
    """恒力应退化为 MEC-002（受力质点，匀加速，2D 形式）。"""
    m, F = 1.0, 3.0
    x0, y0, vx0, vy0 = 0.0, 0.0, 0.0, 0.0
    t_end, n = 5.0, 201
    t, x_n, y_n, vx_n, vy_n = _solve(
        x0, y0, vx0, vy0, m=m, Fx=F, Fy=0.0, t_end=t_end, n=n)
    # MEC-002 解析解：x = 0.5*(F/m)*t², vx = (F/m)*t, y=0, vy=0
    a = F / m
    x_exp = 0.5 * a * t ** 2
    vx_exp = a * t
    err_x = np.max(np.abs(x_n - x_exp))
    err_vx = np.max(np.abs(vx_n - vx_exp))
    err_y = np.max(np.abs(y_n))
    err_vy = np.max(np.abs(vy_n))
    assert err_x < TOL, f"恒力 x 误差 {err_x:.3e}（未退化为 MEC-002）"
    assert err_vx < TOL, f"恒力 vx 误差 {err_vx:.3e}（未退化为 MEC-002）"
    assert err_y < TOL, f"恒力 y 不为零（未退化为 MEC-002）"
    assert err_vy < TOL, f"恒力 vy 不为零（未退化为 MEC-002）"


def test_gravity_degenerates_to_MEC003():
    """重力场（Fx=0, Fy=-mg）应退化为 MEC-003（抛体运动）。"""
    g = 9.81
    m = 2.0
    x0, y0, vx0, vy0 = 0.0, 10.0, 10.0, 15.0
    t_end, n = 3.0, 201
    t, x_n, y_n, vx_n, vy_n = _solve(
        x0, y0, vx0, vy0, m=m, Fx=0.0, Fy=-m*g, t_end=t_end, n=n)
    # MEC-003 解析解
    x_exp = x0 + vx0 * t
    y_exp = y0 + vy0 * t - 0.5 * g * t ** 2
    vx_exp = np.full_like(t, vx0)
    vy_exp = vy0 - g * t
    err_x = np.max(np.abs(x_n - x_exp))
    err_y = np.max(np.abs(y_n - y_exp))
    err_vx = np.max(np.abs(vx_n - vx_exp))
    err_vy = np.max(np.abs(vy_n - vy_exp))
    assert err_x < TOL, f"重力 x 误差 {err_x:.3e}（未退化为 MEC-003）"
    assert err_y < TOL, f"重力 y 误差 {err_y:.3e}（未退化为 MEC-003）"
    assert err_vx < TOL, f"重力 vx 误差 {err_vx:.3e}（未退化为 MEC-003）"
    assert err_vy < TOL, f"重力 vy 误差 {err_vy:.3e}（未退化为 MEC-003）"


def test_momentum_conserved_no_force():
    """无外力时动量应守恒。"""
    m = 2.5
    x0, y0, vx0, vy0 = 1.0, 2.0, 3.0, -4.0
    t_end, n = 5.0, 201
    t, x_n, y_n, vx_n, vy_n = _solve(
        x0, y0, vx0, vy0, m=m, Fx=0.0, Fy=0.0, t_end=t_end, n=n)
    P_num = np.array([momentum(
        [x_n[i], y_n[i], vx_n[i], vy_n[i]], m=m)
        for i in range(len(t))])
    P0 = momentum([x0, y0, vx0, vy0], m=m)
    assert np.allclose(P_num, P0, atol=TOL), \
        f"动量不守恒：波动 {np.max(np.abs(P_num - P0)):.3e}"


def test_invalid_parameters_rejected():
    """m ≤ 0 应被拒绝。"""
    try:
        validate_parameters(m=0.0, Fx=0.0, Fy=0.0)
        raise AssertionError("应拒绝 m=0")
    except AssertionError as e:
        assert "m" in str(e)

    try:
        validate_parameters(m=-1.0, Fx=0.0, Fy=0.0)
        raise AssertionError("应拒绝 m<0")
    except AssertionError as e:
        assert "m" in str(e)


if __name__ == "__main__":
    test_constant_force_matches_analytical()
    test_no_force_degenerates_to_MEC001()
    test_constant_force_degenerates_to_MEC002()
    test_gravity_degenerates_to_MEC003()
    test_momentum_conserved_no_force()
    test_invalid_parameters_rejected()
    print("OK: MEC-020 数值解与解析解一致")
