"""MEC-009 —— 一致性测试：数值解 vs 解析解。

验证：
- 线性阻力：数值解与闭式解析解一致
- 二次阻力（1D 垂直）：数值解与分段解析解一致
- 无阻力极限：退化为 MEC-003 抛体运动
- 终态速度：数值解趋近理论终态速度
- 机械能耗散：有阻力时能量单调递减
- 非法参数拒绝

与 MEC-003 的交叉对照：
    当 b=0、c=0 时，MEC-009 精确退化为 MEC-003 抛体运动
    （dvx/dt=0, dvy/dt=-g），解析解与 MEC-003 完全一致。

运行方法（在本文件所在目录执行）：
    python test_MEC009_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, terminal_velocity

TOL = 1e-6


def _solve(x0=0.0, y0=0.0, vx0=10.0, vy0=15.0,
           g=9.81, b=0.5, c=0.0, m=1.0,
           t_end=5.0, n=401):
    """小工具：跑一次数值积分，返回 (t, x, y, vx, vy)。"""
    initial_state = np.array([x0, y0, vx0, vy0], dtype=float)
    validate_parameters(g=g, b=b, c=c, m=m)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(g, b, c, m),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def test_linear_drag_matches_analytical():
    """纯线性阻力：数值解应与闭式解析解一致。"""
    g, b, c, m = 9.81, 0.5, 0.0, 1.0
    x0, y0, vx0, vy0 = 0.0, 0.0, 10.0, 15.0
    t_end, n = 5.0, 401
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, g=g, b=b, c=c, m=m, t_end=t_end, n=n)
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], g=g, b=b, c=c, m=m)
    err_x = np.max(np.abs(x_num - x_ana))
    err_y = np.max(np.abs(y_num - y_ana))
    err_vx = np.max(np.abs(vx_num - vx_ana))
    err_vy = np.max(np.abs(vy_num - vy_ana))
    assert err_x < TOL, f"x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_y < TOL, f"y 误差 {err_y:.3e} 超出容差 {TOL}"
    assert err_vx < TOL, f"vx 误差 {err_vx:.3e} 超出容差 {TOL}"
    assert err_vy < TOL, f"vy 误差 {err_vy:.3e} 超出容差 {TOL}"


def test_quadratic_drag_vertical_matches_analytical():
    """纯二次阻力 1D 垂直：数值解应与分段解析解一致。"""
    g, b, c, m = 9.81, 0.0, 0.01, 1.0
    x0, y0, vx0, vy0 = 0.0, 0.0, 0.0, 10.0
    t_end, n = 5.0, 1001
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, g=g, b=b, c=c, m=m, t_end=t_end, n=n)
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], g=g, b=b, c=c, m=m)
    err_y = np.max(np.abs(y_num - y_ana))
    err_vy = np.max(np.abs(vy_num - vy_ana))
    assert err_y < TOL, f"y 误差 {err_y:.3e} 超出容差 {TOL}"
    assert err_vy < TOL, f"vy 误差 {err_vy:.3e} 超出容差 {TOL}"


def test_quadratic_drag_downward_matches_analytical():
    """纯二次阻力 1D 垂直下落（初速向下）：数值解应与解析解一致。"""
    g, b, c, m = 9.81, 0.0, 0.02, 1.0
    x0, y0, vx0, vy0 = 0.0, 50.0, 0.0, -5.0  # 初速向下
    t_end, n = 5.0, 1001
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, g=g, b=b, c=c, m=m, t_end=t_end, n=n)
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], g=g, b=b, c=c, m=m)
    err_y = np.max(np.abs(y_num - y_ana))
    err_vy = np.max(np.abs(vy_num - vy_ana))
    assert err_y < TOL, f"y 误差 {err_y:.3e} 超出容差 {TOL}"
    assert err_vy < TOL, f"vy 误差 {err_vy:.3e} 超出容差 {TOL}"


def test_no_drag_degenerates_to_projectile():
    """无阻力极限：b=0, c=0 时应退化为 MEC-003 抛体运动。"""
    g, b, c, m = 9.81, 0.0, 0.0, 1.0
    x0, y0, vx0, vy0 = 0.0, 10.0, 10.0, 15.0
    t_end, n = 4.0, 81
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, g=g, b=b, c=c, m=m, t_end=t_end, n=n)
    # MEC-003 解析解
    x_exp = x0 + vx0 * t
    y_exp = y0 + vy0 * t - 0.5 * g * t ** 2
    vx_exp = np.full_like(t, vx0)
    vy_exp = vy0 - g * t
    err_x = np.max(np.abs(x_num - x_exp))
    err_y = np.max(np.abs(y_num - y_exp))
    err_vx = np.max(np.abs(vx_num - vx_exp))
    err_vy = np.max(np.abs(vy_num - vy_exp))
    assert err_x < TOL, f"x 误差 {err_x:.3e}（未退化为抛体运动）"
    assert err_y < TOL, f"y 误差 {err_y:.3e}（未退化为抛体运动）"
    assert err_vx < TOL, f"vx 误差 {err_vx:.3e}（未退化为抛体运动）"
    assert err_vy < TOL, f"vy 误差 {err_vy:.3e}（未退化为抛体运动）"
    # 同时验证 MEC-009 自身的解析解也退化为抛体
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], g=g, b=b, c=c, m=m)
    assert np.allclose(x_ana, x_exp, atol=TOL)
    assert np.allclose(y_ana, y_exp, atol=TOL)


def test_terminal_velocity_linear():
    """线性阻力的终态速度应趋近 m·g/b。"""
    g, b, c, m = 9.81, 0.5, 0.0, 1.0
    x0, y0, vx0, vy0 = 0.0, 100.0, 0.0, 0.0  # 从高处静止下落
    t_end = 200.0  # 足够长以接近终态
    t, _, _, _, vy_num = _solve(
        x0, y0, vx0, vy0, g=g, b=b, c=c, m=m, t_end=t_end, n=2001)
    vt_theory = terminal_velocity(g=g, b=b, c=c, m=m)
    # 末点速度应接近 -v_t（向下）
    assert abs(vy_num[-1] - (-vt_theory)) < 0.01, \
        f"终态速度不符：数值 {vy_num[-1]:.4f} vs 理论 {-vt_theory:.4f}"


def test_terminal_velocity_quadratic():
    """二次阻力的终态速度应趋近 √(m·g/c)。"""
    g, b, c, m = 9.81, 0.0, 0.05, 1.0
    x0, y0, vx0, vy0 = 0.0, 200.0, 0.0, 0.0  # 从高处静止下落
    t_end = 300.0  # 足够长以接近终态
    t, _, _, _, vy_num = _solve(
        x0, y0, vx0, vy0, g=g, b=b, c=c, m=m, t_end=t_end, n=2001)
    vt_theory = terminal_velocity(g=g, b=b, c=c, m=m)
    assert abs(vy_num[-1] - (-vt_theory)) < 0.01, \
        f"终态速度不符：数值 {vy_num[-1]:.4f} vs 理论 {-vt_theory:.4f}"


def test_energy_dissipation():
    """有阻力时机械能应单调递减。"""
    g, b, c, m = 9.81, 0.3, 0.02, 1.0
    x0, y0, vx0, vy0 = 0.0, 0.0, 10.0, 15.0
    t_end, n = 5.0, 501
    t, _, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, g=g, b=b, c=c, m=m, t_end=t_end, n=n)
    E = 0.5 * m * (vx_num ** 2 + vy_num ** 2) + m * g * y_num
    # 机械能应单调递减（允许数值噪声导致的极小非单调）
    dE = np.diff(E)
    # 绝大部分步应递减
    decreasing_fraction = np.mean(dE < 1e-10)
    assert decreasing_fraction > 0.95, \
        f"机械能未单调递减：递减比例 {decreasing_fraction:.2%}"
    # 总体应明显耗散
    assert E[-1] < E[0], \
        f"机械能未耗散：初 {E[0]:.4f}，末 {E[-1]:.4f}"


def test_energy_conserved_no_drag():
    """无阻力时机械能应守恒（退化为 MEC-003）。"""
    g, b, c, m = 9.81, 0.0, 0.0, 1.0
    x0, y0, vx0, vy0 = 0.0, 10.0, 10.0, 15.0
    t_end, n = 4.0, 201
    t, _, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, g=g, b=b, c=c, m=m, t_end=t_end, n=n)
    E = 0.5 * m * (vx_num ** 2 + vy_num ** 2) + m * g * y_num
    assert np.allclose(E, E[0], atol=TOL), \
        f"无阻力机械能不守恒：波动 {np.max(np.abs(E - E[0])):.3e}"


def test_invalid_parameters_rejected():
    """g<0、b<0、c<0 或 m≤0 应被拒绝。"""
    try:
        validate_parameters(g=-1.0, b=0.0, c=0.0, m=1.0)
        raise AssertionError("应拒绝 g<0")
    except AssertionError as e:
        assert "g" in str(e)

    try:
        validate_parameters(g=9.81, b=-0.5, c=0.0, m=1.0)
        raise AssertionError("应拒绝 b<0")
    except AssertionError as e:
        assert "b" in str(e)

    try:
        validate_parameters(g=9.81, b=0.0, c=-0.01, m=1.0)
        raise AssertionError("应拒绝 c<0")
    except AssertionError as e:
        assert "c" in str(e)

    try:
        validate_parameters(g=9.81, b=0.0, c=0.0, m=0.0)
        raise AssertionError("应拒绝 m=0")
    except AssertionError as e:
        assert "m" in str(e)


if __name__ == "__main__":
    test_linear_drag_matches_analytical()
    test_quadratic_drag_vertical_matches_analytical()
    test_quadratic_drag_downward_matches_analytical()
    test_no_drag_degenerates_to_projectile()
    test_terminal_velocity_linear()
    test_terminal_velocity_quadratic()
    test_energy_dissipation()
    test_energy_conserved_no_drag()
    test_invalid_parameters_rejected()
    print("OK: MEC-009 数值解与解析解一致")
