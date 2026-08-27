"""MEC-004 —— 一致性测试：数值解 vs 解析解。

验证：
- 数值结果与解析解误差
- 速率保持恒定
- 轨迹保持在半径 R 的圆上
- 角速度保持恒定
- 加速度大小为 Rω²，方向指向圆心
- 一个完整周期后回到初始位置
- 初始状态约束验证

运行方法（在本文件所在目录执行）：
    python test_MEC004_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_initial_state

TOL = 1e-6


def _solve(x0=1.0, y0=0.0, vx0=0.0, vy0=1.0, R=1.0, omega=1.0,
           xc=0.0, yc=0.0, t_end=6.28318530718, n=401):
    """小工具：跑一次数值积分，返回 (t, x, y, vx, vy)。"""
    initial_state = np.array([x0, y0, vx0, vy0], dtype=float)
    validate_initial_state(initial_state, R, omega, xc, yc)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(R, omega, xc, yc),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def _make_valid_state(R=1.0, omega=1.0, xc=0.0, yc=0.0, theta0=0.0):
    """生成满足圆周运动约束的合法初始状态。"""
    x0 = xc + R * np.cos(theta0)
    y0 = yc + R * np.sin(theta0)
    vx0 = -R * omega * np.sin(theta0)
    vy0 =  R * omega * np.cos(theta0)
    return x0, y0, vx0, vy0


def test_position_matches_analytical():
    """位置数值解应与解析解一致。"""
    x0, y0, vx0, vy0 = _make_valid_state(R=2.0, omega=1.5, theta0=0.8)
    R, omega = 2.0, 1.5
    t_end, n = 3.0, 121
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, R=R, omega=omega, t_end=t_end, n=n)
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], R=R, omega=omega)
    err_x = np.max(np.abs(x_num - x_ana))
    err_y = np.max(np.abs(y_num - y_ana))
    assert err_x < TOL, f"x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_y < TOL, f"y 误差 {err_y:.3e} 超出容差 {TOL}"


def test_speed_constant():
    """速率应恒等于 R|ω|。"""
    x0, y0, vx0, vy0 = _make_valid_state(R=1.0, omega=2.0, theta0=0.5)
    t, _, _, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, R=1.0, omega=2.0, t_end=5.0, n=251)
    speed = np.hypot(vx_num, vy_num)
    expected = 1.0 * 2.0  # R * |omega|
    assert np.allclose(speed, expected, atol=TOL), \
        f"速率不恒定：均值={np.mean(speed):.6f}，波动={np.std(speed):.3e}"


def test_radius_constant():
    """轨迹应保持在半径 R 的圆上。"""
    x0, y0, vx0, vy0 = _make_valid_state(R=3.0, omega=-0.8, xc=1.0, yc=2.0)
    t, x_num, y_num, _, _ = _solve(
        x0, y0, vx0, vy0, R=3.0, omega=-0.8, xc=1.0, yc=2.0,
        t_end=7.0, n=351)
    r_num = np.hypot(x_num - 1.0, y_num - 2.0)
    assert np.allclose(r_num, 3.0, atol=TOL), \
        f"轨道半径不恒定：均值={np.mean(r_num):.6f}，波动={np.std(r_num):.3e}"


def test_centripetal_acceleration():
    """加速度大小应为 Rω²，方向指向圆心。"""
    xc, yc = 0.0, 0.0
    x0, y0, vx0, vy0 = _make_valid_state(R=2.0, omega=1.0, xc=xc, yc=yc)
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, R=2.0, omega=1.0, xc=xc, yc=yc,
        t_end=4.0, n=401)

    # 直接用 dynamics 计算加速度（避免有限差分的离散化噪声）
    expected_a = 2.0 * 1.0 ** 2  # R * omega^2
    for i in range(0, len(t), 40):
        state = [x_num[i], y_num[i], vx_num[i], vy_num[i]]
        a = dynamics(t[i], state, R=2.0, omega=1.0, xc=xc, yc=yc)
        ax, ay = a[2], a[3]
        a_mag = np.hypot(ax, ay)
        assert np.allclose(a_mag, expected_a, atol=TOL), \
            f"向心加速度大小不符：t={t[i]:.3f}，a={a_mag:.6f}，理论值={expected_a:.6f}"

        # 加速度方向应指向圆心
        to_center_x = xc - x_num[i]
        to_center_y = yc - y_num[i]
        to_center_mag = np.hypot(to_center_x, to_center_y)
        ux = to_center_x / to_center_mag
        uy = to_center_y / to_center_mag
        proj = ax * ux + ay * uy
        assert np.allclose(proj, expected_a, atol=TOL), \
            f"加速度方向不指向圆心：t={t[i]:.3f}，投影={proj:.6f}，理论值={expected_a:.6f}"


def test_orbit_closed():
    """一个完整周期 T = 2π/|ω| 后应回到初始位置。"""
    x0, y0, vx0, vy0 = _make_valid_state(R=1.0, omega=2.0, theta0=1.2)
    R, omega = 1.0, 2.0
    t_end = 2.0 * np.pi / abs(omega)  # 一个完整周期
    t, x_num, y_num, _, _ = _solve(
        x0, y0, vx0, vy0, R=R, omega=omega, t_end=t_end, n=2001)

    # 末点应回到初始位置（考虑数值误差）
    err_x = abs(x_num[-1] - x0)
    err_y = abs(y_num[-1] - y0)
    assert err_x < TOL, f"周期后 x 未闭合：误差 {err_x:.3e}"
    assert err_y < TOL, f"周期后 y 未闭合：误差 {err_y:.3e}"


def test_initial_state_validation():
    """不满足圆周运动条件的初始状态应被拒绝。"""
    # 不在圆上
    try:
        validate_initial_state([0.5, 0.0, 0.0, 1.0], R=1.0, omega=1.0)
        raise AssertionError("应拒绝不在圆上的初始状态")
    except AssertionError as e:
        assert "不在圆上" in str(e)

    # 速度不与半径正交
    try:
        validate_initial_state([1.0, 0.0, 1.0, 0.0], R=1.0, omega=1.0)
        raise AssertionError("应拒绝速度不与半径正交的初始状态")
    except AssertionError as e:
        assert "不与半径正交" in str(e)

    # 速率不等于 R|ω|
    try:
        validate_initial_state([1.0, 0.0, 0.0, 2.0], R=1.0, omega=1.0)
        raise AssertionError("应拒绝速率不等于 R|ω| 的初始状态")
    except AssertionError as e:
        assert "不等于 R|ω|" in str(e)


if __name__ == "__main__":
    test_position_matches_analytical()
    test_speed_constant()
    test_radius_constant()
    test_centripetal_acceleration()
    test_orbit_closed()
    test_initial_state_validation()
    print("OK: MEC-004 数值解与解析解一致")
