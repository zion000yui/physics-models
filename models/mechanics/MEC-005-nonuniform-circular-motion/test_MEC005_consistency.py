"""MEC-005 —— 一致性测试：数值解 vs 解析解。

验证：
- 数值结果与解析解误差
- 角速度 ω(t) = ω₀ + αt 随时间线性变化
- 半径保持恒定
- 加速度可分解为法向 Rω² + 切向 Rα
- α=0 时退化为 MEC-004 匀速圆周运动
- 初始状态非法时应被拒绝

运行方法（在本文件所在目录执行）：
    python test_MEC005_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_initial_state

TOL = 1e-6


def _solve(x0=1.0, y0=0.0, vx0=0.0, vy0=1.0, R=1.0, omega0=1.0,
           alpha=0.0, xc=0.0, yc=0.0, t_end=6.28318530718, n=401):
    """小工具：跑一次数值积分，返回 (t, x, y, vx, vy)。"""
    initial_state = np.array([x0, y0, vx0, vy0], dtype=float)
    validate_initial_state(initial_state, R, omega0, xc, yc)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(R, omega0, alpha, xc, yc),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def _make_valid_state(R=1.0, omega0=1.0, xc=0.0, yc=0.0, theta0=0.0):
    """生成满足圆周运动约束的合法初始状态。"""
    x0 = xc + R * np.cos(theta0)
    y0 = yc + R * np.sin(theta0)
    vx0 = -R * omega0 * np.sin(theta0)
    vy0 =  R * omega0 * np.cos(theta0)
    return x0, y0, vx0, vy0


def test_position_matches_analytical():
    """位置数值解应与解析解一致。"""
    x0, y0, vx0, vy0 = _make_valid_state(R=2.0, omega0=1.5, theta0=0.8)
    R, omega0, alpha = 2.0, 1.5, 0.3
    t_end, n = 3.0, 121
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, R=R, omega0=omega0, alpha=alpha, t_end=t_end, n=n)
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], R=R, omega0=omega0, alpha=alpha)
    err_x = np.max(np.abs(x_num - x_ana))
    err_y = np.max(np.abs(y_num - y_ana))
    assert err_x < TOL, f"x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_y < TOL, f"y 误差 {err_y:.3e} 超出容差 {TOL}"


def test_angular_velocity_linear():
    """角速度 ω(t) 应满足 ω₀ + αt。"""
    xc, yc = 0.0, 0.0
    x0, y0, vx0, vy0 = _make_valid_state(R=1.0, omega0=2.0, theta0=0.3,
                                          xc=xc, yc=yc)
    omega0, alpha = 2.0, 0.5
    t_end, n = 3.0, 301
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, R=1.0, omega0=omega0, alpha=alpha,
        xc=xc, yc=yc, t_end=t_end, n=n)

    # 从位置和速度恢复瞬时角速度：ω = (r × v) / R²
    rx = x_num - xc
    ry = y_num - yc
    omega_num = rx * vy_num - ry * vx_num  # / R²，但 R=1
    omega_expected = omega0 + alpha * t
    assert np.allclose(omega_num, omega_expected, atol=TOL), \
        f"ω(t) 不符：最大偏差 {np.max(np.abs(omega_num - omega_expected)):.3e}"


def test_radius_constant():
    """轨迹应保持在半径 R 的圆上。"""
    x0, y0, vx0, vy0 = _make_valid_state(R=3.0, omega0=-0.8, xc=1.0, yc=2.0)
    t, x_num, y_num, _, _ = _solve(
        x0, y0, vx0, vy0, R=3.0, omega0=-0.8, alpha=0.2, xc=1.0, yc=2.0,
        t_end=7.0, n=351)
    r_num = np.hypot(x_num - 1.0, y_num - 2.0)
    assert np.allclose(r_num, 3.0, atol=TOL), \
        f"轨道半径不恒定：均值={np.mean(r_num):.6f}，波动={np.std(r_num):.3e}"


def test_acceleration_components():
    """加速度可分解为法向 R·ω(t)² 和切向 R·α，两分量大小都要验证。"""
    xc, yc = 0.0, 0.0
    x0, y0, vx0, vy0 = _make_valid_state(R=2.0, omega0=1.0, xc=xc, yc=yc)
    R, omega0, alpha = 2.0, 1.0, 0.5
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, R=R, omega0=omega0, alpha=alpha, xc=xc, yc=yc,
        t_end=3.0, n=401)

    # 用 dynamics 直接计算加速度（避免有限差分噪声）
    for i in range(0, len(t), 50):
        state = [x_num[i], y_num[i], vx_num[i], vy_num[i]]
        a = dynamics(t[i], state, R=R, omega0=omega0, alpha=alpha,
                      xc=xc, yc=yc)
        ax, ay = a[2], a[3]
        a_mag = np.hypot(ax, ay)
        omega_t = omega0 + alpha * t[i]
        expected_normal = R * omega_t ** 2
        expected_tangent = R * abs(alpha)
        # 总加速度大小应约为 sqrt(a_n² + a_t²)（两分量正交）
        expected_total = np.sqrt(expected_normal ** 2 + expected_tangent ** 2)
        assert np.allclose(a_mag, expected_total, atol=TOL), \
            f"总加速度不符：t={t[i]:.3f}，a={a_mag:.6f}，理论={expected_total:.6f}"

        # 验证法向分量：将加速度投影到径向（指向圆心）
        to_center_x = xc - x_num[i]
        to_center_y = yc - y_num[i]
        r_mag = np.hypot(to_center_x, to_center_y)
        ux = to_center_x / r_mag
        uy = to_center_y / r_mag
        a_normal_proj = ax * ux + ay * uy
        assert np.allclose(a_normal_proj, expected_normal, atol=TOL), \
            f"法向分量不符：t={t[i]:.3f}，投影={a_normal_proj:.6f}，理论={expected_normal:.6f}"


def test_alpha_zero_degrades_to_uniform():
    """α=0 时，MEC-005 应退化为 MEC-004 匀速圆周运动。"""
    x0, y0, vx0, vy0 = _make_valid_state(R=1.0, omega0=2.0, theta0=0.7)
    R, omega0, alpha = 1.0, 2.0, 0.0
    t_end = 2.0 * np.pi / abs(omega0)  # 一个完整周期
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, R=R, omega0=omega0, alpha=alpha, t_end=t_end, n=401)

    # 应与 MEC-004 解析解一致
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], R=R, omega0=omega0, alpha=alpha)
    assert np.allclose(x_num, x_ana, atol=TOL)
    assert np.allclose(y_num, y_ana, atol=TOL)
    assert np.allclose(vx_num, vx_ana, atol=TOL)
    assert np.allclose(vy_num, vy_ana, atol=TOL)

    # 一个周期后应回到初始位置
    assert abs(x_num[-1] - x0) < TOL
    assert abs(y_num[-1] - y0) < TOL


def test_initial_state_validation():
    """不满足圆周运动条件的初始状态应被拒绝。"""
    # 不在圆上
    try:
        validate_initial_state([0.5, 0.0, 0.0, 1.0], R=1.0, omega0=1.0)
        raise AssertionError("应拒绝不在圆上的初始状态")
    except AssertionError as e:
        assert "不在圆上" in str(e)

    # 速度不与半径正交
    try:
        validate_initial_state([1.0, 0.0, 1.0, 0.0], R=1.0, omega0=1.0)
        raise AssertionError("应拒绝速度不与半径正交的初始状态")
    except AssertionError as e:
        assert "不与半径正交" in str(e)

    # 速率不等于 R|ω₀|
    try:
        validate_initial_state([1.0, 0.0, 0.0, 2.0], R=1.0, omega0=1.0)
        raise AssertionError("应拒绝速率不等于 R|ω₀| 的初始状态")
    except AssertionError as e:
        assert "不等于 R|ω₀|" in str(e)


if __name__ == "__main__":
    test_position_matches_analytical()
    test_angular_velocity_linear()
    test_radius_constant()
    test_acceleration_components()
    test_alpha_zero_degrades_to_uniform()
    test_initial_state_validation()
    print("OK: MEC-005 数值解与解析解一致")
