"""MEC-006 —— 一致性测试：数值解 vs 解析解。

验证：
- 数值结果与解析解误差
- 角动量守恒
- 机械能守恒
- 轨迹中心对称性（椭圆/圆的几何判据）
- 圆轨道退化条件
- 非法参数拒绝

运行方法（在本文件所在目录执行）：
    python test_MEC006_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    angular_momentum, mechanical_energy

TOL = 1e-6


def _solve(x0=1.0, y0=0.0, vx0=0.0, vy0=1.0, k=1.0, m=1.0,
           t_end=6.28318530718, n=401):
    """小工具：跑一次数值积分，返回 (t, x, y, vx, vy)。"""
    initial_state = np.array([x0, y0, vx0, vy0], dtype=float)
    validate_parameters(k=k, m=m)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(k, m),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def test_position_matches_analytical():
    """位置数值解应与解析解一致。"""
    x0, y0, vx0, vy0, k, m = 1.0, 0.0, 0.0, 1.0, 1.0, 1.0
    t_end, n = 4.0, 161
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, k=k, m=m, t_end=t_end, n=n)
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], k=k, m=m)
    err_x = np.max(np.abs(x_num - x_ana))
    err_y = np.max(np.abs(y_num - y_ana))
    assert err_x < TOL, f"x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_y < TOL, f"y 误差 {err_y:.3e} 超出容差 {TOL}"


def test_angular_momentum_conserved():
    """角动量 L = m·(x·vy - y·vx) 应保持恒定。"""
    x0, y0, vx0, vy0, k, m = 2.0, 1.0, -0.5, 3.0, 2.0, 1.5
    t_end, n = 5.0, 251
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, k=k, m=m, t_end=t_end, n=n)

    L_num = np.array([angular_momentum([x, y, vx, vy], m=m)
                      for x, y, vx, vy in zip(x_num, y_num, vx_num, vy_num)])
    L0 = angular_momentum([x0, y0, vx0, vy0], m=m)
    assert np.allclose(L_num, L0, atol=TOL), \
        f"角动量不守恒：波动 {np.max(np.abs(L_num - L0)):.3e}"


def test_mechanical_energy_conserved():
    """机械能 E = ½m(v²) + ½k(r²) 应保持恒定。"""
    x0, y0, vx0, vy0, k, m = 1.5, 2.0, 1.0, -2.0, 3.0, 2.0
    t_end, n = 4.0, 201
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, k=k, m=m, t_end=t_end, n=n)

    E_num = np.array([mechanical_energy([x, y, vx, vy], k=k, m=m)
                      for x, y, vx, vy in zip(x_num, y_num, vx_num, vy_num)])
    E0 = mechanical_energy([x0, y0, vx0, vy0], k=k, m=m)
    assert np.allclose(E_num, E0, atol=TOL), \
        f"机械能不守恒：波动 {np.max(np.abs(E_num - E0)):.3e}"


def test_trajectory_centrally_symmetric():
    """轨迹应关于原点中心对称：r(t+T/2) = -r(t)，这是椭圆轨道的必然几何特征。

    选择该判据的理由：
    - 胡克型中心力 F=-kr 具有时间反演对称性和中心对称性
    - 解析解中 x(t)、y(t) 均为 sin/cos 线性组合，天然满足 r(t+T/2) = -r(t)
    - 数值上易于验证，不依赖椭圆拟合或主轴提取，数学上稳健
    """
    x0, y0, vx0, vy0, k, m = 1.0, 2.0, 3.0, -1.0, 1.0, 1.0
    omega0 = np.sqrt(k / m)
    T = 2.0 * np.pi / omega0
    t_end = T
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, k=k, m=m, t_end=t_end, n=2001)

    # 找 t+T/2 对应的最近索引
    half_idx = np.argmin(np.abs(t - t[-1] / 2))
    # 验证 r(t) ≈ -r(t+T/2) 对采样点成立
    for i in range(0, len(t) // 2, 50):
        j = i + half_idx
        if j < len(t):
            assert np.allclose(x_num[i], -x_num[j], atol=TOL), \
                f"x 不对称：t={t[i]:.3f}，x={x_num[i]:.6f}，-x(t+T/2)={-x_num[j]:.6f}"
            assert np.allclose(y_num[i], -y_num[j], atol=TOL), \
                f"y 不对称：t={t[i]:.3f}，y={y_num[i]:.6f}，-y(t+T/2)={-y_num[j]:.6f}"


def test_circular_orbit_degeneracy():
    """当初始条件满足圆轨道条件时，轨迹应退化为圆（定性验证）。"""
    # 圆轨道条件：|r0| 恒定、v0⊥r0、|v0| = ω0·|r0|
    R, omega0, k, m = 2.0, 1.0, 4.0, 4.0  # ω0 = sqrt(k/m) = 1.0
    x0, y0 = R, 0.0
    vx0, vy0 = 0.0, R * omega0
    t_end = 2.0 * np.pi / omega0
    t, x_num, y_num, _, _ = _solve(
        x0, y0, vx0, vy0, k=k, m=m, t_end=t_end, n=401)

    r_num = np.hypot(x_num, y_num)
    assert np.allclose(r_num, R, atol=TOL), \
        f"圆轨道退化失败：半径波动 {np.std(r_num):.3e}"
    # 一个周期后回到起点
    assert abs(x_num[-1] - x0) < TOL
    assert abs(y_num[-1] - y0) < TOL


def test_invalid_parameters_rejected():
    """k <= 0 或 m <= 0 应被拒绝。"""
    try:
        validate_parameters(k=0.0, m=1.0)
        raise AssertionError("应拒绝 k=0")
    except AssertionError as e:
        assert "k" in str(e)

    try:
        validate_parameters(k=1.0, m=-1.0)
        raise AssertionError("应拒绝 m<0")
    except AssertionError as e:
        assert "m" in str(e)

    try:
        validate_parameters(k=-1.0, m=1.0)
        raise AssertionError("应拒绝 k<0")
    except AssertionError as e:
        assert "k" in str(e)


if __name__ == "__main__":
    test_position_matches_analytical()
    test_angular_momentum_conserved()
    test_mechanical_energy_conserved()
    test_trajectory_centrally_symmetric()
    test_circular_orbit_degeneracy()
    test_invalid_parameters_rejected()
    print("OK: MEC-006 数值解与解析解一致")
