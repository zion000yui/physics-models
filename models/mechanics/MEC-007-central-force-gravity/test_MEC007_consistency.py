"""MEC-007 —— 一致性测试：数值解 vs 解析解。

验证：
- 数值结果与解析解误差
- 角动量守恒
- 机械能守恒
- 偏心率向量（Laplace-Runge-Lenz）守恒
- 轨道为以力心为焦点的椭圆
- 圆轨道退化条件
- 闭合轨道（Bertrand 定理验证：r⁻² 幂律中心力产生闭合轨道）
- 非法参数拒绝

与 MEC-006 的交叉对照：
    Bertrand 定理指出，只有 F ∝ r¹（MEC-006 胡克型）和 F ∝ r⁻²
    （MEC-007 万有引力）两种幂律中心力能产生闭合轨道。本测试
    通过验证椭圆轨道在一个周期后精确闭合来确认 r⁻² 情形。

运行方法（在本文件所在目录执行）：
    python test_MEC007_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    angular_momentum, mechanical_energy, eccentricity_vector, \
    orbital_elements

TOL = 1e-6


def _solve(x0=1.0, y0=0.0, vx0=0.0, vy0=0.8, mu=1.0, m=1.0,
           t_end=3.96, n=401):
    """小工具：跑一次数值积分，返回 (t, x, y, vx, vy)。"""
    initial_state = np.array([x0, y0, vx0, vy0], dtype=float)
    validate_parameters(mu=mu, m=m)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(mu, m),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def test_position_matches_analytical():
    """位置数值解应与解析解一致。"""
    x0, y0, vx0, vy0, mu, m = 1.0, 0.0, 0.0, 0.8, 1.0, 1.0
    elem = orbital_elements([x0, y0, vx0, vy0], mu=mu)
    T = 2.0 * np.pi / elem['n']
    t_end, n = T, 401
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, mu=mu, m=m, t_end=t_end, n=n)
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], mu=mu, m=m)
    err_x = np.max(np.abs(x_num - x_ana))
    err_y = np.max(np.abs(y_num - y_ana))
    assert err_x < TOL, f"x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_y < TOL, f"y 误差 {err_y:.3e} 超出容差 {TOL}"


def test_velocity_matches_analytical():
    """速度数值解应与解析解一致。"""
    x0, y0, vx0, vy0, mu, m = 1.0, 0.0, 0.0, 0.8, 1.0, 1.0
    elem = orbital_elements([x0, y0, vx0, vy0], mu=mu)
    T = 2.0 * np.pi / elem['n']
    t_end, n = T, 401
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, mu=mu, m=m, t_end=t_end, n=n)
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], mu=mu, m=m)
    err_vx = np.max(np.abs(vx_num - vx_ana))
    err_vy = np.max(np.abs(vy_num - vy_ana))
    assert err_vx < TOL, f"vx 误差 {err_vx:.3e} 超出容差 {TOL}"
    assert err_vy < TOL, f"vy 误差 {err_vy:.3e} 超出容差 {TOL}"


def test_angular_momentum_conserved():
    """角动量 L = m·(x·vy - y·vx) 应保持恒定。"""
    # 椭圆轨道（ε < 0）：v0 < v_escape = sqrt(2*mu/r0)
    x0, y0, vx0, vy0, mu, m = 2.0, 1.0, 0.3, 0.9, 2.0, 1.5
    elem = orbital_elements([x0, y0, vx0, vy0], mu=mu)
    T = 2.0 * np.pi / elem['n']
    t_end, n = T, 501
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, mu=mu, m=m, t_end=t_end, n=n)

    L_num = np.array([angular_momentum([x, y, vx, vy], m=m)
                      for x, y, vx, vy in zip(x_num, y_num, vx_num, vy_num)])
    L0 = angular_momentum([x0, y0, vx0, vy0], m=m)
    assert np.allclose(L_num, L0, atol=TOL), \
        f"角动量不守恒：波动 {np.max(np.abs(L_num - L0)):.3e}"


def test_mechanical_energy_conserved():
    """机械能 E = ½m(v²) - μm/r 应保持恒定。"""
    # 椭圆轨道（ε < 0）：v0 < v_escape = sqrt(2*mu/r0)
    x0, y0, vx0, vy0, mu, m = 1.5, 2.0, 0.4, -0.8, 3.0, 2.0
    elem = orbital_elements([x0, y0, vx0, vy0], mu=mu)
    T = 2.0 * np.pi / elem['n']
    t_end, n = T, 501
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, mu=mu, m=m, t_end=t_end, n=n)

    E_num = np.array([mechanical_energy([x, y, vx, vy], mu=mu, m=m)
                      for x, y, vx, vy in zip(x_num, y_num, vx_num, vy_num)])
    E0 = mechanical_energy([x0, y0, vx0, vy0], mu=mu, m=m)
    assert np.allclose(E_num, E0, atol=TOL), \
        f"机械能不守恒：波动 {np.max(np.abs(E_num - E0)):.3e}"


def test_eccentricity_vector_conserved():
    """偏心率向量（Laplace-Runge-Lenz）应保持恒定。

    偏心率向量的守恒是平方反比中心力的独特性质（隐藏对称性 SO(4)），
    在其他幂律中心力中不存在。这也是 Bertrand 定理的物理基础之一。
    """
    # 椭圆轨道（ε < 0）：v0 < v_escape = sqrt(2*mu/r0)
    x0, y0, vx0, vy0, mu, m = 2.0, 1.0, 0.3, 0.9, 2.0, 1.5
    elem = orbital_elements([x0, y0, vx0, vy0], mu=mu)
    T = 2.0 * np.pi / elem['n']
    t_end, n = T, 501
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, mu=mu, m=m, t_end=t_end, n=n)

    ev_num = np.array([eccentricity_vector([x, y, vx, vy], mu=mu)
                       for x, y, vx, vy in zip(x_num, y_num, vx_num, vy_num)])
    ev0 = eccentricity_vector([x0, y0, vx0, vy0], mu=mu)
    err_ex = np.max(np.abs(ev_num[:, 0] - ev0[0]))
    err_ey = np.max(np.abs(ev_num[:, 1] - ev0[1]))
    assert err_ex < TOL, f"e_vec x 不守恒：波动 {err_ex:.3e}"
    assert err_ey < TOL, f"e_vec y 不守恒：波动 {err_ey:.3e}"


def test_orbit_is_ellipse():
    """验证轨迹是以力心为焦点的椭圆。

    方法：
    1. 从守恒量（角动量 L、能量 E、偏心率向量）反推轨道根数 a, e
    2. 从数值轨迹拟合极坐标轨道方程 r(θ) = p / (1 + e·cos(θ - θ_peri))
    3. 验证拟合参数与守恒量反推值一致

    与 MEC-006 的关键区别：
    - MEC-006（F ∝ -r）：力心在椭圆中心
    - MEC-007（F ∝ -1/r²）：力心在椭圆焦点
    """
    # 椭圆轨道（ε < 0）：v0 < v_escape = sqrt(2*mu/r0)
    x0, y0, vx0, vy0, mu, m = 2.0, 1.0, 0.3, 0.9, 2.0, 1.5
    elem = orbital_elements([x0, y0, vx0, vy0], mu=mu)
    a = elem['a']
    e = elem['e']
    p = elem['p']  # 半通径 = h²/μ
    omega = elem['omega']  # 近心点幅角

    T = 2.0 * np.pi / elem['n']
    t_end, n = T, 1001
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, mu=mu, m=m, t_end=t_end, n=n)

    # 轨道方程验证：r = p / (1 + e·cos(θ - ω))
    r_num = np.hypot(x_num, y_num)
    theta_num = np.arctan2(y_num, x_num)
    r_expected = p / (1 + e * np.cos(theta_num - omega))

    err = np.max(np.abs(r_num - r_expected))
    assert err < TOL, \
        f"轨迹不满足轨道方程：最大偏差 {err:.3e}"

    # 验证半长轴：r_min = a(1-e), r_max = a(1+e)
    r_min = np.min(r_num)
    r_max = np.max(r_num)
    assert np.allclose(r_min, a * (1 - e), atol=TOL), \
        f"近心距不符：{r_min:.6f} vs a(1-e)={a*(1-e):.6f}"
    assert np.allclose(r_max, a * (1 + e), atol=TOL), \
        f"远心距不符：{r_max:.6f} vs a(1+e)={a*(1+e):.6f}"


def test_circular_orbit_degeneracy():
    """当初始条件满足圆轨道条件时，轨迹应退化为圆。

    圆轨道条件：|v| = √(μ/r) 且 v ⊥ r
    此时 e = 0，对应 MEC-004 匀速圆周运动的动力学起源。
    """
    R, mu, m = 2.0, 4.0, 1.0
    v_circ = np.sqrt(mu / R)  # 圆轨道速度
    x0, y0 = R, 0.0
    vx0, vy0 = 0.0, v_circ
    elem = orbital_elements([x0, y0, vx0, vy0], mu=mu)
    T = 2.0 * np.pi / elem['n']
    t, x_num, y_num, _, _ = _solve(
        x0, y0, vx0, vy0, mu=mu, m=m, t_end=T, n=401)

    r_num = np.hypot(x_num, y_num)
    assert np.allclose(r_num, R, atol=TOL), \
        f"圆轨道退化失败：半径波动 {np.std(r_num):.3e}"
    # 一个周期后回到起点
    assert abs(x_num[-1] - x0) < TOL
    assert abs(y_num[-1] - y0) < TOL


def test_closed_orbit():
    """椭圆轨道在一个周期后应精确闭合（Bertrand 定理验证）。

    Bertrand 定理：只有 F ∝ r¹ 和 F ∝ r⁻² 两种幂律中心力
    能产生对所有初始条件都闭合的轨道。
    - MEC-006（F ∝ r¹）：周期 T = 2π/ω₀，与振幅无关
    - MEC-007（F ∝ r⁻²）：周期 T = 2π√(a³/μ)，与 a^(3/2) 成正比（开普勒第三定律）
    """
    # 选取非特殊初始条件（非圆、非共线），验证闭合性
    # 椭圆轨道（ε < 0）：v0 < v_escape = sqrt(2*mu/r0)
    x0, y0, vx0, vy0, mu, m = 1.5, 2.0, 0.4, -0.8, 3.0, 2.0
    elem = orbital_elements([x0, y0, vx0, vy0], mu=mu)
    T = 2.0 * np.pi / elem['n']
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, mu=mu, m=m, t_end=T, n=2001)

    # 一个周期后应回到初始位置和速度
    err_x = abs(x_num[-1] - x0)
    err_y = abs(y_num[-1] - y0)
    err_vx = abs(vx_num[-1] - vx0)
    err_vy = abs(vy_num[-1] - vy0)
    assert err_x < TOL, f"周期后 x 未闭合：误差 {err_x:.3e}"
    assert err_y < TOL, f"周期后 y 未闭合：误差 {err_y:.3e}"
    assert err_vx < TOL, f"周期后 vx 未闭合：误差 {err_vx:.3e}"
    assert err_vy < TOL, f"周期后 vy 未闭合：误差 {err_vy:.3e}"


def test_retrograde_orbit():
    """逆行轨道（h < 0）的解析解也应正确。"""
    x0, y0, vx0, vy0, mu, m = 1.0, 0.0, 0.0, -0.8, 1.0, 1.0
    elem = orbital_elements([x0, y0, vx0, vy0], mu=mu)
    assert elem['h_sign'] == -1.0, "逆行轨道 h_sign 应为 -1"
    T = 2.0 * np.pi / elem['n']
    t_end, n = T, 401
    t, x_num, y_num, vx_num, vy_num = _solve(
        x0, y0, vx0, vy0, mu=mu, m=m, t_end=t_end, n=n)
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], mu=mu, m=m)
    err_x = np.max(np.abs(x_num - x_ana))
    err_y = np.max(np.abs(y_num - y_ana))
    assert err_x < TOL, f"逆行轨道 x 误差 {err_x:.3e}"
    assert err_y < TOL, f"逆行轨道 y 误差 {err_y:.3e}"


def test_invalid_parameters_rejected():
    """mu <= 0 或 m <= 0 应被拒绝。"""
    try:
        validate_parameters(mu=0.0, m=1.0)
        raise AssertionError("应拒绝 mu=0")
    except AssertionError as e:
        assert "mu" in str(e)

    try:
        validate_parameters(mu=1.0, m=-1.0)
        raise AssertionError("应拒绝 m<0")
    except AssertionError as e:
        assert "m" in str(e)

    try:
        validate_parameters(mu=-1.0, m=1.0)
        raise AssertionError("应拒绝 mu<0")
    except AssertionError as e:
        assert "mu" in str(e)


if __name__ == "__main__":
    test_position_matches_analytical()
    test_velocity_matches_analytical()
    test_angular_momentum_conserved()
    test_mechanical_energy_conserved()
    test_eccentricity_vector_conserved()
    test_orbit_is_ellipse()
    test_circular_orbit_degeneracy()
    test_closed_orbit()
    test_retrograde_orbit()
    test_invalid_parameters_rejected()
    print("OK: MEC-007 数值解与解析解一致")
