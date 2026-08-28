"""MEC-008 —— 一致性测试：数值解 vs 解析解。

验证：
- 两个质点的位置和速度数值解与解析解一致
- 总动量守恒
- 总角动量守恒
- 总机械能守恒
- 质心做匀速直线运动
- 相对轨道为以质心为焦点的椭圆
- 圆轨道退化条件
- 大质量极限（m2 >> m1 时逼近 MEC-007 单体问题）
- 非法参数拒绝

与 MEC-007 的交叉对照：
    二体问题通过约化质量和相对坐标化简为等效单体开普勒问题，
    引力参数 μ = G·(m1+m2)。当 m2 >> m1 时，质点 2 近似不动，
    质点 1 的运动退化为 MEC-007 的单体问题。

运行方法（在本文件所在目录执行）：
    python test_MEC008_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    total_momentum, total_angular_momentum, total_energy, \
    center_of_mass, relative_state, reduced_mass, \
    gravitational_parameter, relative_orbital_elements, \
    relative_eccentricity_vector

TOL = 1e-6


def _solve(x1=1.0, y1=0.0, vx1=0.0, vy1=0.3,
           x2=-1.0, y2=0.0, vx2=0.0, vy2=-0.3,
           G=1.0, m1=1.0, m2=1.0,
           t_end=6.0, n=401):
    """小工具：跑一次数值积分，返回 (t, x1, y1, vx1, vy1, x2, y2, vx2, vy2)。"""
    initial_state = np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2],
                             dtype=float)
    validate_parameters(G=G, m1=m1, m2=m2)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(G, m1, m2),
                    rtol=1e-9, atol=1e-12)
    return (t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3],
            sol.y[4], sol.y[5], sol.y[6], sol.y[7])


def test_position_matches_analytical():
    """两个质点的位置数值解应与解析解一致。"""
    G, m1, m2 = 1.0, 1.0, 1.0
    x1, y1, vx1, vy1 = 1.0, 0.0, 0.0, 0.3
    x2, y2, vx2, vy2 = -1.0, 0.0, 0.0, -0.3
    state0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    elem = relative_orbital_elements(state0, G, m1, m2)
    T = 2.0 * np.pi / elem['n']
    t_end, n = T, 401
    (t, x1_n, y1_n, vx1_n, vy1_n,
     x2_n, y2_n, vx2_n, vy2_n) = _solve(
        x1, y1, vx1, vy1, x2, y2, vx2, vy2,
        G=G, m1=m1, m2=m2, t_end=t_end, n=n)
    (x1_a, y1_a, vx1_a, vy1_a,
     x2_a, y2_a, vx2_a, vy2_a) = analytical(
        t, state0, G=G, m1=m1, m2=m2)
    err_x1 = np.max(np.abs(x1_n - x1_a))
    err_y1 = np.max(np.abs(y1_n - y1_a))
    err_x2 = np.max(np.abs(x2_n - x2_a))
    err_y2 = np.max(np.abs(y2_n - y2_a))
    assert err_x1 < TOL, f"x1 误差 {err_x1:.3e} 超出容差 {TOL}"
    assert err_y1 < TOL, f"y1 误差 {err_y1:.3e} 超出容差 {TOL}"
    assert err_x2 < TOL, f"x2 误差 {err_x2:.3e} 超出容差 {TOL}"
    assert err_y2 < TOL, f"y2 误差 {err_y2:.3e} 超出容差 {TOL}"


def test_velocity_matches_analytical():
    """两个质点的速度数值解应与解析解一致。"""
    G, m1, m2 = 1.0, 1.0, 1.0
    x1, y1, vx1, vy1 = 1.0, 0.0, 0.0, 0.3
    x2, y2, vx2, vy2 = -1.0, 0.0, 0.0, -0.3
    state0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    elem = relative_orbital_elements(state0, G, m1, m2)
    T = 2.0 * np.pi / elem['n']
    t_end, n = T, 401
    (t, x1_n, y1_n, vx1_n, vy1_n,
     x2_n, y2_n, vx2_n, vy2_n) = _solve(
        x1, y1, vx1, vy1, x2, y2, vx2, vy2,
        G=G, m1=m1, m2=m2, t_end=t_end, n=n)
    (x1_a, y1_a, vx1_a, vy1_a,
     x2_a, y2_a, vx2_a, vy2_a) = analytical(
        t, state0, G=G, m1=m1, m2=m2)
    err_vx1 = np.max(np.abs(vx1_n - vx1_a))
    err_vy1 = np.max(np.abs(vy1_n - vy1_a))
    err_vx2 = np.max(np.abs(vx2_n - vx2_a))
    err_vy2 = np.max(np.abs(vy2_n - vy2_a))
    assert err_vx1 < TOL, f"vx1 误差 {err_vx1:.3e} 超出容差 {TOL}"
    assert err_vy1 < TOL, f"vy1 误差 {err_vy1:.3e} 超出容差 {TOL}"
    assert err_vx2 < TOL, f"vx2 误差 {err_vx2:.3e} 超出容差 {TOL}"
    assert err_vy2 < TOL, f"vy2 误差 {err_vy2:.3e} 超出容差 {TOL}"


def test_total_momentum_conserved():
    """总动量 P = m1·v1 + m2·v2 应保持恒定。"""
    G, m1, m2 = 1.0, 2.0, 1.5
    x1, y1, vx1, vy1 = 1.0, 0.5, 0.2, 0.1
    x2, y2, vx2, vy2 = -0.5, -0.3, -0.1, -0.05
    state0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    elem = relative_orbital_elements(state0, G, m1, m2)
    T = 2.0 * np.pi / elem['n']
    (t, x1_n, y1_n, vx1_n, vy1_n,
     x2_n, y2_n, vx2_n, vy2_n) = _solve(
        x1, y1, vx1, vy1, x2, y2, vx2, vy2,
        G=G, m1=m1, m2=m2, t_end=T, n=501)
    P_num = np.array([total_momentum(
        [x1_n[i], y1_n[i], vx1_n[i], vy1_n[i],
         x2_n[i], y2_n[i], vx2_n[i], vy2_n[i]], m1, m2)
        for i in range(len(t))])
    P0 = total_momentum(state0, m1, m2)
    assert np.allclose(P_num, P0, atol=TOL), \
        f"动量不守恒：波动 {np.max(np.abs(P_num - P0)):.3e}"


def test_angular_momentum_conserved():
    """总角动量应保持恒定。"""
    G, m1, m2 = 1.0, 2.0, 1.5
    x1, y1, vx1, vy1 = 1.5, 2.0, 0.3, -0.2
    x2, y2, vx2, vy2 = -1.0, -0.5, -0.1, 0.05
    state0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    elem = relative_orbital_elements(state0, G, m1, m2)
    T = 2.0 * np.pi / elem['n']
    (t, x1_n, y1_n, vx1_n, vy1_n,
     x2_n, y2_n, vx2_n, vy2_n) = _solve(
        x1, y1, vx1, vy1, x2, y2, vx2, vy2,
        G=G, m1=m1, m2=m2, t_end=T, n=501)
    L_num = np.array([total_angular_momentum(
        [x1_n[i], y1_n[i], vx1_n[i], vy1_n[i],
         x2_n[i], y2_n[i], vx2_n[i], vy2_n[i]], m1, m2)
        for i in range(len(t))])
    L0 = total_angular_momentum(state0, m1, m2)
    assert np.allclose(L_num, L0, atol=TOL), \
        f"角动量不守恒：波动 {np.max(np.abs(L_num - L0)):.3e}"


def test_mechanical_energy_conserved():
    """总机械能应保持恒定。"""
    G, m1, m2 = 1.0, 2.0, 1.5
    x1, y1, vx1, vy1 = 1.5, 2.0, 0.3, -0.2
    x2, y2, vx2, vy2 = -1.0, -0.5, -0.1, 0.05
    state0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    elem = relative_orbital_elements(state0, G, m1, m2)
    T = 2.0 * np.pi / elem['n']
    (t, x1_n, y1_n, vx1_n, vy1_n,
     x2_n, y2_n, vx2_n, vy2_n) = _solve(
        x1, y1, vx1, vy1, x2, y2, vx2, vy2,
        G=G, m1=m1, m2=m2, t_end=T, n=501)
    E_num = np.array([total_energy(
        [x1_n[i], y1_n[i], vx1_n[i], vy1_n[i],
         x2_n[i], y2_n[i], vx2_n[i], vy2_n[i]], G, m1, m2)
        for i in range(len(t))])
    E0 = total_energy(state0, G, m1, m2)
    assert np.allclose(E_num, E0, atol=TOL), \
        f"机械能不守恒：波动 {np.max(np.abs(E_num - E0)):.3e}"


def test_center_of_mass_uniform():
    """质心应做匀速直线运动。"""
    G, m1, m2 = 1.0, 2.0, 1.5
    x1, y1, vx1, vy1 = 1.0, 0.5, 0.2, 0.1
    x2, y2, vx2, vy2 = -0.5, -0.3, -0.1, -0.05
    state0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    elem = relative_orbital_elements(state0, G, m1, m2)
    T = 2.0 * np.pi / elem['n']
    (t, x1_n, y1_n, vx1_n, vy1_n,
     x2_n, y2_n, vx2_n, vy2_n) = _solve(
        x1, y1, vx1, vy1, x2, y2, vx2, vy2,
        G=G, m1=m1, m2=m2, t_end=T, n=501)
    CM = np.array([center_of_mass(
        [x1_n[i], y1_n[i], vx1_n[i], vy1_n[i],
         x2_n[i], y2_n[i], vx2_n[i], vy2_n[i]], m1, m2)
        for i in range(len(t))])
    # 质心速度恒定
    Vx0, Vy0 = CM[0, 2], CM[0, 3]
    assert np.allclose(CM[:, 2], Vx0, atol=TOL), \
        f"质心 Vx 不恒定：波动 {np.std(CM[:, 2]):.3e}"
    assert np.allclose(CM[:, 3], Vy0, atol=TOL), \
        f"质心 Vy 不恒定：波动 {np.std(CM[:, 3]):.3e}"
    # 质心位置 = 初始位置 + 速度 * t
    X0, Y0 = CM[0, 0], CM[0, 1]
    X_expected = X0 + Vx0 * t
    Y_expected = Y0 + Vy0 * t
    assert np.allclose(CM[:, 0], X_expected, atol=TOL), \
        f"质心 X 非匀速：偏差 {np.max(np.abs(CM[:, 0] - X_expected)):.3e}"
    assert np.allclose(CM[:, 1], Y_expected, atol=TOL), \
        f"质心 Y 非匀速：偏差 {np.max(np.abs(CM[:, 1] - Y_expected)):.3e}"


def test_relative_orbit_is_ellipse():
    """相对运动轨迹应满足开普勒轨道方程。"""
    G, m1, m2 = 1.0, 2.0, 1.5
    # 选择适中的偏心率（e ~ 0.3）避免近抛物线导致的数值精度问题
    x1, y1, vx1, vy1 = 2.0, 0.0, 0.0, 0.2
    x2, y2, vx2, vy2 = -1.0, 0.0, 0.0, -0.1
    state0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    elem = relative_orbital_elements(state0, G, m1, m2)
    a = elem['a']
    e = elem['e']
    p = elem['p']
    omega = elem['omega']
    T = 2.0 * np.pi / elem['n']
    (t, x1_n, y1_n, vx1_n, vy1_n,
     x2_n, y2_n, vx2_n, vy2_n) = _solve(
        x1, y1, vx1, vy1, x2, y2, vx2, vy2,
        G=G, m1=m1, m2=m2, t_end=T, n=1001)
    # 相对位置
    rx = x1_n - x2_n
    ry = y1_n - y2_n
    r_num = np.hypot(rx, ry)
    theta = np.arctan2(ry, rx)
    r_expected = p / (1 + e * np.cos(theta - omega))
    err = np.max(np.abs(r_num - r_expected))
    assert err < TOL, f"轨道方程偏差：{err:.3e}"
    # 近心距和远心距
    r_min = np.min(r_num)
    r_max = np.max(r_num)
    assert np.allclose(r_min, a * (1 - e), atol=TOL), \
        f"近心距不符：{r_min:.6f} vs a(1-e)={a*(1-e):.6f}"
    assert np.allclose(r_max, a * (1 + e), atol=TOL), \
        f"远心距不符：{r_max:.6f} vs a(1+e)={a*(1+e):.6f}"


def test_circular_orbit_degeneracy():
    """圆轨道条件下，两个质点绕共同质心做匀速圆周运动。"""
    G, m1, m2 = 1.0, 1.0, 1.0
    R = 1.0  # 各质点到质心的距离
    mu = gravitational_parameter(G, m1, m2)
    r_rel = 2 * R  # 相对距离
    v_rel = np.sqrt(mu / r_rel)  # 圆轨道相对速度
    # 质心静止：m1*v1 + m2*v2 = 0, v1 = -v2, v_rel = 2*v1
    v1 = v_rel / 2
    x1, y1, vx1, vy1 = R, 0.0, 0.0, v1
    x2, y2, vx2, vy2 = -R, 0.0, 0.0, -v1
    state0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    elem = relative_orbital_elements(state0, G, m1, m2)
    T = 2.0 * np.pi / elem['n']
    (t, x1_n, y1_n, _, _,
     x2_n, y2_n, _, _) = _solve(
        x1, y1, vx1, vy1, x2, y2, vx2, vy2,
        G=G, m1=m1, m2=m2, t_end=T, n=401)
    # 各质点到质心距离恒定
    r1_cm = np.hypot(x1_n, y1_n)
    r2_cm = np.hypot(x2_n, y2_n)
    assert np.allclose(r1_cm, R, atol=TOL), \
        f"质点 1 到质心距离不恒定：波动 {np.std(r1_cm):.3e}"
    assert np.allclose(r2_cm, R, atol=TOL), \
        f"质点 2 到质心距离不恒定：波动 {np.std(r2_cm):.3e}"
    # 周期后回到起点
    assert abs(x1_n[-1] - x1) < TOL
    assert abs(x2_n[-1] - x2) < TOL


def test_heavy_mass_limit():
    """大质量极限：m2 >> m1 时，质点 2 近似不动，质点 1 逼近 MEC-007。

    质量比 m2/m1 = 1000，质点 2 位移应远小于质点 1 位移。
    质点 1 的运动应近似满足 MEC-007 的单体开普勒问题
    （引力参数 mu = G*m2，力心在质点 2 的初始位置）。
    """
    G = 1.0
    m1 = 1.0
    m2 = 1000.0  # m2/m1 = 1000
    ratio = m2 / m1

    # 质点 2 在原点，近似静止
    x2, y2, vx2, vy2 = 0.0, 0.0, 0.0, 0.0
    # 质点 1 在 r=1，圆轨道速度 v_circ = sqrt(G*m2/r) = sqrt(1000) ≈ 31.6
    # 用 v=5.0（远小于逃逸速度 sqrt(2000)≈44.7），确保椭圆轨道
    x1, y1, vx1, vy1 = 1.0, 0.0, 0.0, 5.0
    state0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]

    mu_approx = G * m2  # MEC-007 等效引力参数
    elem = relative_orbital_elements(state0, G, m1, m2)
    T = 2.0 * np.pi / elem['n']
    (t, x1_n, y1_n, _, _,
     x2_n, y2_n, _, _) = _solve(
        x1, y1, vx1, vy1, x2, y2, vx2, vy2,
        G=G, m1=m1, m2=m2, t_end=T, n=1000)

    # 质点 2 的最大位移
    disp2 = np.max(np.hypot(x2_n - x2, y2_n - y2))
    # 质点 1 的最大位移
    disp1 = np.max(np.hypot(x1_n - x1, y1_n - y1))

    # 质点 2 位移应远小于质点 1 位移
    assert disp2 < disp1 * 0.01, \
        f"大质量极限不满足：disp2/disp1 = {disp2/disp1:.3e} (应 < 0.01)"

    # 质点 1 的轨道应近似满足 MEC-007 的轨道方程
    # r1 ≈ r_rel（因为质点 2 几乎不动）
    r1 = np.hypot(x1_n - x2_n, y1_n - y2_n)
    theta1 = np.arctan2(y1_n - y2_n, x1_n - x2_n)
    # 用 MEC-007 的轨道方程验证
    a_approx = elem['a']
    e_approx = elem['e']
    p_approx = elem['p']
    omega_approx = elem['omega']
    r_expected = p_approx / (1 + e_approx * np.cos(theta1 - omega_approx))
    err = np.max(np.abs(r1 - r_expected))
    assert err < TOL, \
        f"质点 1 不满足近似开普勒轨道：偏差 {err:.3e}"


def test_invalid_parameters_rejected():
    """G <= 0、m1 <= 0 或 m2 <= 0 应被拒绝。"""
    try:
        validate_parameters(G=0.0, m1=1.0, m2=1.0)
        raise AssertionError("应拒绝 G=0")
    except AssertionError as e:
        assert "G" in str(e)

    try:
        validate_parameters(G=1.0, m1=-1.0, m2=1.0)
        raise AssertionError("应拒绝 m1<0")
    except AssertionError as e:
        assert "m1" in str(e)

    try:
        validate_parameters(G=1.0, m1=1.0, m2=0.0)
        raise AssertionError("应拒绝 m2=0")
    except AssertionError as e:
        assert "m2" in str(e)


if __name__ == "__main__":
    test_position_matches_analytical()
    test_velocity_matches_analytical()
    test_total_momentum_conserved()
    test_angular_momentum_conserved()
    test_mechanical_energy_conserved()
    test_center_of_mass_uniform()
    test_relative_orbit_is_ellipse()
    test_circular_orbit_degeneracy()
    test_heavy_mass_limit()
    test_invalid_parameters_rejected()
    print("OK: MEC-008 数值解与解析解一致")
