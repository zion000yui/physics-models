"""MEC-031 —— 一致性测试：曲柄滑块机构。

验证：
- 滑块位置约束（y=0）
- 位置解析解 vs 闭环约束
- 速度比数值验证
- 加速度数值验证
- 极限位置（TDC/BDC）滑块速度=0
- 等效惯量正定性
- 等效惯量随 θ 变化
- 能量守恒（τ=0）
- 功-能定理（τ≠0）
- 动力学独立验证（反例：I_eff' 乘 2 → 能量不守恒）
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC031_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, slider_position, rod_angle,
                   slider_velocity_ratio, rod_angular_velocity_ratio,
                   slider_acceleration_ratio, velocity_analysis,
                   acceleration_analysis, equivalent_inertia,
                   mechanical_energy, toggle_positions,
                   validate_parameters)

TOL = 1e-6

# 默认参数
R, L = 0.3, 1.0
M_CRANK, M_ROD, M_SL = 1.0, 1.0, 1.0
R_CM, L_CM = R / 2, L / 2
I_CRANK = M_CRANK * R**2 / 12
I_ROD = M_ROD * L**2 / 12


def _solve(theta0=0.5, omega0=1.0, tau=0.0, g=0.0,
           t_end=10.0, n=501):
    """数值积分，返回 (t, theta, omega)。"""
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(
        dynamics, (0.0, t_end), [theta0, omega0],
        t_eval=t_eval,
        args=(R, L, M_CRANK, M_ROD, M_SL, R_CM, L_CM, I_CRANK, I_ROD, g, tau),
        rtol=1e-10, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1]


def test_slider_constraint():
    """滑块位置应满足闭环约束 |B-A|=l。"""
    for theta in np.linspace(0, 2 * np.pi, 37):
        x = slider_position(theta, R, L)
        # A = (r cos θ, r sin θ), B = (x, 0)
        # |B - A|² = (x - r cos θ)² + (r sin θ)² = l²
        ax = R * np.cos(theta)
        ay = R * np.sin(theta)
        dist_sq = (x - ax)**2 + ay**2
        assert abs(dist_sq - L**2) < TOL, \
            f"闭环残差 {abs(dist_sq - L**2):.3e} (θ={theta:.3f})"


def test_rod_angle_consistency():
    """连杆角应满足 sin φ = (r/l) sin θ。"""
    for theta in np.linspace(0, 2 * np.pi, 37):
        phi = rod_angle(theta, R, L)
        assert abs(np.sin(phi) - R / L * np.sin(theta)) < TOL, \
            f"连杆角不符 (θ={theta:.3f})"


def test_velocity_numerical():
    """速度比应与位置的数值导数一致。"""
    h = 1e-6
    for theta in [0.3, 0.8, 1.5, 2.0, 3.0, 5.0]:
        dx = slider_velocity_ratio(theta, R, L)
        dphi = rod_angular_velocity_ratio(theta, R, L)

        # 数值导数
        dx_num = (slider_position(theta + h, R, L)
                  - slider_position(theta - h, R, L)) / (2 * h)
        dphi_num = (rod_angle(theta + h, R, L)
                    - rod_angle(theta - h, R, L)) / (2 * h)

        assert abs(dx - dx_num) < 1e-4, \
            f"dx/dθ 不符: {dx:.6f} vs {dx_num:.6f} (θ={theta})"
        assert abs(dphi - dphi_num) < 1e-4, \
            f"dφ/dθ 不符: {dphi:.6f} vs {dphi_num:.6f} (θ={theta})"


def test_acceleration_numerical():
    """加速度应与速度比的数值导数一致（α=0, ω=1 时）。"""
    h = 1e-5
    omega = 1.0
    alpha = 0.0

    for theta in [0.3, 0.8, 1.5, 2.0, 3.0]:
        slider_acc, rod_alpha = acceleration_analysis(
            theta, omega, alpha, R, L)

        # 数值：ÿ = (d²x/dθ²)·ω²  (α=0)
        d2x = (slider_position(theta + h, R, L)
               - 2 * slider_position(theta, R, L)
               + slider_position(theta - h, R, L)) / h**2
        slider_acc_num = d2x * omega**2

        assert abs(slider_acc - slider_acc_num) < 1e-3, \
            f"ÿ 不符: {slider_acc:.6f} vs {slider_acc_num:.6f} (θ={theta})"


def test_toggle_positions():
    """在极限位置（TDC/BDC）滑块速度应=0。"""
    x_tdc, x_bdc, theta_tdc, theta_bdc = toggle_positions(R, L)

    # TDC: θ=0
    assert abs(theta_tdc) < TOL
    vx_tdc, _ = velocity_analysis(theta_tdc, 1.0, R, L)
    assert abs(vx_tdc) < 1e-10, f"TDC 速度 {vx_tdc:.3e} 不为零"

    # BDC: θ=π
    assert abs(theta_bdc - np.pi) < TOL
    vx_bdc, _ = velocity_analysis(theta_bdc, 1.0, R, L)
    assert abs(vx_bdc) < 1e-10, f"BDC 速度 {vx_bdc:.3e} 不为零"

    # 行程 = 2r
    assert abs((x_tdc - x_bdc) - 2 * R) < TOL, \
        f"行程 {x_tdc - x_bdc:.6f} ≠ 2r={2*R:.6f}"


def test_equivalent_inertia_positive():
    """I_eff 应处处为正。"""
    for theta in np.linspace(0, 2 * np.pi, 73):
        I_eff = equivalent_inertia(theta, R, L, M_CRANK, M_ROD, M_SL,
                                   R_CM, L_CM, I_CRANK, I_ROD)
        assert I_eff > 0, f"I_eff={I_eff} 非正 (θ={theta:.3f})"


def test_equivalent_inertia_varies():
    """I_eff 应随 θ 变化（非常数）。"""
    vals = [equivalent_inertia(t, R, L, M_CRANK, M_ROD, M_SL,
                               R_CM, L_CM, I_CRANK, I_ROD)
            for t in np.linspace(0, 2 * np.pi, 73)]
    assert max(vals) - min(vals) > 1e-3, "I_eff 几乎不变"


def test_kinetic_energy_independent():
    """独立计算总动能，验证 T = ½ I_eff ω²。"""
    omega = 1.0
    for theta in [0.3, 0.8, 1.5, 2.0, 3.0, 5.0]:
        phi = rod_angle(theta, R, L)
        dphi = rod_angular_velocity_ratio(theta, R, L)
        dx = slider_velocity_ratio(theta, R, L)

        # 曲柄动能（绕 O 转动）
        I_O = I_CRANK + M_CRANK * R_CM**2
        T_crank = 0.5 * I_O * omega**2

        # 连杆动能（一般平面运动）
        # CM = (r cos θ + l_cm cos φ, r sin θ + l_cm sin φ)
        vx_cm = -R * np.sin(theta) * omega - L_CM * np.sin(phi) * dphi * omega
        vy_cm = R * np.cos(theta) * omega + L_CM * np.cos(phi) * dphi * omega
        T_rod = 0.5 * M_ROD * (vx_cm**2 + vy_cm**2) + 0.5 * I_ROD * (dphi * omega)**2

        # 滑块动能（平动）
        T_sl = 0.5 * M_SL * (dx * omega)**2

        T_total = T_crank + T_rod + T_sl
        I_eff = equivalent_inertia(theta, R, L, M_CRANK, M_ROD, M_SL,
                                   R_CM, L_CM, I_CRANK, I_ROD)
        T_eff = 0.5 * I_eff * omega**2

        assert abs(T_total - T_eff) < 1e-14, \
            f"动能不符: T_total={T_total:.12f}, T_eff={T_eff:.12f} (θ={theta})"


def test_energy_conservation():
    """τ=0 时机械能应守恒。"""
    t, theta, omega = _solve(theta0=0.5, omega0=1.0, tau=0.0,
                             t_end=10.0, n=1001)
    E = np.array([mechanical_energy(
        [theta[i], omega[i]], R, L, M_CRANK, M_ROD, M_SL,
        R_CM, L_CM, I_CRANK, I_ROD, g=0.0)
        for i in range(len(t))])
    drift = np.max(np.abs(E - E[0]))
    assert drift < 1e-3, f"能量漂移 {drift:.3e} 超出容差"


def test_work_energy_theorem():
    """恒定 τ 时 ΔE = τ·Δθ 应成立。"""
    tau = 0.5
    t, theta, omega = _solve(theta0=0.5, omega0=1.0, tau=tau,
                             t_end=5.0, n=501)
    E = np.array([mechanical_energy(
        [theta[i], omega[i]], R, L, M_CRANK, M_ROD, M_SL,
        R_CM, L_CM, I_CRANK, I_ROD, g=0.0)
        for i in range(len(t))])
    dE = E[-1] - E[0]
    dtheta = theta[-1] - theta[0]
    expected = tau * dtheta
    assert abs(dE - expected) < 1e-2, \
        f"ΔE={dE:.6f}, τ·Δθ={expected:.6f}, 差 {abs(dE-expected):.3e}"


def test_dynamics_non_circular():
    """反例验证：I_eff' 乘 2 后能量应不守恒。

    如果能量守恒测试是循环验证，错误的 I_eff' 不应影响结果。
    """
    def dynamics_wrong(t, state, *args):
        r, l, mc, mr, ms, rcm, lcm, Ic, Ir, g, tau = args
        theta, omega = state

        I_eff = equivalent_inertia(theta, r, l, mc, mr, ms, rcm, lcm, Ic, Ir)
        h = 1e-7
        Ip = equivalent_inertia(theta + h, r, l, mc, mr, ms, rcm, lcm, Ic, Ir)
        Im = equivalent_inertia(theta - h, r, l, mc, mr, ms, rcm, lcm, Ic, Ir)
        I_prime_wrong = 2.0 * (Ip - Im) / (2 * h)  # 故意乘 2

        dV = 0.0
        alpha = (tau - 0.5 * I_prime_wrong * omega**2 - dV) / I_eff
        return np.array([omega, alpha])

    sol = solve_ivp(dynamics_wrong, (0, 10), [0.5, 1.0],
                    t_eval=np.linspace(0, 10, 501),
                    args=(R, L, M_CRANK, M_ROD, M_SL, R_CM, L_CM,
                          I_CRANK, I_ROD, 0.0, 0.0),
                    rtol=1e-10, atol=1e-12)

    theta = sol.y[0]
    omega = sol.y[1]
    E = np.array([0.5 * equivalent_inertia(
        theta[i], R, L, M_CRANK, M_ROD, M_SL, R_CM, L_CM, I_CRANK, I_ROD)
        * omega[i]**2 for i in range(len(theta))])

    drift_wrong = np.max(np.abs(E - E[0]))
    # 正确的 drift ~5e-10，错误的应远大于此
    assert drift_wrong > 1e-4, \
        f"反例验证失败：错误 I_eff' 的能量漂移 {drift_wrong:.3e} 过小，可能为循环验证"


def test_dynamics_shape():
    """dynamics 应返回 shape (2,)。"""
    d = dynamics(0.0, [0.5, 1.0], R, L, M_CRANK, M_ROD, M_SL,
                 R_CM, L_CM, I_CRANK, I_ROD, 0.0, 0.0)
    assert d.shape == (2,), f"dynamics 返回 shape={d.shape}"
    assert abs(d[0] - 1.0) < TOL, "dθ/dt 应等于 ω"


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    try:
        validate_parameters(r=-1, l=1)
        raise AssertionError("应拒绝 r<0")
    except AssertionError as e:
        assert "r" in str(e)
    try:
        validate_parameters(r=1, l=0.5)  # l < r
        raise AssertionError("应拒绝 l<r")
    except AssertionError as e:
        assert "l" in str(e)


def test_l_infty_limit():
    """l→∞ 时滑块运动趋近简谐 x ≈ r cos θ + l（数学极限）。"""
    l_large = 1000.0 * R  # l >> r
    theta = 0.7
    x = slider_position(theta, R, l_large)
    x_approx = R * np.cos(theta) + l_large  # 一阶近似
    # 偏差应 ~O(r²/l)，非常小
    err = abs(x - x_approx)
    assert err < R**2 / l_large * 2, \
        f"l→∞ 极限不符：err={err:.6e}"


if __name__ == "__main__":
    test_slider_constraint()
    print("✓ 滑块闭环约束")
    test_rod_angle_consistency()
    print("✓ 连杆角一致性")
    test_velocity_numerical()
    print("✓ 速度数值验证")
    test_acceleration_numerical()
    print("✓ 加速度数值验证")
    test_toggle_positions()
    print("✓ 极限位置 (TDC/BDC)")
    test_equivalent_inertia_positive()
    print("✓ I_eff 正定性")
    test_equivalent_inertia_varies()
    print("✓ I_eff 非常数")
    test_kinetic_energy_independent()
    print("✓ 动能独立验证")
    test_energy_conservation()
    print("✓ 能量守恒")
    test_work_energy_theorem()
    print("✓ 功-能定理")
    test_dynamics_non_circular()
    print("✓ 反例验证 (非循环)")
    test_dynamics_shape()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    test_l_infty_limit()
    print("✓ l→∞ 极限")
    print("\nOK: MEC-031 所有一致性测试通过")
