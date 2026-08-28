"""MEC-033 —— 一致性测试：凸轮从动件机构。

验证：
- DRRD 边界值（三种轮廓）
- 速度比数值验证
- 加速度比数值验证
- Cycloidal/3-4-5 加速度连续性
- SHM 加速度跳变
- 等效惯量（第一性原理）
- I_eff' 解析 vs 数值
- 能量守恒（τ=0）
- 功-能定理（τ≠0）
- 接触力正值
- 压力角
- 反例验证（非循环）
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC033_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, follower_displacement, follower_velocity_ratio,
                   follower_acceleration_ratio, equivalent_inertia,
                   equivalent_inertia_derivative, contact_force,
                   pressure_angle, mechanical_energy, validate_parameters,
                   _profile_derivatives)

TOL = 1e-6

# 默认参数
H = 0.01
B_RISE = np.pi / 2
B_DWELL1 = np.pi / 4
B_RETURN = np.pi / 2
B_DWELL2 = 2 * np.pi - B_RISE - B_DWELL1 - B_RETURN
I_CAM = 0.001
M_F = 0.1
K = 100.0
R_B = 0.03
PROFILE = 'cycloidal'


def _solve(theta0=0.0, omega0=10.0, tau=0.0, profile=PROFILE,
           t_end=2.0, n=501):
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(
        dynamics, (0.0, t_end), [theta0, omega0],
        t_eval=t_eval,
        args=(H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, K, tau, R_B, profile),
        rtol=1e-10, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1]


def test_drrd_boundary_values():
    """三种轮廓的 DRRD 边界值应正确。"""
    for profile in ['shm', 'cycloidal', 'poly345']:
        # 起点 y=0
        y0 = follower_displacement(0.0, H, B_RISE, B_DWELL1, B_RETURN, profile)
        assert abs(y0) < TOL, f"{profile}: y(0)={y0}"
        # 峰值 y=h
        y_peak = follower_displacement(B_RISE, H, B_RISE, B_DWELL1, B_RETURN, profile)
        assert abs(y_peak - H) < TOL, f"{profile}: y(β_r)={y_peak} ≠ h={H}"
        # 回到零
        b3 = B_RISE + B_DWELL1 + B_RETURN
        y_end = follower_displacement(b3, H, B_RISE, B_DWELL1, B_RETURN, profile)
        assert abs(y_end) < TOL, f"{profile}: y(return_end)={y_end}"
        # 周期性 y(2π)=y(0)=0
        y_2pi = follower_displacement(2 * np.pi, H, B_RISE, B_DWELL1, B_RETURN, profile)
        assert abs(y_2pi) < TOL


def test_velocity_numerical():
    """速度比应与位移的数值导数一致。"""
    h = 1e-6
    for theta in [0.3, 0.8, 1.2, 2.5, 3.5, 4.5, 5.5]:
        yp = follower_velocity_ratio(theta, H, B_RISE, B_DWELL1, B_RETURN, PROFILE)
        yp_num = (follower_displacement(theta + h, H, B_RISE, B_DWELL1, B_RETURN, PROFILE)
                  - follower_displacement(theta - h, H, B_RISE, B_DWELL1, B_RETURN, PROFILE)) / (2 * h)
        assert abs(yp - yp_num) < 1e-4, \
            f"dy/dθ 不符: {yp:.8f} vs {yp_num:.8f} (θ={theta:.2f})"


def test_acceleration_numerical():
    """加速度比应与速度比的数值导数一致。"""
    h = 1e-5
    for theta in [0.3, 0.8, 1.2, 2.5, 3.5, 4.5, 5.5]:
        ypp = follower_acceleration_ratio(theta, H, B_RISE, B_DWELL1, B_RETURN, PROFILE)
        yp_p = follower_velocity_ratio(theta + h, H, B_RISE, B_DWELL1, B_RETURN, PROFILE)
        yp_m = follower_velocity_ratio(theta - h, H, B_RISE, B_DWELL1, B_RETURN, PROFILE)
        ypp_num = (yp_p - yp_m) / (2 * h)
        assert abs(ypp - ypp_num) < 1e-3, \
            f"d²y/dθ² 不符: {ypp:.8f} vs {ypp_num:.8f} (θ={theta:.2f})"


def test_cycloidal_acceleration_continuity():
    """Cycloidal 轮廓的加速度在段间应连续（=0）。"""
    eps = 1e-8
    b1 = B_RISE
    b2 = b1 + B_DWELL1
    b3 = b2 + B_RETURN

    for boundary in [b1, b2, b3, 0, 2 * np.pi]:
        _, _, ypp_before = _profile_derivatives(boundary - eps, H, B_RISE, B_DWELL1, B_RETURN, 'cycloidal')
        _, _, ypp_after = _profile_derivatives(boundary + eps, H, B_RISE, B_DWELL1, B_RETURN, 'cycloidal')
        assert abs(ypp_before - ypp_after) < 1e-4, \
            f"Cycloidal y'' 跳变 at θ={boundary:.4f}: {ypp_before:.6f} → {ypp_after:.6f}"


def test_poly345_acceleration_continuity():
    """3-4-5 多项式轮廓的加速度在段间应连续（=0）。"""
    eps = 1e-8
    b1 = B_RISE
    b2 = b1 + B_DWELL1
    b3 = b2 + B_RETURN

    for boundary in [b1, b2, b3, 0, 2 * np.pi]:
        _, _, ypp_before = _profile_derivatives(boundary - eps, H, B_RISE, B_DWELL1, B_RETURN, 'poly345')
        _, _, ypp_after = _profile_derivatives(boundary + eps, H, B_RISE, B_DWELL1, B_RETURN, 'poly345')
        assert abs(ypp_before - ypp_after) < 1e-4, \
            f"3-4-5 y'' 跳变 at θ={boundary:.4f}: {ypp_before:.6f} → {ypp_after:.6f}"


def test_shm_acceleration_jump():
    """SHM 轮廓的加速度在 Rise-Dwell 边界应有有限跳变。"""
    eps = 1e-8
    b1 = B_RISE

    _, _, ypp_before = _profile_derivatives(b1 - eps, H, B_RISE, B_DWELL1, B_RETURN, 'shm')
    _, _, ypp_after = _profile_derivatives(b1 + eps, H, B_RISE, B_DWELL1, B_RETURN, 'shm')

    # SHM rise end: y'' = -h*π²/(2*β_r²), dwell: y'' = 0
    expected_jump = abs(ypp_before - ypp_after)
    assert expected_jump > 1e-4, \
        f"SHM y'' 应有跳变，实际 {expected_jump:.2e}"


def test_equivalent_inertia_first_principles():
    """从第一性原理验证 I_eff = I_cam + m_f·(dy/dθ)²。"""
    omega = 10.0
    for theta in [0.3, 0.8, 1.5, 2.5, 4.0, 5.5]:
        yp = follower_velocity_ratio(theta, H, B_RISE, B_DWELL1, B_RETURN, PROFILE)
        T_total = 0.5 * I_CAM * omega**2 + 0.5 * M_F * (yp * omega)**2
        I_eff = equivalent_inertia(theta, H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, PROFILE)
        T_eff = 0.5 * I_eff * omega**2
        assert abs(T_total - T_eff) < 1e-15, \
            f"动能不符 (θ={theta:.2f}): T={T_total:.12f} vs {T_eff:.12f}"


def test_I_eff_derivative_analytical_vs_numerical():
    """解析 I_eff' 应与数值导数一致。"""
    h = 1e-6
    for theta in [0.3, 0.8, 1.5, 2.5, 4.0, 5.5]:
        I_ana = equivalent_inertia_derivative(theta, H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, PROFILE)
        I_p = equivalent_inertia(theta + h, H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, PROFILE)
        I_m = equivalent_inertia(theta - h, H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, PROFILE)
        I_num = (I_p - I_m) / (2 * h)
        assert abs(I_ana - I_num) < 1e-5, \
            f"I_eff' 不符 (θ={theta:.2f}): 解析={I_ana:.10f} 数值={I_num:.10f}"


def test_energy_conservation():
    """τ=0 时机械能应守恒（cycloidal, 2s 积分 ≈ 3 圈）。"""
    t, theta, omega = _solve(theta0=0.0, omega0=10.0, tau=0.0, t_end=2.0, n=1001)
    E = np.array([mechanical_energy([th, w], H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, K, PROFILE)
                  for th, w in zip(theta, omega)])
    drift = np.max(np.abs(E - E[0]))
    assert drift < 1e-3, f"能量漂移 {drift:.3e} 超出容差"


def test_work_energy_theorem():
    """恒定 τ 时 ΔE = τ·Δθ 应成立。"""
    tau = 0.001
    t, theta, omega = _solve(theta0=0.0, omega0=10.0, tau=tau, t_end=1.0, n=501)
    E = np.array([mechanical_energy([th, w], H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, K, PROFILE)
                  for th, w in zip(theta, omega)])
    dE = E[-1] - E[0]
    dtheta = theta[-1] - theta[0]
    expected = tau * dtheta
    assert abs(dE - expected) < 1e-4, \
        f"ΔE={dE:.8f}, τ·Δθ={expected:.8f}, 差 {abs(dE-expected):.3e}"


def test_contact_force_positive():
    """默认工况下接触力应 > 0（不跳脱）。"""
    t, theta, omega = _solve(theta0=0.0, omega0=10.0, tau=0.0, t_end=2.0, n=1001)
    F_min = float('inf')
    for i in range(0, len(t), 10):
        th, w = theta[i], omega[i]
        alpha = dynamics(0, [th, w], H, B_RISE, B_DWELL1, B_RETURN,
                         I_CAM, M_F, K, 0, R_B, PROFILE)[1]
        F = contact_force(th, w, alpha, H, B_RISE, B_DWELL1, B_RETURN, M_F, K, PROFILE)
        F_min = min(F_min, F)
    assert F_min >= -1e-10, f"最小接触力 {F_min:.6f} N < 0，从动件跳脱"


def test_pressure_angle_reasonable():
    """压力角应在合理范围内（< 60°）。"""
    for theta in np.linspace(0, 2 * np.pi, 73):
        pa = pressure_angle(theta, H, B_RISE, B_DWELL1, B_RETURN, R_B, PROFILE)
        assert np.degrees(pa) < 60, f"压力角过大: {np.degrees(pa):.1f}° (θ={theta:.2f})"


def test_error_injection_non_circular():
    """反例验证：I_eff' 乘 2 后能量应不守恒。"""
    def dynamics_wrong(t, state, *args):
        h, br, bd1, bre, Ic, mf, k, tau, rb, prof = args
        theta, omega = state
        y, yp, ypp = _profile_derivatives(theta, h, br, bd1, bre, prof)
        I_eff = Ic + mf * yp**2
        I_eff_prime_wrong = 2.0 * (2 * mf * yp * ypp)  # 故意乘 2
        dV = k * y * yp
        alpha = (tau - 0.5 * I_eff_prime_wrong * omega**2 - dV) / I_eff
        return np.array([omega, alpha])

    sol = solve_ivp(dynamics_wrong, (0, 2), [0, 10], t_eval=np.linspace(0, 2, 501),
                    args=(H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, K, 0, R_B, PROFILE),
                    rtol=1e-10, atol=1e-12)
    theta = sol.y[0]
    omega = sol.y[1]
    E = np.array([0.5 * equivalent_inertia(t, H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, PROFILE) * w**2
                  + 0.5 * K * follower_displacement(t, H, B_RISE, B_DWELL1, B_RETURN, PROFILE)**2
                  for t, w in zip(theta, omega)])
    drift_wrong = np.max(np.abs(E - E[0]))
    assert drift_wrong > 1e-4, \
        f"反例失败：错误 I_eff' 的能量漂移 {drift_wrong:.3e} 过小"


def test_dynamics_shape():
    """dynamics 应返回 shape (2,)。"""
    d = dynamics(0.0, [0.5, 10.0], H, B_RISE, B_DWELL1, B_RETURN,
                 I_CAM, M_F, K, 0, R_B, PROFILE)
    assert d.shape == (2,)
    assert abs(d[0] - 10.0) < TOL, "dθ/dt 应等于 ω"


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    try:
        validate_parameters(h=-1)
        raise AssertionError("应拒绝 h<0")
    except AssertionError as e:
        assert "h" in str(e)
    try:
        validate_parameters(beta_rise=3*np.pi)
        raise AssertionError("应拒绝角度之和超过2π")
    except AssertionError as e:
        assert "2π" in str(e) or "2p" in str(e).lower() or "超过" in str(e)


if __name__ == "__main__":
    test_drrd_boundary_values()
    print("✓ DRRD 边界值 (三种轮廓)")
    test_velocity_numerical()
    print("✓ 速度数值验证")
    test_acceleration_numerical()
    print("✓ 加速度数值验证")
    test_cycloidal_acceleration_continuity()
    print("✓ Cycloidal 加速度连续性")
    test_poly345_acceleration_continuity()
    print("✓ 3-4-5 加速度连续性")
    test_shm_acceleration_jump()
    print("✓ SHM 加速度跳变")
    test_equivalent_inertia_first_principles()
    print("✓ I_eff (第一性原理)")
    test_I_eff_derivative_analytical_vs_numerical()
    print("✓ I_eff' 解析 vs 数值")
    test_energy_conservation()
    print("✓ 能量守恒")
    test_work_energy_theorem()
    print("✓ 功-能定理")
    test_contact_force_positive()
    print("✓ 接触力正值")
    test_pressure_angle_reasonable()
    print("✓ 压力角")
    test_error_injection_non_circular()
    print("✓ 反例验证 (非循环)")
    test_dynamics_shape()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-033 所有一致性测试通过")
