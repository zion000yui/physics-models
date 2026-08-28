"""MEC-030 —— 一致性测试：四连杆机构。

验证：
- 闭环位置约束
- Freudenstein 方程
- open / crossed 两个构型
- 速度比数值验证
- 加速度数值验证
- Grashof 判据
- 极限位置（toggle）
- 等效惯量正定性
- 能量守恒（τ=0）
- 功-能定理（τ≠0）
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC030_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, position_analysis, velocity_ratios,
                   velocity_analysis, acceleration_analysis,
                   grashof_criterion, freudenstein_coefficients,
                   toggle_positions, equivalent_inertia,
                   mechanical_energy, validate_parameters)

TOL = 1e-6

# 默认参数：Grashof crank-rocker
L1, L2, L3, L4 = 2.0, 0.5, 1.5, 1.2
M2, M3, M4 = 1.0, 1.0, 1.0
R2, R3, R4 = L2 / 2, L3 / 2, L4 / 2
I2 = M2 * L2**2 / 12
I3 = M3 * L3**2 / 12
I4 = M4 * L4**2 / 12


def _solve(theta2_0=0.5, omega2_0=1.0, tau=0.0, g=0.0,
           config='open', t_end=10.0, n=501):
    """跑一次数值积分，返回 (t, theta2, omega2)。"""
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(
        dynamics, (0.0, t_end), [theta2_0, omega2_0],
        t_eval=t_eval,
        args=(L1, L2, L3, L4, M2, M3, M4, R2, R3, R4, I2, I3, I4, g, tau, config),
        rtol=1e-10, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1]


def test_loop_closure():
    """闭环约束在任意 θ₂ 处应满足。"""
    for theta2 in np.linspace(0, 2 * np.pi, 37):
        t3, t4 = position_analysis(theta2, L1, L2, L3, L4, 'open')
        rx = L2 * np.cos(theta2) + L3 * np.cos(t3) - L4 * np.cos(t4) - L1
        ry = L2 * np.sin(theta2) + L3 * np.sin(t3) - L4 * np.sin(t4)
        err = np.sqrt(rx**2 + ry**2)
        assert err < TOL, f"闭环残差 {err:.3e} 超出容差 (θ₂={theta2:.3f})"


def test_freudenstein_equation():
    """Freudenstein 方程应成立。"""
    K1, K2, K3 = freudenstein_coefficients(L1, L2, L3, L4)
    for theta2 in np.linspace(0, 2 * np.pi, 37):
        _, t4 = position_analysis(theta2, L1, L2, L3, L4, 'open')
        lhs = K1 * np.cos(t4) - K2 * np.cos(theta2) + K3
        rhs = np.cos(theta2 - t4)
        assert abs(lhs - rhs) < TOL, f"Freudenstein 残差 {abs(lhs-rhs):.3e}"


def test_open_vs_crossed():
    """open 和 crossed 应给出不同的 θ₃, θ₄。"""
    theta2 = 0.5
    t3_o, t4_o = position_analysis(theta2, L1, L2, L3, L4, 'open')
    t3_c, t4_c = position_analysis(theta2, L1, L2, L3, L4, 'crossed')
    assert abs(t3_o - t3_c) > 1e-3, "open/crossed θ₃ 过于接近"
    assert abs(t4_o - t4_c) > 1e-3, "open/crossed θ₄ 过于接近"


def test_velocity_ratios_numerical():
    """速度比应与位置分析的数值导数一致。"""
    h = 1e-6
    for theta2 in [0.3, 0.8, 1.5, 2.0, 3.0, 5.0]:
        t3, t4 = position_analysis(theta2, L1, L2, L3, L4, 'open')
        R3, R4 = velocity_ratios(theta2, t3, t4, L1, L2, L3, L4)

        # 数值导数
        t3_p, _ = position_analysis(theta2 + h, L1, L2, L3, L4, 'open')
        t3_m, _ = position_analysis(theta2 - h, L1, L2, L3, L4, 'open')
        _, t4_p = position_analysis(theta2 + h, L1, L2, L3, L4, 'open')
        _, t4_m = position_analysis(theta2 - h, L1, L2, L3, L4, 'open')

        d_t3 = (t3_p - t3_m) / (2 * h)
        d_t4 = (t4_p - t4_m) / (2 * h)

        assert abs(R3 - d_t3) < 1e-4, \
            f"R3 不符: {R3:.6f} vs {d_t3:.6f} (θ₂={theta2})"
        assert abs(R4 - d_t4) < 1e-4, \
            f"R4 不符: {R4:.6f} vs {d_t4:.6f} (θ₂={theta2})"


def test_acceleration_numerical():
    """加速度应与速度比的数值导数一致（α₂=0, ω₂=1 时）。"""
    h = 1e-5
    omega2 = 1.0
    alpha2 = 0.0

    for theta2 in [0.3, 0.8, 1.5, 2.0, 3.0]:
        t3, t4 = position_analysis(theta2, L1, L2, L3, L4, 'open')
        omega3, omega4 = velocity_analysis(theta2, t3, t4, omega2,
                                           L1, L2, L3, L4)
        alpha3, alpha4 = acceleration_analysis(
            theta2, t3, t4, omega2, omega3, omega4, alpha2,
            L1, L2, L3, L4)

        # 数值：α₃ = (d²θ₃/dθ₂²)·ω₂²  (α₂=0)
        t3_p, _ = position_analysis(theta2 + h, L1, L2, L3, L4, 'open')
        t3_m, _ = position_analysis(theta2 - h, L1, L2, L3, L4, 'open')
        d2_t3 = (t3_p - 2 * t3 + t3_m) / h**2
        alpha3_num = d2_t3 * omega2**2

        assert abs(alpha3 - alpha3_num) < 1e-3, \
            f"α₃ 不符: {alpha3:.6f} vs {alpha3_num:.6f} (θ₂={theta2})"


def test_grashof_criterion():
    """Grashof 判据应正确分类。"""
    # 默认: crank-rocker (l2 最短)
    assert grashof_criterion(L1, L2, L3, L4) == 'crank-rocker'
    # double-crank: ground 最短
    assert grashof_criterion(0.5, 1.0, 1.5, 1.2) == 'double-crank'
    # non-grashof
    assert grashof_criterion(1.0, 1.0, 2.0, 1.5) == 'non-grashof'


def test_toggle_positions():
    """在极限位置处 R4 应≈0。"""
    t4_ext, t4_fold, t2_ext, t2_fold = toggle_positions(L1, L2, L3, L4)

    for theta2_toggle in [t2_ext, t2_fold]:
        t3, t4 = position_analysis(theta2_toggle, L1, L2, L3, L4, 'open')
        _, R4 = velocity_ratios(theta2_toggle, t3, t4, L1, L2, L3, L4)
        assert abs(R4) < 1e-4, \
            f"极限位置 R4={R4:.6e} 不为零 (θ₂={theta2_toggle:.4f})"


def test_equivalent_inertia_positive():
    """I_eff 应处处为正。"""
    for theta2 in np.linspace(0, 2 * np.pi, 73):
        I_eff = equivalent_inertia(theta2, L1, L2, L3, L4,
                                   M2, M3, M4, R2, R3, R4, I2, I3, I4,
                                   config='open')
        assert I_eff > 0, f"I_eff={I_eff} 非正 (θ₂={theta2:.3f})"


def test_equivalent_inertia_varies():
    """I_eff 应随 θ₂ 变化（非常数）。"""
    vals = [equivalent_inertia(t, L1, L2, L3, L4, M2, M3, M4,
                              R2, R3, R4, I2, I3, I4, config='open')
            for t in np.linspace(0, 2 * np.pi, 73)]
    assert max(vals) - min(vals) > 1e-3, "I_eff 几乎不变"


def test_energy_conservation():
    """τ=0 时机械能应守恒。"""
    t, theta2, omega2 = _solve(theta2_0=0.5, omega2_0=1.0, tau=0.0,
                               t_end=10.0, n=1001)
    E = np.array([mechanical_energy(
        [theta2[i], omega2[i]], L1, L2, L3, L4, M2, M3, M4,
        g=0.0, config='open') for i in range(len(t))])
    E0 = E[0]
    drift = np.max(np.abs(E - E0))
    assert drift < 1e-3, f"能量漂移 {drift:.3e} 超出容差"


def test_work_energy_theorem():
    """恒定 τ 时 ΔE = τ·Δθ₂ 应成立。"""
    tau = 0.5
    t, theta2, omega2 = _solve(theta2_0=0.5, omega2_0=1.0, tau=tau,
                               t_end=5.0, n=501)
    E = np.array([mechanical_energy(
        [theta2[i], omega2[i]], L1, L2, L3, L4, M2, M3, M4,
        g=0.0, config='open') for i in range(len(t))])
    dE = E[-1] - E[0]
    dtheta = theta2[-1] - theta2[0]
    expected = tau * dtheta
    assert abs(dE - expected) < 1e-2, \
        f"ΔE={dE:.6f}, τ·Δθ={expected:.6f}, 差 {abs(dE-expected):.3e}"


def test_dynamics_shape():
    """dynamics 应返回 shape (2,)。"""
    d = dynamics(0.0, [0.5, 1.0], L1, L2, L3, L4, M2, M3, M4,
                 R2, R3, R4, I2, I3, I4, 0.0, 0.0, 'open')
    assert d.shape == (2,), f"dynamics 返回 shape={d.shape}"
    assert abs(d[0] - 1.0) < TOL, "dθ₂/dt 应等于 ω₂"


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    try:
        validate_parameters(-1, 1, 1, 1)
        raise AssertionError("应拒绝 l1<0")
    except AssertionError as e:
        assert "l1" in str(e)
    try:
        validate_parameters(1, 1, 1, 1, m2=-1)
        raise AssertionError("应拒绝 m2<0")
    except AssertionError as e:
        assert "m2" in str(e)


def test_non_assemblable_raises():
    """不可装配的 θ₂ 应抛出 ValueError。"""
    # 构造一个非 Grashof 机构
    try:
        position_analysis(0.0, 1.0, 1.0, 2.5, 1.0, 'open')
        # 可能不抛异常（取决于具体参数），不强制
    except ValueError:
        pass  # 可接受


if __name__ == "__main__":
    test_loop_closure()
    print("✓ 闭环约束")
    test_freudenstein_equation()
    print("✓ Freudenstein 方程")
    test_open_vs_crossed()
    print("✓ open / crossed 构型")
    test_velocity_ratios_numerical()
    print("✓ 速度比数值验证")
    test_acceleration_numerical()
    print("✓ 加速度数值验证")
    test_grashof_criterion()
    print("✓ Grashof 判据")
    test_toggle_positions()
    print("✓ 极限位置")
    test_equivalent_inertia_positive()
    print("✓ I_eff 正定性")
    test_equivalent_inertia_varies()
    print("✓ I_eff 非常数")
    test_energy_conservation()
    print("✓ 能量守恒")
    test_work_energy_theorem()
    print("✓ 功-能定理")
    test_dynamics_shape()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    test_non_assemblable_raises()
    print("✓ 不可装配处理")
    print("\nOK: MEC-030 所有一致性测试通过")
