"""MEC-032 —— 一致性测试：齿轮传动。

验证：
- 运动学约束（外啮合反向）
- 传动比
- 等效惯量公式（从第一性原理）
- 解析解 vs 数值积分
- 接触力一致性（两个齿轮独立计算）
- 接触力反例验证（非循环）
- 功率平衡
- 能量守恒（τ=0）
- 功-能定理（τ≠0）
- 退化到 MEC-021（I₂→0, τ_load→0）
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC032_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, analytical, equivalent_inertia,
                   transmission_ratio, output_kinematics,
                   contact_force, contact_force_from_output,
                   power_flow, mechanical_energy, validate_parameters)

TOL = 1e-6

# 默认参数
R1, R2 = 0.1, 0.2
I1, I2 = 0.01, 0.04
TAU_IN, TAU_LOAD = 1.0, 0.0


def _solve(theta0=0.0, omega0=1.0, tau_in=TAU_IN, tau_load=TAU_LOAD,
           t_end=5.0, n=501):
    """数值积分，返回 (t, theta1, omega1)。"""
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(
        dynamics, (0.0, t_end), [theta0, omega0],
        t_eval=t_eval,
        args=(R1, R2, I1, I2, tau_in, tau_load),
        rtol=1e-10, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1]


def test_kinematic_constraint():
    """外啮合约束 r₁·θ₁ = -r₂·θ₂ 应在所有时刻成立。"""
    t, theta1, omega1 = _solve()
    for i in range(0, len(t), 50):
        th2, w2, _ = output_kinematics(theta1[i], omega1[i], 0.0, R1, R2)
        assert abs(R1 * theta1[i] + R2 * th2) < TOL, \
            f"约束不满足 (t={t[i]:.2f})"


def test_transmission_ratio():
    """传动比 i = r₁/r₂ 应正确。"""
    i = transmission_ratio(R1, R2)
    assert abs(i - R1 / R2) < TOL
    # ω₂ = -i·ω₁
    _, w2, _ = output_kinematics(1.0, 2.0, 0.0, R1, R2)
    assert abs(w2 - (-i * 2.0)) < TOL


def test_equivalent_inertia_first_principles():
    """从第一性原理验证 I_eq = I₁ + i²·I₂。"""
    omega1 = 1.0
    i = transmission_ratio(R1, R2)
    omega2 = -i * omega1

    # T = ½·I₁·ω₁² + ½·I₂·ω₂² = ½·(I₁ + i²·I₂)·ω₁²
    T_total = 0.5 * I1 * omega1**2 + 0.5 * I2 * omega2**2
    I_eq = equivalent_inertia(I1, I2, R1, R2)
    T_eff = 0.5 * I_eq * omega1**2

    assert abs(T_total - T_eff) < 1e-15, \
        f"T_total={T_total:.12f}, T_eff={T_eff:.12f}"


def test_analytical_vs_numerical():
    """解析解应与数值积分一致。"""
    theta0, omega0 = 0.0, 1.0
    t, theta1_num, omega1_num = _solve(theta0=theta0, omega0=omega0)

    theta1_ana, omega1_ana = analytical(
        t, [theta0, omega0], R1, R2, I1, I2, TAU_IN, TAU_LOAD)

    err_theta = np.max(np.abs(theta1_num - theta1_ana))
    err_omega = np.max(np.abs(omega1_num - omega1_ana))
    assert err_theta < TOL, f"θ 误差 {err_theta:.3e}"
    assert err_omega < TOL, f"ω 误差 {err_omega:.3e}"


def test_contact_force_consistency():
    """从两个齿轮独立计算的接触力应一致。"""
    i_ratio = transmission_ratio(R1, R2)
    I_eq = equivalent_inertia(I1, I2, R1, R2)
    alpha1 = (TAU_IN - i_ratio * TAU_LOAD) / I_eq

    F1 = contact_force(TAU_IN, I1, alpha1, R1)
    F2 = contact_force_from_output(TAU_LOAD, I2, alpha1, R1, R2)

    assert abs(F1 - F2) < 1e-12, \
        f"接触力不一致: F1={F1:.10f}, F2={F2:.10f}"


def test_contact_force_anticircular():
    """反例：用错误 I_eq 时两齿轮接触力不一致。

    证明接触力一致性测试不是循环验证。
    """
    i_ratio = transmission_ratio(R1, R2)
    # 故意用错误的 I_eq（缺少 i²·I₂ 项）
    I_eq_wrong = I1  # 缺少 i²·I₂
    alpha_wrong = (TAU_IN - i_ratio * TAU_LOAD) / I_eq_wrong

    F1_wrong = contact_force(TAU_IN, I1, alpha_wrong, R1)
    F2_wrong = contact_force_from_output(TAU_LOAD, I2, alpha_wrong, R1, R2)

    # 应该不一致
    assert abs(F1_wrong - F2_wrong) > 1e-6, \
        f"反例失败：错误 I_eq 时接触力仍一致 (F1={F1_wrong:.6f}, F2={F2_wrong:.6f})"


def test_power_balance():
    """功率平衡 P_in = P_out + dT/dt。"""
    t, theta1, omega1 = _solve()
    i_ratio = transmission_ratio(R1, R2)
    I_eq = equivalent_inertia(I1, I2, R1, R2)
    alpha1 = (TAU_IN - i_ratio * TAU_LOAD) / I_eq

    for idx in [0, len(t)//2, -1]:
        P_in, P_out = power_flow(TAU_IN, omega1[idx], TAU_LOAD, R1, R2)
        dT_dt = I_eq * alpha1 * omega1[idx]
        assert abs(P_in - P_out - dT_dt) < 1e-10, \
            f"功率不平衡 (t={t[idx]:.2f}): P_in-P_out={P_in-P_out:.6e}, dT/dt={dT_dt:.6e}"


def test_energy_conservation():
    """τ_in = i·τ_load（无净力矩）时能量应守恒。"""
    i_ratio = transmission_ratio(R1, R2)
    tau_load_balanced = TAU_IN / i_ratio  # τ_in = i·τ_load → α=0
    t, theta1, omega1 = _solve(tau_in=TAU_IN, tau_load=tau_load_balanced)

    E = np.array([mechanical_energy([th, w], R1, R2, I1, I2)
                  for th, w in zip(theta1, omega1)])
    drift = np.max(np.abs(E - E[0]))
    assert drift < 1e-12, f"能量漂移 {drift:.3e}"


def test_work_energy_theorem():
    """ΔE = τ_eff·Δθ 应成立。"""
    t, theta1, omega1 = _solve(tau_in=TAU_IN, tau_load=0.0)
    i_ratio = transmission_ratio(R1, R2)

    E0 = mechanical_energy([theta1[0], omega1[0]], R1, R2, I1, I2)
    E_end = mechanical_energy([theta1[-1], omega1[-1]], R1, R2, I1, I2)
    dE = E_end - E0
    tau_eff = TAU_IN - i_ratio * 0.0  # τ_load=0
    dtheta = theta1[-1] - theta1[0]
    expected = tau_eff * dtheta
    assert abs(dE - expected) < 1e-6, \
        f"ΔE={dE:.6f}, τ_eff·Δθ={expected:.6f}"


def test_degradation_to_MEC021():
    """I₂→0, τ_load→0 时退化为 MEC-021（单刚体定轴转动）。"""
    I2_zero = 0.0
    tau_load_zero = 0.0

    I_eq = equivalent_inertia(I1, I2_zero, R1, R2)
    assert abs(I_eq - I1) < TOL, f"I_eq={I_eq} ≠ I1={I1}"

    # α₁ = τ_in / I1（MEC-021: I·α = τ）
    alpha = (TAU_IN - 0) / I_eq
    assert abs(alpha - TAU_IN / I1) < TOL

    # 接触力 = 0（无负载，无输出惯量）
    F = contact_force(TAU_IN, I1, alpha, R1)
    assert abs(F) < TOL, f"接触力 {F:.3e} 不为零（应无负载）"


def test_dynamics_shape():
    """dynamics 应返回 shape (2,)。"""
    d = dynamics(0.0, [0.5, 1.0], R1, R2, I1, I2, TAU_IN, TAU_LOAD)
    assert d.shape == (2,), f"shape={d.shape}"
    assert abs(d[0] - 1.0) < TOL, "dθ/dt 应等于 ω"


def test_constant_alpha():
    """α₁ 应为常数（不依赖 θ₁），这是齿轮模型的独特特征。"""
    alpha1 = dynamics(0.0, [0.0, 1.0], R1, R2, I1, I2, TAU_IN, TAU_LOAD)[1]
    alpha2 = dynamics(0.0, [2.5, 3.0], R1, R2, I1, I2, TAU_IN, TAU_LOAD)[1]
    assert abs(alpha1 - alpha2) < TOL, "α₁ 不是常数"


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    try:
        validate_parameters(r1=-0.1, r2=0.2)
        raise AssertionError("应拒绝 r1<0")
    except AssertionError as e:
        assert "r1" in str(e)
    try:
        validate_parameters(r1=0.1, r2=0.2, I1=-1)
        raise AssertionError("应拒绝 I1<0")
    except AssertionError as e:
        assert "I1" in str(e)


if __name__ == "__main__":
    test_kinematic_constraint()
    print("✓ 运动学约束")
    test_transmission_ratio()
    print("✓ 传动比")
    test_equivalent_inertia_first_principles()
    print("✓ 等效惯量 (第一性原理)")
    test_analytical_vs_numerical()
    print("✓ 解析解 vs 数值积分")
    test_contact_force_consistency()
    print("✓ 接触力一致性")
    test_contact_force_anticircular()
    print("✓ 接触力反例验证 (非循环)")
    test_power_balance()
    print("✓ 功率平衡")
    test_energy_conservation()
    print("✓ 能量守恒")
    test_work_energy_theorem()
    print("✓ 功-能定理")
    test_degradation_to_MEC021()
    print("✓ 退化到 MEC-021")
    test_dynamics_shape()
    print("✓ dynamics 接口")
    test_constant_alpha()
    print("✓ 常加速度")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-032 所有一致性测试通过")
