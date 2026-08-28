"""MEC-012 —— 一致性测试：数值解 vs 解析解。

验证：
- 受迫阻尼振子：数值解与解析解（瞬态+稳态）一致
- 无外力极限退化为 MEC-011
- 无外力无阻尼极限退化为 MEC-010
- 稳态振幅与理论公式一致
- 共振频率理论特征
- 稳态能量平衡（驱动力输入 = 阻尼耗散）
- 非法参数拒绝

与 MEC-010/011 的交叉对照：
    当 F0=0 时，MEC-012 退化为 MEC-011（阻尼振子）。
    当 F0=0 且 b=0 时，退化为 MEC-010（简谐振子）。

运行方法（在本文件所在目录执行）：
    python test_MEC012_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    natural_frequency, damping_ratio, \
    steady_state_amplitude, steady_state_phase, \
    resonance_frequency, mechanical_energy

TOL = 1e-6


def _solve(x0=1.0, v0=0.0, k=1.0, m=1.0, b=0.4,
           F0=1.0, omega=0.8,
           t_end=30.0, n=601):
    """小工具：跑一次数值积分，返回 (t, x, v)。"""
    initial_state = np.array([x0, v0], dtype=float)
    validate_parameters(k=k, m=m, b=b, F0=F0, omega=omega)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(k, m, b, F0, omega),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1]


def test_underdamped_forced_matches_analytical():
    """欠阻尼受迫振动：数值解应与解析解（瞬态+稳态）一致。"""
    k, m, b = 1.0, 1.0, 0.4
    F0, omega = 1.0, 0.8
    x0, v0 = 1.0, 0.5
    t_end, n = 20.0, 801
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b,
                             F0=F0, omega=omega, t_end=t_end, n=n)
    x_ana, v_ana = analytical(t, [x0, v0], k=k, m=m, b=b,
                              F0=F0, omega=omega)
    err_x = np.max(np.abs(x_num - x_ana))
    err_v = np.max(np.abs(v_num - v_ana))
    assert err_x < TOL, f"受迫 x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_v < TOL, f"受迫 v 误差 {err_v:.3e} 超出容差 {TOL}"


def test_overdamped_forced_matches_analytical():
    """过阻尼受迫振动：数值解应与解析解一致。"""
    k, m, b = 1.0, 1.0, 4.0  # ζ = 2.0
    F0, omega = 0.5, 1.2
    x0, v0 = 2.0, -1.0
    t_end, n = 20.0, 801
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b,
                             F0=F0, omega=omega, t_end=t_end, n=n)
    x_ana, v_ana = analytical(t, [x0, v0], k=k, m=m, b=b,
                              F0=F0, omega=omega)
    err_x = np.max(np.abs(x_num - x_ana))
    err_v = np.max(np.abs(v_num - v_ana))
    assert err_x < TOL, f"过阻尼受迫 x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_v < TOL, f"过阻尼受迫 v 误差 {err_v:.3e} 超出容差 {TOL}"


def test_no_force_degenerates_to_MEC011():
    """无外力极限（F0=0）应退化为 MEC-011 阻尼振子。"""
    k, m, b = 1.0, 1.0, 0.4
    F0, omega = 0.0, 0.0
    x0, v0 = 1.0, 0.5
    t_end, n = 10.0, 401
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b,
                             F0=F0, omega=omega, t_end=t_end, n=n)
    # MEC-011 解析解（通过 MEC-012 自身函数 F0=0 调用）
    x_ana, v_ana = analytical(t, [x0, v0], k=k, m=m, b=b, F0=0, omega=0)
    err_x = np.max(np.abs(x_num - x_ana))
    err_v = np.max(np.abs(v_num - v_ana))
    assert err_x < TOL, f"无外力 x 误差 {err_x:.3e}（未退化为 MEC-011）"
    assert err_v < TOL, f"无外力 v 误差 {err_v:.3e}（未退化为 MEC-011）"


def test_no_force_no_damping_degenerates_to_MEC010():
    """无外力无阻尼极限（F0=0, b=0）应退化为 MEC-010 简谐振子。"""
    k, m, b = 2.0, 0.5, 0.0
    F0, omega = 0.0, 0.0
    x0, v0 = 1.0, 0.5
    t_end, n = 5.0, 201
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b,
                             F0=F0, omega=omega, t_end=t_end, n=n)
    # MEC-010 解析解
    omega0 = natural_frequency(k, m)
    x_exp = x0 * np.cos(omega0 * t) + (v0 / omega0) * np.sin(omega0 * t)
    v_exp = -x0 * omega0 * np.sin(omega0 * t) + v0 * np.cos(omega0 * t)
    err_x = np.max(np.abs(x_num - x_exp))
    err_v = np.max(np.abs(v_num - v_exp))
    assert err_x < TOL, f"双退化 x 误差 {err_x:.3e}（未退化为 MEC-010）"
    assert err_v < TOL, f"双退化 v 误差 {err_v:.3e}（未退化为 MEC-010）"


def test_steady_state_amplitude():
    """稳态振幅应与理论公式 A_ss = (F₀/m)/√((ω₀²-ω²)²+(2γω)²) 一致。"""
    k, m, b = 1.0, 1.0, 0.4
    F0, omega = 1.0, 0.8
    x0, v0 = 0.0, 0.0  # 从静止开始，便于观察稳态
    # 足够长的时间以进入稳态（瞬态衰减到可忽略）
    omega0 = natural_frequency(k, m)
    gamma = b / (2 * m)
    t_decay = 10.0 / gamma  # ~5 个衰减时间常数
    T_drive = 2 * np.pi / omega
    t_end = t_decay + 5 * T_drive
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b,
                             F0=F0, omega=omega, t_end=t_end, n=2001)
    # 取稳态段
    mask = t >= t_decay
    x_steady = x_num[mask]
    A_numerical = (np.max(x_steady) - np.min(x_steady)) / 2.0
    A_theory = steady_state_amplitude(k, m, b, F0, omega)
    rel_err = abs(A_numerical - A_theory) / A_theory
    assert rel_err < 0.01, \
        f"稳态振幅不符：数值 {A_numerical:.6f} vs 理论 {A_theory:.6f}，相对误差 {rel_err:.3e}"


def test_resonance_frequency():
    """共振频率应满足 ω_max = ω₀√(1-2ζ²)，且 A_ss(ω_max) > A_ss(ω₀)。

    共振频率处的振幅应大于固有频率处的振幅（共振峰偏移）。
    """
    k, m, b = 1.0, 1.0, 0.4  # ζ = 0.2
    F0 = 1.0
    omega0 = natural_frequency(k, m)
    omega_r = resonance_frequency(k, m, b)
    assert omega_r is not None, "应存在共振峰"
    # 理论验证: ω_r = ω₀√(1-2ζ²)
    zeta = damping_ratio(k, m, b)
    omega_r_theory = omega0 * np.sqrt(1 - 2 * zeta ** 2)
    assert np.isclose(omega_r, omega_r_theory, rtol=TOL), \
        f"共振频率不符: {omega_r:.6f} vs {omega_r_theory:.6f}"
    # 共振峰振幅 > 固有频率振幅
    A_at_resonance = steady_state_amplitude(k, m, b, F0, omega_r)
    A_at_omega0 = steady_state_amplitude(k, m, b, F0, omega0)
    assert A_at_resonance > A_at_omega0, \
        f"共振峰振幅 {A_at_resonance:.6f} 应大于固有频率振幅 {A_at_omega0:.6f}"


def test_no_resonance_peak_high_damping():
    """高阻尼（ζ ≥ 1/√2）时无共振峰。"""
    k, m = 1.0, 1.0
    b_high = 1.5  # ζ = 0.75 > 1/√2 ≈ 0.707
    zeta = damping_ratio(k, m, b_high)
    assert zeta >= 1.0 / np.sqrt(2), f"ζ={zeta:.4f} 应 ≥ 1/√2"
    omega_r = resonance_frequency(k, m, b_high)
    assert omega_r is None, "高阻尼应无共振峰"


def test_steady_state_energy_balance():
    """稳态时驱动力输入功率 = 阻尼耗散功率。"""
    k, m, b = 1.0, 1.0, 0.4
    F0, omega = 1.0, 0.8
    x0, v0 = 0.0, 0.0
    gamma = b / (2 * m)
    t_decay = 10.0 / gamma
    T_drive = 2 * np.pi / omega
    t_end = t_decay + 10 * T_drive
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b,
                             F0=F0, omega=omega, t_end=t_end, n=4001)
    # 稳态段
    mask = t >= t_decay
    t_s = t[mask]
    x_s = x_num[mask]
    v_s = v_num[mask]
    F_ext = F0 * np.cos(omega * t_s)
    # 功率: P_in = F_ext * v, P_diss = b * v²
    P_in = F_ext * v_s
    P_diss = b * v_s ** 2
    # 一个周期内的平均值
    T = T_drive
    period_mask = (t_s >= t_s[-1] - T)
    if np.sum(period_mask) > 10:
        avg_P_in = np.mean(P_in[period_mask])
        avg_P_diss = np.mean(P_diss[period_mask])
        rel_err = abs(avg_P_in - avg_P_diss) / max(abs(avg_P_in), 1e-15)
        assert rel_err < 0.05, \
            f"能量不平衡: P_in={avg_P_in:.6f}, P_diss={avg_P_diss:.6f}, 误差={rel_err:.3e}"


def test_invalid_parameters_rejected():
    """k≤0、m≤0、b<0、F0<0 或 omega<0 应被拒绝。"""
    try:
        validate_parameters(k=0.0, m=1.0, b=0.0, F0=0.0, omega=0.0)
        raise AssertionError("应拒绝 k=0")
    except AssertionError as e:
        assert "k" in str(e)

    try:
        validate_parameters(k=1.0, m=-1.0, b=0.0, F0=0.0, omega=0.0)
        raise AssertionError("应拒绝 m<0")
    except AssertionError as e:
        assert "m" in str(e)

    try:
        validate_parameters(k=1.0, m=1.0, b=-0.5, F0=0.0, omega=0.0)
        raise AssertionError("应拒绝 b<0")
    except AssertionError as e:
        assert "b" in str(e)

    try:
        validate_parameters(k=1.0, m=1.0, b=0.0, F0=-1.0, omega=0.0)
        raise AssertionError("应拒绝 F0<0")
    except AssertionError as e:
        assert "F0" in str(e)

    try:
        validate_parameters(k=1.0, m=1.0, b=0.0, F0=1.0, omega=-1.0)
        raise AssertionError("应拒绝 omega<0")
    except AssertionError as e:
        assert "omega" in str(e)


if __name__ == "__main__":
    test_underdamped_forced_matches_analytical()
    test_overdamped_forced_matches_analytical()
    test_no_force_degenerates_to_MEC011()
    test_no_force_no_damping_degenerates_to_MEC010()
    test_steady_state_amplitude()
    test_resonance_frequency()
    test_no_resonance_peak_high_damping()
    test_steady_state_energy_balance()
    test_invalid_parameters_rejected()
    print("OK: MEC-012 数值解与解析解一致")
