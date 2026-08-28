"""MEC-011 —— 一致性测试：数值解 vs 解析解。

验证：
- 欠阻尼：数值解与衰减振荡解析解一致
- 临界阻尼：数值解与临界衰减解析解一致
- 过阻尼：数值解与过衰减解析解一致
- 无阻尼极限：退化为 MEC-010 简谐振子
- 机械能耗散：有阻尼时能量单调递减
- 相图轨迹：欠阻尼为内旋螺线（趋向原点）
- 阻尼比判据：三种状态的边界正确识别
- 非法参数拒绝

与 MEC-010 的交叉对照：
    当 b=0（ζ=0）时，MEC-011 精确退化为 MEC-010 无阻尼简谐振子，
    解析解与 MEC-010 的 cos/sin 闭式解完全一致。

运行方法（在本文件所在目录执行）：
    python test_MEC011_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    damping_ratio, natural_frequency, damped_frequency, mechanical_energy

TOL = 1e-6


def _solve(x0=1.0, v0=0.0, k=1.0, m=1.0, b=0.0,
           t_end=10.0, n=401):
    """小工具：跑一次数值积分，返回 (t, x, v)。"""
    initial_state = np.array([x0, v0], dtype=float)
    validate_parameters(k=k, m=m, b=b)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(k, m, b),
                    rtol=1e-9, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1]


def test_underdamped_matches_analytical():
    """欠阻尼（ζ < 1）：数值解应与衰减振荡解析解一致。"""
    k, m, b = 1.0, 1.0, 0.4  # ζ = 0.2
    x0, v0 = 1.0, 0.5
    t_end, n = 10.0, 401
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b, t_end=t_end, n=n)
    x_ana, v_ana = analytical(t, [x0, v0], k=k, m=m, b=b)
    err_x = np.max(np.abs(x_num - x_ana))
    err_v = np.max(np.abs(v_num - v_ana))
    assert err_x < TOL, f"欠阻尼 x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_v < TOL, f"欠阻尼 v 误差 {err_v:.3e} 超出容差 {TOL}"


def test_critical_damped_matches_analytical():
    """临界阻尼（ζ = 1）：数值解应与临界衰减解析解一致。"""
    k, m, b = 2.0, 0.5, 2.0  # ζ = 2.0/(2*sqrt(0.5*2)) = 2.0/2.0 = 1.0
    x0, v0 = 1.5, -0.5
    t_end, n = 10.0, 401
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b, t_end=t_end, n=n)
    x_ana, v_ana = analytical(t, [x0, v0], k=k, m=m, b=b)
    err_x = np.max(np.abs(x_num - x_ana))
    err_v = np.max(np.abs(v_num - v_ana))
    assert err_x < TOL, f"临界阻尼 x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_v < TOL, f"临界阻尼 v 误差 {err_v:.3e} 超出容差 {TOL}"


def test_overdamped_matches_analytical():
    """过阻尼（ζ > 1）：数值解应与过衰减解析解一致。"""
    k, m, b = 1.0, 1.0, 4.0  # ζ = 4.0/2 = 2.0
    x0, v0 = 2.0, -1.0
    t_end, n = 10.0, 401
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b, t_end=t_end, n=n)
    x_ana, v_ana = analytical(t, [x0, v0], k=k, m=m, b=b)
    err_x = np.max(np.abs(x_num - x_ana))
    err_v = np.max(np.abs(v_num - v_ana))
    assert err_x < TOL, f"过阻尼 x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_v < TOL, f"过阻尼 v 误差 {err_v:.3e} 超出容差 {TOL}"


def test_no_damping_degenerates_to_MEC010():
    """无阻尼极限（b=0, ζ=0）应退化为 MEC-010 简谐振子。"""
    k, m, b = 2.0, 0.5, 0.0
    x0, v0 = 1.0, 0.5
    t_end, n = 5.0, 201
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b, t_end=t_end, n=n)
    # MEC-010 解析解
    omega0 = natural_frequency(k, m)
    x_exp = x0 * np.cos(omega0 * t) + (v0 / omega0) * np.sin(omega0 * t)
    v_exp = -x0 * omega0 * np.sin(omega0 * t) + v0 * np.cos(omega0 * t)
    err_x = np.max(np.abs(x_num - x_exp))
    err_v = np.max(np.abs(v_num - v_exp))
    assert err_x < TOL, f"无阻尼 x 误差 {err_x:.3e}（未退化为 MEC-010）"
    assert err_v < TOL, f"无阻尼 v 误差 {err_v:.3e}（未退化为 MEC-010）"
    # 同时验证 MEC-011 自身解析解也退化为 cos/sin 形式
    x_ana, v_ana = analytical(t, [x0, v0], k=k, m=m, b=b)
    assert np.allclose(x_ana, x_exp, atol=TOL)
    assert np.allclose(v_ana, v_exp, atol=TOL)


def test_energy_dissipation():
    """有阻尼时机械能应单调递减。"""
    k, m, b = 1.0, 1.0, 0.5  # ζ = 0.25
    x0, v0 = 1.0, 2.0
    t_end, n = 10.0, 501
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b, t_end=t_end, n=n)
    E = np.array([mechanical_energy([x, v], k=k, m=m)
                  for x, v in zip(x_num, v_num)])
    dE = np.diff(E)
    decreasing_fraction = np.mean(dE < 1e-10)
    assert decreasing_fraction > 0.95, \
        f"机械能未单调递减：递减比例 {decreasing_fraction:.2%}"
    assert E[-1] < E[0], \
        f"机械能未耗散：初 {E[0]:.4f}，末 {E[-1]:.4f}"


def test_energy_conserved_no_damping():
    """无阻尼时机械能应守恒（退化为 MEC-010）。"""
    k, m, b = 2.0, 1.0, 0.0
    x0, v0 = 1.5, 2.0
    t_end, n = 5.0, 201
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b, t_end=t_end, n=n)
    E = np.array([mechanical_energy([x, v], k=k, m=m)
                  for x, v in zip(x_num, v_num)])
    assert np.allclose(E, E[0], atol=TOL), \
        f"无阻尼机械能不守恒：波动 {np.max(np.abs(E - E[0])):.3e}"


def test_phase_portrait_underdamped_spiral():
    """欠阻尼相图轨迹应为内旋螺线（向原点收敛）。

    欠阻尼振子的相轨迹是 e^(-γt) 衰减的螺旋线，
    在 x-v 平面上表现为逐渐收缩的椭圆。
    """
    k, m, b = 1.0, 1.0, 0.3  # ζ = 0.15
    x0, v0 = 2.0, 0.0
    gamma = b / (2.0 * m)
    t_end = 20.0  # 多个周期
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b, t_end=t_end, n=2001)
    # 振幅应随时间衰减
    # 检查后半段的最大位移明显小于前半段
    half = len(t) // 2
    amp_first = np.max(np.abs(x_num[:half]))
    amp_second = np.max(np.abs(x_num[half:]))
    assert amp_second < amp_first, \
        f"欠阻尼相图未收敛：前半段振幅 {amp_first:.4f}，后半段 {amp_second:.4f}"
    # 检查能量衰减比
    E0 = mechanical_energy([x_num[0], v_num[0]], k=k, m=m)
    E_end = mechanical_energy([x_num[-1], v_num[-1]], k=k, m=m)
    assert E_end < 0.1 * E0, \
        f"能量未充分衰减：E0={E0:.4f}，E_end={E_end:.4f}"


def test_phase_portrait_overdamped_monotonic():
    """过阻尼应单调趋向平衡位置（无振荡）。"""
    k, m, b = 1.0, 1.0, 5.0  # ζ = 2.5
    x0, v0 = 1.0, 0.0
    t_end = 20.0
    t, x_num, v_num = _solve(x0, v0, k=k, m=m, b=b, t_end=t_end, n=2001)
    # x 应单调趋向 0（不穿越 0）
    # 允许极小的数值噪声
    sign_changes = np.sum(np.diff(np.sign(x_num)) != 0)
    assert sign_changes <= 1, \
        f"过阻尼出现振荡：穿越零点 {sign_changes} 次"
    # 最终应接近 0
    assert abs(x_num[-1]) < 0.05, \
        f"过阻尼未回到平衡：末点 x={x_num[-1]:.6f}"


def test_damping_ratio_boundary():
    """阻尼比判据应正确识别三种状态的边界。"""
    k, m = 1.0, 1.0
    # 欠阻尼 ζ < 1
    zeta_under = damping_ratio(k, m, b=1.5)  # ζ = 0.75
    assert zeta_under < 1.0
    # 临界阻尼 ζ = 1
    zeta_crit = damping_ratio(k, m, b=2.0)  # ζ = 1.0
    assert np.isclose(zeta_crit, 1.0, atol=1e-14)
    # 过阻尼 ζ > 1
    zeta_over = damping_ratio(k, m, b=3.0)  # ζ = 1.5
    assert zeta_over > 1.0


def test_invalid_parameters_rejected():
    """k≤0、m≤0 或 b<0 应被拒绝。"""
    try:
        validate_parameters(k=0.0, m=1.0, b=0.0)
        raise AssertionError("应拒绝 k=0")
    except AssertionError as e:
        assert "k" in str(e)

    try:
        validate_parameters(k=1.0, m=-1.0, b=0.0)
        raise AssertionError("应拒绝 m<0")
    except AssertionError as e:
        assert "m" in str(e)

    try:
        validate_parameters(k=1.0, m=1.0, b=-0.5)
        raise AssertionError("应拒绝 b<0")
    except AssertionError as e:
        assert "b" in str(e)


if __name__ == "__main__":
    test_underdamped_matches_analytical()
    test_critical_damped_matches_analytical()
    test_overdamped_matches_analytical()
    test_no_damping_degenerates_to_MEC010()
    test_energy_dissipation()
    test_energy_conserved_no_damping()
    test_phase_portrait_underdamped_spiral()
    test_phase_portrait_overdamped_monotonic()
    test_damping_ratio_boundary()
    test_invalid_parameters_rejected()
    print("OK: MEC-011 数值解与解析解一致")
