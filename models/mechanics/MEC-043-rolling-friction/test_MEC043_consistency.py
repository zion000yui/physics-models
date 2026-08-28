"""MEC-043 —— 一致性测试：滚动摩擦。

验证：
- 解析解 vs 数值积分
- 停止时间和距离
- 能量单调递减
- 退化到 MEC-024（μ_r=0，纯滚动）
- 摩擦力方向
- 有效质量公式
- 典型转动惯量
- 反例验证（非循环）
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC043_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, analytical, mechanical_energy,
                   effective_mass, rolling_friction_force,
                   rolling_friction_torque, validate_parameters)

TOL = 1e-6

M = 1.0
R = 0.5
I = 0.4 * M * R**2  # 实心球
G = 9.81
MU_R = 0.01


def _solve(v0=3.0, mu_r=MU_R, m=M, R_v=R, I_v=I, g=G,
           t_end=5.0, n=1001):
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), [0.0, v0],
                    t_eval=t_eval, args=(m, R_v, I_v, g, mu_r),
                    rtol=1e-10, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1]


def test_analytical_vs_numerical():
    """解析解应与数值积分一致。"""
    v0 = 3.0
    m_eff = effective_mass(M, I, R)
    a_decel = MU_R * M * G / m_eff
    t_stop = abs(v0) / a_decel

    t, x, v = _solve(v0=v0, t_end=t_stop * 0.9, n=501)
    x_ana, v_ana = analytical(t, v0, M, R, I, G, MU_R)
    err_x = np.max(np.abs(x - x_ana))
    err_v = np.max(np.abs(v - v_ana))
    assert err_x < 1e-4, f"x 误差 {err_x:.3e}"
    assert err_v < 1e-4, f"v 误差 {err_v:.3e}"


def test_stops_correctly():
    """物体应在解析停止时间后停止。"""
    v0 = 3.0
    m_eff = effective_mass(M, I, R)
    a_decel = MU_R * M * G / m_eff
    t_stop = abs(v0) / a_decel
    x_stop = v0**2 / (2 * a_decel)

    t, x, v = _solve(v0=v0, t_end=t_stop * 2, n=2001)
    assert abs(v[-1]) < 0.01, f"未停止: v(end)={v[-1]:.4f}"
    assert abs(x[-1] - x_stop) < 0.1, f"停止距离不符: {x[-1]:.4f} vs {x_stop:.4f}"


def test_energy_monotone_decreasing():
    """能量应单调递减（摩擦耗散）。"""
    t, x, v = _solve(v0=3.0, t_end=5.0, n=2001)
    E = np.array([mechanical_energy([x[i], v[i]], M, R, I) for i in range(len(t))])
    assert E[-1] < E[0], f"能量未递减"
    assert np.all(np.diff(E[::100]) <= 1e-8), "能量非单调递减"


def test_degradation_to_pure_rolling():
    """μ_r=0 时退化为匀速滚动（MEC-024 无耗散）。"""
    t, x, v = _solve(v0=3.0, mu_r=0.0, t_end=2.0, n=501)
    # 匀速：v 恒定，x = v0·t
    assert np.allclose(v, 3.0, atol=1e-4), f"μ_r=0 时速度不恒定"
    assert np.allclose(x, 3.0 * t, atol=1e-4), f"μ_r=0 时位移不线性"


def test_friction_direction():
    """摩擦力应反对运动方向。"""
    F_pos = rolling_friction_force(1.0, M, G, MU_R)
    assert F_pos < 0, f"正速度时摩擦力应<0: {F_pos}"
    F_neg = rolling_friction_force(-1.0, M, G, MU_R)
    assert F_neg > 0, f"负速度时摩擦力应>0: {F_neg}"
    F_zero = rolling_friction_force(0.0, M, G, MU_R)
    assert F_zero == 0, f"静止时摩擦力应为0: {F_zero}"


def test_effective_mass():
    """有效质量应与 MEC-024 一致。"""
    m_eff = effective_mass(M, I, R)
    expected = M + I / R**2
    assert abs(m_eff - expected) < 1e-15
    # 实心球: I=2/5·m·R² → m_eff = m + 2/5·m = 7/5·m
    assert abs(m_eff - 7/5 * M) < 1e-15


def test_typical_inertias():
    """不同形状的转动惯量应给出不同的有效质量。"""
    for name, I_ratio in [("实心球", 0.4), ("实心圆柱", 0.5), ("薄壁球壳", 2/3)]:
        I_test = I_ratio * M * R**2
        m_eff = effective_mass(M, I_test, R)
        expected = M * (1 + I_ratio)
        assert abs(m_eff - expected) < 1e-15


def test_error_injection_non_circular():
    """反例：μ_r 乘2 应导致解析解不匹配。"""
    v0 = 3.0
    m_eff = effective_mass(M, I, R)
    t_stop = abs(v0) / (MU_R * M * G / m_eff)
    t = np.linspace(0, t_stop * 0.8, 501)

    # 正确
    x_c, v_c = analytical(t, v0, M, R, I, G, MU_R)

    # 错误 μ_r 翻倍
    t_w, x_w, v_w = _solve(v0=v0, mu_r=2*MU_R, t_end=t_stop*0.8, n=501)

    err = np.max(np.abs(v_w - v_c))
    assert err > 0.01, f"反例失败：错误μ_r 仍匹配解析解 (err={err:.3e})"


def test_dynamics_shape():
    """dynamics 应返回 shape (2,)。"""
    d = dynamics(0.0, [0.5, 1.0], M, R, I, G, MU_R)
    assert d.shape == (2,)


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    try:
        validate_parameters(mu_r=-0.1)
        raise AssertionError("应拒绝 mu_r<0")
    except AssertionError as e:
        assert "mu_r" in str(e) or "μ_r" in str(e)
    try:
        validate_parameters(R=-1)
        raise AssertionError("应拒绝 R<0")
    except AssertionError as e:
        assert "R" in str(e)


if __name__ == "__main__":
    test_analytical_vs_numerical()
    print("✓ 解析解 vs 数值积分")
    test_stops_correctly()
    print("✓ 停止时间/距离")
    test_energy_monotone_decreasing()
    print("✓ 能量单调递减")
    test_degradation_to_pure_rolling()
    print("✓ 退化到 MEC-024 (μ_r=0)")
    test_friction_direction()
    print("✓ 摩擦力方向")
    test_effective_mass()
    print("✓ 有效质量")
    test_typical_inertias()
    print("✓ 典型转动惯量")
    test_error_injection_non_circular()
    print("✓ 反例验证 (非循环)")
    test_dynamics_shape()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-043 所有一致性测试通过")
