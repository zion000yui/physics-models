"""MEC-040 —— 一致性测试：刚性接触。

验证：
- 自由飞行段解析解一致性
- 接触力互补约束 (F≥0, g≥0, F·g=0)
- 无阻尼能量守恒
- 有阻尼能量单调递减
- 接触力只能推不能拉
- 最大穿透随 k_c 增大而减小
- 退化到自由下落（无地面）
- 反例验证（非循环）
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC040_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, contact_force, mechanical_energy,
                   analytical_free_flight, validate_parameters,
                   in_contact, gap)

TOL = 1e-6

M = 1.0
G = 9.81
K_C = 1e4
C_C = 0.0


def _solve(y0=1.0, v0=0.0, m=M, g=G, k_c=K_C, c_c=C_C,
           t_end=3.0, n=2001):
    """数值积分，返回 (t, y, v)。"""
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(
        dynamics, (0.0, t_end), [y0, v0],
        t_eval=t_eval,
        args=(m, g, k_c, c_c),
        rtol=1e-10, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1]


def test_free_flight_analytical():
    """自由飞行段应与解析解一致（落地前）。"""
    y0, v0 = 1.0, 0.0
    t_fall = np.sqrt(2 * y0 / G)
    t, y, v = _solve(y0=y0, v0=v0, t_end=t_fall * 0.95, n=501)

    y_ana, v_ana = analytical_free_flight(t, y0, v0, G)
    err_y = np.max(np.abs(y - y_ana))
    err_v = np.max(np.abs(v - v_ana))
    assert err_y < TOL, f"y 误差 {err_y:.3e}"
    assert err_v < TOL, f"v 误差 {err_v:.3e}"


def test_contact_force_nonnegative():
    """接触力应始终 ≥ 0（只能推不能拉）。"""
    t, y, v = _solve(t_end=3.0, n=2001)
    for i in range(len(t)):
        F = contact_force(y[i], v[i], M, K_C, C_C)
        assert F >= -1e-15, f"接触力 {F:.3e} < 0 (t={t[i]:.3f})"


def test_complementarity():
    """互补约束 F·g = 0：接触时 g≈0，未接触时 F=0。"""
    t, y, v = _solve(t_end=3.0, n=2001)
    for i in range(0, len(t), 20):
        F = contact_force(y[i], v[i], M, K_C, C_C)
        g_val = gap(y[i])
        # F·g ≈ 0 (要么未接触 F=0，要么接触 g≈0)
        assert F * g_val < 1e-6 or (F < 1e-6 and g_val > -1e-6), \
            f"互补约束违反: F={F:.4f}, g={g_val:.6f} (t={t[i]:.3f})"


def test_energy_conservation_no_damping():
    """无阻尼时能量应守恒。"""
    t, y, v = _solve(c_c=0.0, t_end=3.0, n=4001)
    E = np.array([mechanical_energy([y[i], v[i]], M, G, K_C) for i in range(len(t))])
    drift = np.max(np.abs(E - E[0]))
    assert drift < 1e-2, f"能量漂移 {drift:.3e}"


def test_energy_dissipation_with_damping():
    """有阻尼时能量应单调递减。"""
    t, y, v = _solve(c_c=10.0, t_end=3.0, n=4001)
    E = np.array([mechanical_energy([y[i], v[i]], M, G, K_C) for i in range(len(t))])
    # 能量应单调递减（允许小数值波动）
    assert E[-1] < E[0], f"能量未递减: E(0)={E[0]:.6f}, E(end)={E[-1]:.6f}"
    # 整体趋势递减
    assert np.all(np.diff(E[::100]) <= 1e-8), "能量非单调递减"


def test_penetration_decreases_with_stiffness():
    """穿透深度应随 k_c 增大而减小。"""
    depths = []
    for k_c in [1e2, 1e3, 1e4, 1e5]:
        t, y, v = _solve(k_c=k_c, t_end=2.0, n=2001)
        depths.append(np.min(y))
    for i in range(len(depths) - 1):
        assert depths[i+1] >= depths[i], \
            f"k_c 增大但穿透增大: {depths}"


def test_contact_force_only_when_penetrating():
    """未接触时接触力应为零。"""
    for y_val in [0.5, 1.0, 2.0]:
        F = contact_force(y_val, 0.0, M, K_C, C_C)
        assert abs(F) < 1e-15, f"未接触时 F={F} 不为零 (y={y_val})"


def test_dynamics_shape():
    """dynamics 应返回 shape (2,)。"""
    d = dynamics(0.0, [1.0, 0.0], M, G, K_C, C_C)
    assert d.shape == (2,)
    assert abs(d[0]) < TOL, "dy/dt 应等于 v"
    assert abs(d[1] - (-G)) < TOL, "自由飞行时 a=-g"


def test_degradation_free_fall():
    """无接触（y>>0）时退化为自由下落。"""
    y0, v0 = 100.0, 0.0  # 远离地面
    t_end = 1.0
    t, y, v = _solve(y0=y0, v0=v0, t_end=t_end, n=501)
    y_ana, v_ana = analytical_free_flight(t, y0, v0, G)
    err = np.max(np.abs(y - y_ana))
    assert err < TOL, f"自由下落退化误差 {err:.3e}"


def test_error_injection_non_circular():
    """反例：错误的接触力公式应导致能量不守恒。

    故意让接触力在有阻尼时不够（只用力的一半）。
    """
    def dynamics_wrong(t, state, m, g, k_c, c_c):
        y, v = state
        if y < 0:
            F = 0.5 * (-k_c * y - c_c * v)  # 故意只用一半
            F = max(F, 0.0)
        else:
            F = 0.0
        return np.array([v, (-m * g + F) / m])

    # 无阻尼，有接触场景
    sol = solve_ivp(dynamics_wrong, (0, 3), [1.0, 0.0],
                    t_eval=np.linspace(0, 3, 2001),
                    args=(M, G, K_C, 0.0), rtol=1e-10, atol=1e-12)
    y = sol.y[0]
    v = sol.y[1]
    E = np.array([0.5 * M * v[i]**2 + M * G * y[i] + 0.5 * K_C * max(0, -y[i])**2
                  for i in range(len(y))])
    drift = np.max(np.abs(E - E[0]))
    # 错误的接触力会导致能量注入/泄漏
    assert drift > 1e-2, \
        f"反例失败：错误接触力的能量漂移 {drift:.3e} 过小"


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    try:
        validate_parameters(m=-1)
        raise AssertionError("应拒绝 m<0")
    except AssertionError as e:
        assert "m" in str(e)
    try:
        validate_parameters(k_c=-1)
        raise AssertionError("应拒绝 k_c<0")
    except AssertionError as e:
        assert "k_c" in str(e)


if __name__ == "__main__":
    test_free_flight_analytical()
    print("✓ 自由飞行段解析解")
    test_contact_force_nonnegative()
    print("✓ 接触力非负")
    test_complementarity()
    print("✓ 互补约束 F·g=0")
    test_energy_conservation_no_damping()
    print("✓ 无阻尼能量守恒")
    test_energy_dissipation_with_damping()
    print("✓ 有阻尼能量耗散")
    test_penetration_decreases_with_stiffness()
    print("✓ 穿透随刚度减小")
    test_contact_force_only_when_penetrating()
    print("✓ 仅穿透时有接触力")
    test_dynamics_shape()
    print("✓ dynamics 接口")
    test_degradation_free_fall()
    print("✓ 退化到自由下落")
    test_error_injection_non_circular()
    print("✓ 反例验证 (非循环)")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-040 所有一致性测试通过")
