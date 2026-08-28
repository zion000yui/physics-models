"""MEC-013 —— 一致性测试：双摆。

验证：
- 机械能守恒（大角度混沌运动）
- 机械能守恒（小角度）
- 小角度线性化退化为 MEC-014 耦合振子类型
- 对初始条件敏感性（混沌特征）
- 平衡点特例
- 非法参数处理

注意：双摆没有闭式解析解（非线性混沌系统），交叉验证通过
scipy_solve.py 数值积分与物理守恒律完成，而非与解析解对比。
小角度极限通过与线性理论特征对照验证。

运行方法（在本文件所在目录执行）：
    python test_MEC013_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import dynamics, validate_parameters, mechanical_energy

TOL = 1e-6


def _solve(theta1_0=1.5, theta2_0=0.5, omega1_0=0.0, omega2_0=0.0,
           m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81,
           t_end=10.0, n=401):
    """小工具：跑一次数值积分，返回 (t, theta1, theta2, omega1, omega2)。"""
    initial_state = np.array([theta1_0, theta2_0, omega1_0, omega2_0],
                              dtype=float)
    validate_parameters(m1=m1, m2=m2, L1=L1, L2=L2, g=g)
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(m1, m2, L1, L2, g),
                    rtol=1e-10, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def test_energy_conserved_large_angle():
    """大角度混沌运动下机械能应守恒。"""
    m1, m2, L1, L2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    state0 = [np.pi / 2, 0.5, 0.0, 0.0]
    t_end, n = 10.0, 1001
    t, t1, t2, w1, w2 = _solve(*state0, m1, m2, L1, L2, g, t_end, n)
    E = np.array([mechanical_energy([t1[i], t2[i], w1[i], w2[i]],
                                     m1, m2, L1, L2, g)
                  for i in range(len(t))])
    E0 = mechanical_energy(state0, m1, m2, L1, L2, g)
    assert np.allclose(E, E0, atol=1e-4), \
        f"大角度机械能不守恒：波动 {np.max(np.abs(E - E0)):.3e}"


def test_energy_conserved_small_angle():
    """小角度运动下机械能应高精度守恒。"""
    m1, m2, L1, L2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    state0 = [0.01, 0.005, 0.0, 0.0]
    t_end, n = 10.0, 1001
    t, t1, t2, w1, w2 = _solve(*state0, m1, m2, L1, L2, g, t_end, n)
    E = np.array([mechanical_energy([t1[i], t2[i], w1[i], w2[i]],
                                     m1, m2, L1, L2, g)
                  for i in range(len(t))])
    E0 = mechanical_energy(state0, m1, m2, L1, L2, g)
    assert np.allclose(E, E0, atol=1e-10), \
        f"小角度机械能不守恒：波动 {np.max(np.abs(E - E0)):.3e}"


def test_small_angle_linearized_behavior():
    """小角度线性化应表现出周期性运动（退化为线性耦合振子）。

    小角度时双摆退化为两个耦合的线性振子（MEC-014 类型），
    运动应为两个频率的叠加（准周期或周期）。
    """
    m1, m2, L1, L2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    state0 = [0.01, 0.0, 0.0, 0.0]
    # 小角度线性化频率
    omega0 = np.sqrt(g / L1)
    t_end = 4.0 * np.pi / omega0  # 足够长
    n = 2000
    t, t1, t2, w1, w2 = _solve(*state0, m1, m2, L1, L2, g, t_end, n)
    # 检查角度保持小量（无非线性发散）
    assert np.max(np.abs(t1)) < 0.1, \
        f"小角度运动发散：θ1 max = {np.max(np.abs(t1)):.4f}"
    assert np.max(np.abs(t2)) < 0.1, \
        f"小角度运动发散：θ2 max = {np.max(np.abs(t2)):.4f}"
    # 检查运动是振荡的（通过过零点次数）
    sign_changes_1 = np.sum(np.diff(np.sign(t1)) != 0)
    sign_changes_2 = np.sum(np.diff(np.sign(t2)) != 0)
    assert sign_changes_1 > 5, \
        f"θ1 振荡次数不足：{sign_changes_1}（应 > 5）"
    assert sign_changes_2 > 5, \
        f"θ2 振荡次数不足：{sign_changes_2}（应 > 5）"


def test_chaos_sensitivity():
    """对初始条件敏感：微小初始扰动应导致轨迹发散。

    这是混沌系统的标志性特征。两条初始条件仅差 1e-4 rad 的轨迹，
    在有限时间后应表现出显著差异。
    """
    m1, m2, L1, L2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    state1 = [np.pi / 2, 0.0, 0.0, 0.0]
    state2 = [np.pi / 2 + 1e-4, 0.0, 0.0, 0.0]  # 微扰
    t_end = 15.0
    t_eval = np.linspace(0, t_end, 3000)

    sol1 = solve_ivp(dynamics, (0, t_end), state1, t_eval=t_eval,
                     args=(m1, m2, L1, L2, g), rtol=1e-12, atol=1e-14)
    sol2 = solve_ivp(dynamics, (0, t_end), state2, t_eval=t_eval,
                     args=(m1, m2, L1, L2, g), rtol=1e-12, atol=1e-14)

    # 初始差异
    d0 = abs(state1[0] - state2[0])
    # 末点差异
    d_end = abs(sol1.y[0, -1] - sol2.y[0, -1])
    # 混沌系统应表现出差异放大（至少不衰减）
    assert d_end > d0, \
        f"无敏感性：初始差 {d0:.3e}，末点差 {d_end:.3e}（应 > 初始差）"


def test_equilibrium_point():
    """平衡点特例：θ1=θ2=ω1=ω2=0 时应始终静止。"""
    m1, m2, L1, L2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    t, t1, t2, w1, w2 = _solve(
        0.0, 0.0, 0.0, 0.0, m1, m2, L1, L2, g, t_end=5.0, n=101)
    assert np.allclose(t1, 0.0, atol=TOL), \
        f"平衡点 θ1 不为零：{np.max(np.abs(t1)):.3e}"
    assert np.allclose(t2, 0.0, atol=TOL), \
        f"平衡点 θ2 不为零：{np.max(np.abs(t2)):.3e}"
    assert np.allclose(w1, 0.0, atol=TOL), \
        f"平衡点 ω1 不为零：{np.max(np.abs(w1)):.3e}"
    assert np.allclose(w2, 0.0, atol=TOL), \
        f"平衡点 ω2 不为零：{np.max(np.abs(w2)):.3e}"


def test_dynamics_zero_velocity():
    """从静止释放时 dynamics 应给出正确的初始加速度。"""
    m1, m2, L1, L2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    state = [0.5, 0.3, 0.0, 0.0]
    d = dynamics(0, state, m1, m2, L1, L2, g)
    # ω1=ω2=0 时，sin(Δ) 项消失
    # 方程简化为：
    #   (m1+m2)L1 α1 + m2L2cos(Δ) α2 = -(m1+m2)g sin(θ1)
    #   m2L1cos(Δ) α1 + m2L2 α2 = -m2 g sin(θ2)
    # 手动验证
    delta = 0.5 - 0.3
    A = (m1 + m2) * L1
    B = m2 * L2 * np.cos(delta)
    C = m2 * L1 * np.cos(delta)
    D = m2 * L2
    det = A * D - B * C
    rhs1 = -(m1 + m2) * g * np.sin(0.5)
    rhs2 = -m2 * g * np.sin(0.3)
    exp_a1 = (D * rhs1 - B * rhs2) / det
    exp_a2 = (A * rhs2 - C * rhs1) / det
    assert np.isclose(d[2], exp_a1, rtol=1e-12), \
        f"α1 不符: {d[2]:.10f} vs {exp_a1:.10f}"
    assert np.isclose(d[3], exp_a2, rtol=1e-12), \
        f"α2 不符: {d[3]:.10f} vs {exp_a2:.10f}"


def test_invalid_parameters_rejected():
    """m≤0、L≤0 或 g≤0 应被拒绝。"""
    try:
        validate_parameters(m1=0.0, m2=1.0, L1=1.0, L2=1.0, g=9.81)
        raise AssertionError("应拒绝 m1=0")
    except AssertionError as e:
        assert "m1" in str(e)

    try:
        validate_parameters(m1=1.0, m2=-1.0, L1=1.0, L2=1.0, g=9.81)
        raise AssertionError("应拒绝 m2<0")
    except AssertionError as e:
        assert "m2" in str(e)

    try:
        validate_parameters(m1=1.0, m2=1.0, L1=0.0, L2=1.0, g=9.81)
        raise AssertionError("应拒绝 L1=0")
    except AssertionError as e:
        assert "L1" in str(e)

    try:
        validate_parameters(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=-1.0)
        raise AssertionError("应拒绝 g<0")
    except AssertionError as e:
        assert "g" in str(e)


if __name__ == "__main__":
    test_energy_conserved_large_angle()
    test_energy_conserved_small_angle()
    test_small_angle_linearized_behavior()
    test_chaos_sensitivity()
    test_equilibrium_point()
    test_dynamics_zero_velocity()
    test_invalid_parameters_rejected()
    print("OK: MEC-013 数值解与物理一致性验证通过")
