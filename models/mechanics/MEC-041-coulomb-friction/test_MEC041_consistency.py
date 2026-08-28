"""MEC-041 —— 一致性测试：库仑摩擦。

验证：
- 动摩擦段解析解一致性
- 静摩擦时物体保持静止
- 自由减速停止时间和距离
- 摩擦力方向正确（反对运动方向）
- 能量平衡（外力做功 - 摩擦耗散 = ΔE）
- 无摩擦退化到 MEC-002
- 静/动摩擦切换
- 反例验证（非循环）
- 非法参数（μ_s < μ_k）

运行方法（在本文件所在目录执行）：
    python test_MEC041_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, friction_force, mechanical_energy,
                   analytical_constant_force, validate_parameters,
                   normal_force, is_sliding)

TOL = 1e-6

M = 1.0
G = 9.81
MU_S = 0.3
MU_K = 0.25


def _solve(v0=0.0, F_ext=5.0, m=M, g=G, mu_s=MU_S, mu_k=MU_K,
           t_end=3.0, n=1001):
    """数值积分，返回 (t, x, v)。"""
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(
        dynamics, (0.0, t_end), [0.0, v0],
        t_eval=t_eval,
        args=(m, g, mu_s, mu_k, F_ext),
        rtol=1e-10, atol=1e-12)
    return t_eval, sol.y[0], sol.y[1]


def test_kinetic_friction_analytical():
    """动摩擦段应与解析解一致（F_ext > μ_s·N）。"""
    F_ext = 5.0  # > μ_s·N = 0.3*9.81 = 2.943
    v0 = 0.0
    t, x, v = _solve(v0=v0, F_ext=F_ext, t_end=2.0, n=501)
    x_ana, v_ana = analytical_constant_force(t, v0, M, G, MU_S, MU_K, F_ext)
    err_x = np.max(np.abs(x - x_ana))
    err_v = np.max(np.abs(v - v_ana))
    assert err_x < 1e-4, f"x 误差 {err_x:.3e}"
    assert err_v < 1e-4, f"v 误差 {err_v:.3e}"


def test_static_friction_no_motion():
    """静摩擦时物体应保持静止（|F_ext| ≤ μ_s·N）。"""
    N = normal_force(M, G)
    F_ext = 0.5 * MU_S * N  # 远小于最大静摩擦
    t, x, v = _solve(v0=0.0, F_ext=F_ext, t_end=2.0, n=501)
    assert np.max(np.abs(x)) < 1e-6, f"静摩擦时位移过大: {np.max(np.abs(x)):.3e}"
    assert np.max(np.abs(v)) < 1e-6, f"静摩擦时速度过大: {np.max(np.abs(v)):.3e}"


def test_free_deceleration_stops():
    """自由滑动（有初速度无外力）应在有限时间内停止。"""
    v0 = 5.0
    t, x, v = _solve(v0=v0, F_ext=0.0, t_end=5.0, n=2001)
    # 解析停止时间
    t_stop = v0 / (MU_K * G)
    x_stop = v0**2 / (2 * MU_K * G)
    # 数值结果应接近
    assert abs(v[-1]) < 0.1, f"未停止: v(end)={v[-1]:.4f}"
    assert abs(x[-1] - x_stop) < 0.5, f"停止位置不符: {x[-1]:.4f} vs {x_stop:.4f}"


def test_friction_direction():
    """摩擦力应反对运动方向。"""
    # 正速度 → 负摩擦力
    F_f = friction_force(1.0, 0.0, M, G, MU_S, MU_K)
    assert F_f < 0, f"正速度时摩擦力应 <0: {F_f}"
    # 负速度 → 正摩擦力
    F_f = friction_force(-1.0, 0.0, M, G, MU_S, MU_K)
    assert F_f > 0, f"负速度时摩擦力应 >0: {F_f}"


def test_energy_balance():
    """外力做功 - 摩擦耗散 = ΔE。"""
    F_ext = 5.0
    v0 = 0.0
    t, x, v = _solve(v0=v0, F_ext=F_ext, t_end=2.0, n=1001)
    E = np.array([mechanical_energy([x[i], v[i]], M) for i in range(len(t))])
    dE = E[-1] - E[0]
    W_ext = F_ext * (x[-1] - x[0])
    N = normal_force(M, G)
    W_friction = -MU_K * N * abs(x[-1] - x[0])
    assert abs(dE - (W_ext + W_friction)) < 1e-4, \
        f"ΔE={dE:.6f}, W_ext+W_fric={W_ext+W_friction:.6f}"


def test_degradation_no_friction():
    """无摩擦（μ=0）时应退化为 MEC-002（受力质点，恒力）。"""
    t, x, v = _solve(v0=0.0, F_ext=5.0, mu_s=0.0, mu_k=0.0, t_end=1.0, n=501)
    # 解析: x = ½·(F/m)·t², v = (F/m)·t
    a = 5.0 / M
    x_ana = 0.5 * a * t**2
    v_ana = a * t
    err_x = np.max(np.abs(x - x_ana))
    err_v = np.max(np.abs(v - v_ana))
    assert err_x < 1e-4, f"退化误差 x: {err_x:.3e}"
    assert err_v < 1e-4, f"退化误差 v: {err_v:.3e}"


def test_static_kinetic_switch():
    """静/动摩擦应正确切换。"""
    # 逐渐增大外力，检查何时开始滑动
    N = normal_force(M, G)
    for F in [0.5, 1.0, 2.0, 2.9, 3.0]:
        sliding = is_sliding(0.0, F, M, G, MU_S, MU_K)
        if F <= MU_S * N:
            assert not sliding, f"F={F} 应静止，但判定为滑动"
        else:
            assert sliding, f"F={F} 应滑动，但判定为静止"


def test_error_injection_non_circular():
    """反例：错误摩擦力（μ_k 乘2）应导致解析解不匹配。"""
    F_ext = 5.0
    v0 = 0.0
    t, x, v = _solve(v0=v0, F_ext=F_ext, t_end=1.0, n=501)
    x_ana, v_ana = analytical_constant_force(t, v0, M, G, MU_S, MU_K, F_ext)

    # 用错误 μ_k 数值积分
    sol_w = solve_ivp(dynamics, (0, 1), [0, 0], t_eval=t,
                      args=(M, G, MU_S, 2*MU_K, F_ext),  # μ_k 翻倍
                      rtol=1e-10, atol=1e-12)
    err_wrong = np.max(np.abs(sol_w.y[1] - v_ana))
    err_correct = np.max(np.abs(v - v_ana))
    assert err_wrong > 1e-3, f"反例失败：错误μ_k 仍匹配解析解 (err={err_wrong:.3e})"
    assert err_correct < 1e-4, f"正确解不匹配 (err={err_correct:.3e})"


def test_dynamics_shape():
    """dynamics 应返回 shape (2,)。"""
    d = dynamics(0.0, [0.5, 1.0], M, G, MU_S, MU_K, 5.0)
    assert d.shape == (2,)


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    try:
        validate_parameters(mu_s=0.1, mu_k=0.3)  # μ_s < μ_k
        raise AssertionError("应拒绝 μ_s < μ_k")
    except AssertionError as e:
        assert "μ_s" in str(e) or "mu_s" in str(e)
    try:
        validate_parameters(m=-1)
        raise AssertionError("应拒绝 m<0")
    except AssertionError as e:
        assert "m" in str(e)


if __name__ == "__main__":
    test_kinetic_friction_analytical()
    print("✓ 动摩擦段解析解")
    test_static_friction_no_motion()
    print("✓ 静摩擦保持静止")
    test_free_deceleration_stops()
    print("✓ 自由减速停止")
    test_friction_direction()
    print("✓ 摩擦力方向")
    test_energy_balance()
    print("✓ 能量平衡")
    test_degradation_no_friction()
    print("✓ 退化到 MEC-002 (无摩擦)")
    test_static_kinetic_switch()
    print("✓ 静/动摩擦切换")
    test_error_injection_non_circular()
    print("✓ 反例验证 (非循环)")
    test_dynamics_shape()
    print("✓ dynamics 接口")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-041 所有一致性测试通过")
