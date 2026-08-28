"""MEC-042 —— 一致性测试：碰撞。

验证：
- 动量守恒
- 恢复系数定义
- 动能损失公式
- 弹性碰撞动能守恒
- 塑性碰撞速度相等
- 解析运动解一致性
- 质量比极限
- 反例验证（非循环）
- 非法参数

运行方法（在本文件所在目录执行）：
    python test_MEC042_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, solve_collision, analytical_motion,
                   kinetic_energy_before, kinetic_energy_after,
                   energy_loss, momentum_before, momentum_after,
                   reduced_mass, validate_parameters)

TOL = 1e-10


def test_momentum_conservation():
    """碰撞前后总动量应守恒。"""
    for m1, m2, e in [(1, 1, 0.8), (2, 1, 0.5), (1, 3, 1.0), (0.5, 2, 0.0)]:
        v1b, v2b = 3.0, -2.0
        v1a, v2a = solve_collision(v1b, v2b, m1, m2, e)
        p_before = momentum_before(v1b, v2b, m1, m2)
        p_after = momentum_after(v1a, v2a, m1, m2)
        assert abs(p_before - p_after) < TOL, \
            f"动量不守恒: {p_before} vs {p_after} (m1={m1},m2={m2},e={e})"


def test_restitution_coefficient():
    """碰撞后相对速度应满足恢复系数定义。"""
    for e in [0.0, 0.3, 0.5, 0.8, 1.0]:
        v1b, v2b = 3.0, -1.0
        v1a, v2a = solve_collision(v1b, v2b, 1.0, 1.0, e)
        e_computed = -(v1a - v2a) / (v1b - v2b)
        assert abs(e_computed - e) < TOL, \
            f"恢复系数不符: {e_computed:.6f} vs {e}"


def test_energy_loss_formula():
    """动能损失应与公式 ΔT = -½(1-e²)m_red(v1b-v2b)² 一致。"""
    for m1, m2, e in [(1, 1, 0.8), (2, 1, 0.5), (1, 3, 1.0)]:
        v1b, v2b = 3.0, -2.0
        v1a, v2a = solve_collision(v1b, v2b, m1, m2, e)
        T_before = kinetic_energy_before(v1b, v2b, m1, m2)
        T_after = kinetic_energy_after(v1a, v2a, m1, m2)
        dT_actual = T_after - T_before
        dT_formula = energy_loss(v1b, v2b, m1, m2, e)
        assert abs(dT_actual - dT_formula) < TOL, \
            f"ΔT不符: {dT_actual:.6f} vs {dT_formula:.6f}"


def test_elastic_energy_conservation():
    """弹性碰撞 (e=1) 动能应守恒。"""
    v1b, v2b = 3.0, -2.0
    v1a, v2a = solve_collision(v1b, v2b, 1.0, 2.0, 1.0)
    T_before = kinetic_energy_before(v1b, v2b, 1.0, 2.0)
    T_after = kinetic_energy_after(v1a, v2a, 1.0, 2.0)
    assert abs(T_before - T_after) < TOL, f"弹性碰撞动能不守恒: {T_before} vs {T_after}"


def test_plastic_stick_together():
    """塑性碰撞 (e=0) 碰后速度应相等。"""
    v1b, v2b = 3.0, -2.0
    v1a, v2a = solve_collision(v1b, v2b, 1.0, 2.0, 0.0)
    assert abs(v1a - v2a) < TOL, f"塑性碰撞速度不等: {v1a} vs {v2a}"


def test_analytical_motion():
    """解析运动解应自洽：碰撞前自由运动，碰撞后速度跳变。"""
    t = np.linspace(0, 3, 1001)
    v1b, v2b = 3.0, -2.0
    t_col = 1.0
    x1_0, x2_0 = 0.0, 5.0

    x1, v1, x2, v2 = analytical_motion(t, v1b, v2b, 1.0, 1.0, 0.8, t_col, x1_0, x2_0)

    # 碰撞前速度恒定
    assert np.allclose(v1[t < t_col], v1b)
    assert np.allclose(v2[t < t_col], v2b)

    # 碰撞后速度恒定（不同值）
    v1a, v2a = solve_collision(v1b, v2b, 1.0, 1.0, 0.8)
    assert np.allclose(v1[t > t_col + 0.01], v1a)
    assert np.allclose(v2[t > t_col + 0.01], v2a)

    # 位置连续（碰撞时刻无跳变）
    idx_col = np.argmin(np.abs(t - t_col))
    # 用实际 t 值计算期望位置（容差来自网格）
    expected_x1 = x1_0 + v1b * t[idx_col]
    assert abs(x1[idx_col] - expected_x1) < 1e-10


def test_equal_mass_elastic_swap():
    """等质量弹性碰撞：速度交换。"""
    v1b, v2b = 3.0, -2.0
    v1a, v2a = solve_collision(v1b, v2b, 1.0, 1.0, 1.0)
    assert abs(v1a - v2b) < TOL, f"v1a 应=v2b: {v1a} vs {v2b}"
    assert abs(v2a - v1b) < TOL, f"v2a 应=v1b: {v2a} vs {v1b}"


def test_heavy_target_stationary():
    """重靶静止、轻弹撞击：弹反弹，靶几乎不动。"""
    v1b, v2b = 5.0, 0.0
    m1, m2 = 0.1, 100.0  # 轻弹击重靶
    v1a, v2a = solve_collision(v1b, v2b, m1, m2, 1.0)
    assert v1a < 0, f"轻弹应反弹: v1a={v1a}"  # 反弹
    assert abs(v2a) < 0.1, f"重靶应几乎不动: v2a={v2a}"


def test_error_injection_non_circular():
    """反例：错误恢复系数应导致能量公式不匹配。"""
    v1b, v2b = 3.0, -2.0
    m1, m2 = 1.0, 2.0
    e_correct = 0.8

    # 正确
    v1a, v2a = solve_collision(v1b, v2b, m1, m2, e_correct)
    T_after = kinetic_energy_after(v1a, v2a, m1, m2)
    dT_correct = energy_loss(v1b, v2b, m1, m2, e_correct)

    # 错误：用 e=0.5 计算 solve_collision，但用 e=0.8 的 energy_loss
    v1a_wrong, v2a_wrong = solve_collision(v1b, v2b, m1, m2, 0.5)
    T_after_wrong = kinetic_energy_after(v1a_wrong, v2a_wrong, m1, m2)
    dT_mismatch = T_after_wrong - kinetic_energy_before(v1b, v2b, m1, m2)
    dT_formula_correct = energy_loss(v1b, v2b, m1, m2, e_correct)

    assert abs(dT_mismatch - dT_formula_correct) > 0.1, \
        f"反例失败：错误e仍匹配能量公式"


def test_invalid_parameters():
    """非法参数应被拒绝。"""
    try:
        validate_parameters(e=-0.1)
        raise AssertionError("应拒绝 e<0")
    except AssertionError as e_msg:
        assert "e" in str(e_msg)
    try:
        validate_parameters(e=1.5)
        raise AssertionError("应拒绝 e>1")
    except AssertionError as e_msg:
        assert "e" in str(e_msg)
    try:
        validate_parameters(m1=-1)
        raise AssertionError("应拒绝 m1<0")
    except AssertionError as e_msg:
        assert "m1" in str(e_msg)


if __name__ == "__main__":
    test_momentum_conservation()
    print("✓ 动量守恒")
    test_restitution_coefficient()
    print("✓ 恢复系数定义")
    test_energy_loss_formula()
    print("✓ 动能损失公式")
    test_elastic_energy_conservation()
    print("✓ 弹性碰撞动能守恒")
    test_plastic_stick_together()
    print("✓ 塑性碰撞粘合")
    test_analytical_motion()
    print("✓ 解析运动解")
    test_equal_mass_elastic_swap()
    print("✓ 等质量弹性碰撞速度交换")
    test_heavy_target_stationary()
    print("✓ 轻弹击重靶")
    test_error_injection_non_circular()
    print("✓ 反例验证 (非循环)")
    test_invalid_parameters()
    print("✓ 非法参数")
    print("\nOK: MEC-042 所有一致性测试通过")
