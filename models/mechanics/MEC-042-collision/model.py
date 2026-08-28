"""MEC-042-collision — 模型定义（引擎无关）

碰撞（collision）：两个质点沿直线运动发生正碰撞。使用恢复系数 e 描述
碰撞后相对速度与碰撞前的关系，分析动量守恒和能量变化。

=== 物理系统 ===

  质点 1: 质量 m1, 碰前速度 v1_before
  质点 2: 质量 m2, 碰前速度 v2_before
  正碰撞（一维），恢复系数 e

  动量守恒: m1·v1 + m2·v2 = const
  恢复系数: e = -(v1_after - v2_after) / (v1_before - v2_before)

  解析解:
    v1_after = (m1·v1b + m2·v2b + m2·e·(v2b - v1b)) / (m1 + m2)
    v2_after = (m1·v1b + m2·v2b + m1·e·(v1b - v2b)) / (m1 + m2)

=== 恢复系数 ===

  e = 1: 完全弹性碰撞（动能守恒）
  e = 0: 完全塑性碰撞（粘在一起）
  0 < e < 1: 非弹性碰撞（动能损失）

=== 能量 ===

  碰前: T_before = ½(m1·v1b² + m2·v2b²)
  碰后: T_after  = ½(m1·v1a² + m2·v2a²)
  损失: ΔT = T_after - T_before = -½·(1-e²)·m_red·(v1b-v2b)²
  其中 m_red = m1·m2/(m1+m2) 为约化质量

=== 与已有 MEC 模型的关系 ===

  e=1 → 完全弹性碰撞，动能守恒
  e=0 → 完全塑性，退化为一体运动（MEC-002 受力质点）
  碰撞前/后自由运动 → MEC-001（自由质点）
  MEC-040 接触力是连续接触，此处是瞬时碰撞

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(m1=1.0, m2=1.0, e=0.8):
    """验证物理参数合法性。"""
    assert m1 > 0, f"质量 m1 必须为正，当前 m1={m1}"
    assert m2 > 0, f"质量 m2 必须为正，当前 m2={m2}"
    assert 0 <= e <= 1, f"恢复系数 e 必须在 [0,1]，当前 e={e}"


def reduced_mass(m1, m2):
    """约化质量 m_red = m1·m2/(m1+m2)。"""
    return m1 * m2 / (m1 + m2)


def solve_collision(v1_before, v2_before, m1=1.0, m2=1.0, e=0.8):
    """求解一维正碰撞后的速度。

    返回 (v1_after, v2_after)。

    推导：
      动量守恒: m1·v1b + m2·v2b = m1·v1a + m2·v2a
      恢复系数: e = -(v1a - v2a)/(v1b - v2b)

      联立解得:
        v1a = (m1·v1b + m2·v2b + m2·e·(v2b - v1b)) / (m1 + m2)
        v2a = (m1·v1b + m2·v2b + m1·e·(v1b - v2b)) / (m1 + m2)
    """
    total_m = m1 + m2
    p_total = m1 * v1_before + m2 * v2_before  # 总动量
    dv = v1_before - v2_before  # 相对速度

    v1_after = (p_total + m2 * e * (v2_before - v1_before)) / total_m
    v2_after = (p_total + m1 * e * (v1_before - v2_before)) / total_m

    return v1_after, v2_after


def kinetic_energy_before(v1, v2, m1=1.0, m2=1.0):
    """碰撞前总动能。"""
    return 0.5 * (m1 * v1**2 + m2 * v2**2)


def kinetic_energy_after(v1, v2, m1=1.0, m2=1.0):
    """碰撞后总动能（与 before 相同形式，只是速度不同）。"""
    return 0.5 * (m1 * v1**2 + m2 * v2**2)


def energy_loss(v1_before, v2_before, m1=1.0, m2=1.0, e=0.8):
    """碰撞动能损失 ΔT = -½(1-e²)·m_red·(v1b-v2b)²。"""
    m_red = reduced_mass(m1, m2)
    dv = v1_before - v2_before
    return -0.5 * (1 - e**2) * m_red * dv**2


def momentum_before(v1, v2, m1=1.0, m2=1.0):
    """碰撞前总动量。"""
    return m1 * v1 + m2 * v2


def momentum_after(v1, v2, m1=1.0, m2=1.0):
    """碰撞后总动量。"""
    return m1 * v1 + m2 * v2


def dynamics(t, state, m1=1.0, m2=1.0, e=0.8, t_collision=1.0):
    """两个自由质点 + 在 t_collision 时刻发生碰撞。

    state = [x1, v1, x2, v2]
    碰撞前: 自由运动（ẋ=v, v̇=0）
    碰撞时: 速度跳变（由 solve_collision 计算）
    碰撞后: 自由运动
    """
    x1, v1, x2, v2 = state

    if abs(t - t_collision) < 1e-10:
        # 碰撞瞬间：速度跳变
        v1_new, v2_new = solve_collision(v1, v2, m1, m2, e)
        return np.array([v1_new, 0.0, v2_new, 0.0])

    # 自由运动
    return np.array([v1, 0.0, v2, 0.0])


def analytical_motion(t, v1_before, v2_before, m1=1.0, m2=1.0, e=0.8,
                       t_collision=1.0, x1_0=0.0, x2_0=2.0):
    """完整运动解析解（碰撞前+碰撞后自由运动）。

    返回 (x1, v1, x2, v2)。
    """
    t = np.asarray(t, dtype=float)

    # 碰撞前
    x1_pre = x1_0 + v1_before * t
    x2_pre = x2_0 + v2_before * t
    v1_pre = np.full_like(t, v1_before)
    v2_pre = np.full_like(t, v2_before)

    # 碰撞后
    v1_after, v2_after = solve_collision(v1_before, v2_before, m1, m2, e)

    # 碰撞时刻位置
    x1_col = x1_0 + v1_before * t_collision
    x2_col = x2_0 + v2_before * t_collision

    x1_post = x1_col + v1_after * (t - t_collision)
    x2_post = x2_col + v2_after * (t - t_collision)
    v1_post = np.full_like(t, v1_after)
    v2_post = np.full_like(t, v2_after)

    # 按时间段选择
    mask_pre = t < t_collision
    mask_post = ~mask_pre

    x1 = np.where(mask_pre, x1_pre, x1_post)
    x2 = np.where(mask_pre, x2_pre, x2_post)
    v1 = np.where(mask_pre, v1_pre, v1_post)
    v2 = np.where(mask_pre, v2_pre, v2_post)

    return x1, v1, x2, v2
