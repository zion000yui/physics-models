"""MEC-040-rigid-contact — 模型定义（引擎无关）

刚性接触（rigid contact）：质点在重力作用下与刚性地面发生法向接触。
采用惩罚法（spring-damper）建模法向接触力，实现接触/分离状态切换。

核心物理：互补约束 F·g = 0
  - 间隙 g > 0（未接触）：F = 0（自由飞行）
  - 间隙 g ≤ 0（穿透）：F = -(k_c·g + c_c·ġ) > 0（接触力阻止穿透）

=== 坐标系 ===

  y 轴向上，地面在 y = 0
  质点位置 y，速度 v = ẏ
  间隙 g = y - 0 = y（质点在地面上方时 g > 0）

  重力：F_gravity = -m·g
  接触力（当 g < 0 时）：F_contact = -k_c·g - c_c·ġ = -k_c·y - c_c·v（y<0时）

=== 运动方程 ===

  m·ÿ = F_gravity + F_contact

  自由飞行（y > 0）：ÿ = -g
  接触（y < 0）：ÿ = -g + (-k_c·y - c_c·v)/m

=== 惩罚参数 ===

  k_c：接触刚度（N/m），越大越接近刚性
  c_c：接触阻尼（N·s/m），控制回弹和能量耗散

  物理约束：k_c > 0, c_c ≥ 0
  数值约束：k_c 不能过大（否则 ODE 刚性过大）

=== 能量 ===

  动能 T = ½·m·v²
  重力势能 V_g = m·g·y
  接触弹性势能 V_c = ½·k_c·max(0, -y)²（仅在接触时存储）
  总机械能 E = T + V_g + V_c

  无阻尼时（c_c=0）：E 守恒
  有阻尼时（c_c>0）：E 单调递减（接触耗散）

=== 与已有 MEC 模型的关系 ===

  无接触时退化为 MEC-001（自由质点，匀速直线运动，无重力）
  或 MEC-002（受力质点，恒力=重力）
  重力下落无地面 → MEC-003（抛体运动 y 分量）
  MEC-024 纯滚动的静摩擦是约束力，此处接触力也是约束力

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(m=1.0, g=9.81, k_c=1e4, c_c=0.0):
    """验证物理参数合法性。"""
    assert m > 0, f"质量 m 必须为正，当前 m={m}"
    assert g > 0, f"重力加速度 g 必须为正，当前 g={g}"
    assert k_c > 0, f"接触刚度 k_c 必须为正，当前 k_c={k_c}"
    assert c_c >= 0, f"接触阻尼 c_c 必须非负，当前 c_c={c_c}"


def contact_force(y, v, m, k_c, c_c):
    """计算法向接触力。

    g = y（间隙，y > 0 表示在地面上方）
    当 y < 0（穿透）：F = -k_c·y - c_c·v（向上为正）
    当 y >= 0：F = 0

    返回接触力（正值=向上推）。
    """
    if y < 0:
        F = -k_c * y - c_c * v
        return max(F, 0.0)  # 接触力只能推，不能拉
    return 0.0


def gap(y):
    """间隙 g = y（y>0 在地面上方）。"""
    return y


def in_contact(y, tol=1e-10):
    """判断是否处于接触状态。"""
    return y < tol


def mechanical_energy(state, m, g, k_c):
    """计算总机械能 E = ½mv² + mgy + ½k_c·max(0,-y)²。"""
    y, v = state
    T = 0.5 * m * v**2
    V_g = m * g * y
    V_c = 0.5 * k_c * max(0.0, -y)**2
    return T + V_g + V_c


def dynamics(t, state, m=1.0, g=9.81, k_c=1e4, c_c=0.0):
    """返回状态时间导数 [dy/dt, dv/dt]。

    自由飞行：ÿ = -g
    接触：ÿ = -g + F_contact/m
    """
    y, v = state

    F_contact = contact_force(y, v, m, k_c, c_c)
    a = (-m * g + F_contact) / m  # = -g + F_contact/m

    return np.array([v, a])


def analytical_free_flight(t, y0, v0, g=9.81):
    """自由飞行段解析解（y > 0，无接触）。

    y(t) = y0 + v0·t - ½·g·t²
    v(t) = v0 - g·t
    """
    t = np.asarray(t, dtype=float)
    y = y0 + v0 * t - 0.5 * g * t**2
    v = v0 - g * t
    return y, v
