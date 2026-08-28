"""MEC-041-coulomb-friction — 模型定义（引擎无关）

库仑摩擦（Coulomb friction）：质点在水平面上受外力作用，与地面之间产生
库仑摩擦力。静摩擦（|F_ext| ≤ μ_s·N 时物体静止）和动摩擦（|v| > 0 时
F_f = -μ_k·N·sign(v)）自动切换。

=== 物理系统 ===

  质点在水平面上运动（y=0，重力 g 向下，法向力 N = m·g）
  外力 F_ext 沿 x 方向施加

  摩擦力 F_f：
    - 静摩擦（v=0, |F_ext| ≤ μ_s·N）：F_f = -F_ext（摩擦力平衡外力，物体不动）
    - 动摩擦（v≠0）：F_f = -μ_k·N·sign(v)
    - 最大静摩擦（v=0, |F_ext| > μ_s·N）：物体开始滑动

=== 运动方程 ===

  m·ẍ = F_ext + F_f

  动摩擦：m·ẍ = F_ext - μ_k·N·sign(v)
  静摩擦：ẍ = 0（外力被摩擦力完全平衡）

=== 参数 ===

  m：质量，g：重力加速度
  μ_s：静摩擦系数（μ_s ≥ μ_k）
  μ_k：动摩擦系数（0 ≤ μ_k ≤ μ_s）
  F_ext：外力（可恒定或时变）

=== 能量 ===

  动能 T = ½·m·v²
  摩擦耗散功率 P_diss = μ_k·N·|v|（动摩擦时 > 0）
  无势能（水平面），E = T

  静止时：E = const = 0（v=0）
  动摩擦时：dE/dt = F_ext·v - μ_k·N·|v|（外力做功 - 摩擦耗散）

=== 与已有 MEC 模型的关系 ===

  无摩擦（μ_s=μ_k=0）→ 退化为 MEC-002（受力质点）
  无外力 + 有摩擦 → 自由滑动减速至停止（退化为 MEC-001 直到停止）
  MEC-024 纯滚动中静摩擦是约束力，此处静摩擦也是约束力
  MEC-040 接触力提供法向力 N，此处使用 N=m·g

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(m=1.0, g=9.81, mu_s=0.3, mu_k=0.25):
    """验证物理参数合法性。"""
    assert m > 0, f"质量 m 必须为正，当前 m={m}"
    assert g > 0, f"重力加速度 g 必须为正，当前 g={g}"
    assert mu_s >= 0, f"静摩擦系数 mu_s 必须非负，当前 mu_s={mu_s}"
    assert mu_k >= 0, f"动摩擦系数 mu_k 必须非负，当前 mu_k={mu_k}"
    assert mu_s >= mu_k, f"μ_s 必须 ≥ μ_k，当前 μ_s={mu_s} < μ_k={mu_k}"


def normal_force(m, g):
    """法向力 N = m·g（水平面）。"""
    return m * g


def friction_force(v, F_ext, m, g, mu_s, mu_k):
    """计算库仑摩擦力。

    静摩擦：v≈0 且 |F_ext| ≤ μ_s·N → F_f = -F_ext
    动摩擦：v≠0 → F_f = -μ_k·N·sign(v)
    超过最大静摩擦：v=0 且 |F_ext| > μ_s·N → 物体开始滑动
    """
    N = normal_force(m, g)
    v_threshold = 1e-8  # 速度阈值（区分静止/滑动）

    if abs(v) < v_threshold:
        # 静止或接近静止
        if abs(F_ext) <= mu_s * N:
            # 静摩擦：完全平衡外力
            return -F_ext
        else:
            # 超过最大静摩擦，开始滑动
            return -mu_k * N * np.sign(F_ext)
    else:
        # 动摩擦
        return -mu_k * N * np.sign(v)


def is_sliding(v, F_ext, m, g, mu_s, mu_k):
    """判断是否处于滑动状态。"""
    N = normal_force(m, g)
    v_threshold = 1e-8
    if abs(v) < v_threshold:
        return abs(F_ext) > mu_s * N
    return True


def mechanical_energy(state, m):
    """动能 E = ½·m·v²（水平面无势能）。"""
    _, v = state
    return 0.5 * m * v**2


def dynamics(t, state, m=1.0, g=9.81, mu_s=0.3, mu_k=0.25, F_ext=0.0):
    """返回状态时间导数 [dx/dt, dv/dt]。

    m·ẍ = F_ext + F_f
    """
    _, v = state
    F_f = friction_force(v, F_ext, m, g, mu_s, mu_k)
    a = (F_ext + F_f) / m
    return np.array([v, a])


def analytical_constant_force(t, v0, m, g, mu_s, mu_k, F_ext):
    """恒定外力下的解析解（分静摩擦/动摩擦两种情况）。

    返回 (x, v)。
    """
    t = np.asarray(t, dtype=float)
    N = normal_force(m, g)
    F_max_static = mu_s * N
    F_kinetic = mu_k * N

    if abs(F_ext) <= F_max_static:
        # 静摩擦：物体不动
        x = np.zeros_like(t)
        v = np.zeros_like(t)
        return x, v
    else:
        # 动摩擦：恒定加速度
        sign_F = np.sign(F_ext)
        a_net = (F_ext - mu_k * N * sign_F) / m
        v = v0 + a_net * t
        x = v0 * t + 0.5 * a_net * t**2
        return x, v
