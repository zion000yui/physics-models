"""MEC-061-hamiltonian-formulation — 模型定义（引擎无关）

哈密顿力学：通过 Legendre 变换从拉格朗日力学导出，
在相空间 (q, p) 中描述系统动力学。

=== 核心原理 ===

  共轭动量：p_i = ∂L/∂q̇_i

  Legendre 变换：H(q, p) = Σ p_i q̇_i - L(q, q̇)
  其中 q̇ 需用 p 表示（通过 p = ∂L/∂q̇ 反解）

  哈密顿正则方程：
    q̇_i = ∂H/∂p_i    （位置方程）
    ṗ_i = -∂H/∂q_i   （动量方程）

  对保守系统：H = T + V = E（总能量守恒）

=== 重新求解的已有模型 ===

  1) MEC-001 自由质点
     L = ½mẋ²,  p = mẋ,  ẋ = p/m
     H = p·(p/m) - ½m(p/m)² = p²/(2m)
     q̇ = ∂H/∂p = p/m = v
     ṗ = -∂H/∂q = 0（动量守恒）

  2) MEC-002 受力质点（保守恒力 F）
     L = ½mẋ² - Fx,  p = mẋ
     H = p²/(2m) + Fx
     q̇ = p/m
     ṗ = -F（动量线性变化）

  3) MEC-010 弹簧振子
     L = ½mẋ² - ½kx²,  p = mẋ
     H = p²/(2m) + ½kx²
     q̇ = p/m
     ṗ = -kx
     H = E = ½mv² + ½kx² = p²/(2m) + ½kx²（守恒）

  4) MEC-006 胡克力 2D
     L = ½m(ẋ²+ẏ²) - ½k(x²+y²)
     p_x = mẋ,  p_y = mẏ
     H = (p_x² + p_y²)/(2m) + ½k(x² + y²)
     ẋ = p_x/m,  ẏ = p_y/m
     ṗ_x = -kx,  ṗ_y = -ky

  5) MEC-011 阻尼振子（非保守，H 不守恒）
     H = p²/(2m) + ½kx²
     q̇ = p/m
     ṗ = -kx - c(p/m)  ← 阻尼力通过 -∂H/∂q + 非保守力
     dH/dt = -c·(p/m)² = -cv² ≤ 0（能量耗散）

=== 哈密顿力学的特点 ===

  - 相空间描述：(q, p) 而非 (q, q̇)
  - 一阶 ODE 系统（而非拉格朗日的二阶）
  - 辛结构：{q, p} = 1（泊松括号）
  - 守恒律自然嵌入：H 守恒 ↔ 时间平移对称性
  - Liouville 定理：相空间体积守恒（哈密顿流不可压缩）

=== 与已有 MEC 模型的关系 ===

  MEC-060 拉格朗日 → MEC-061 哈密顿（Legendre 变换）
  MEC-061 → MEC-062 约束系统的哈密顿处理
  MEC-061 → MEC-090 非线性力学（KAM 定理、混沌，需相空间描述）

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(m=1.0, k=1.0, g=9.81, c=0.0, F=0.0):
    """验证物理参数合法性。"""
    assert m > 0, f"质量 m 必须为正，当前 m={m}"
    assert k > 0, f"弹簧常数 k 必须为正，当前 k={k}"
    assert g > 0, f"重力加速度 g 必须为正，当前 g={g}"
    assert c >= 0, f"阻尼系数 c 必须非负，当前 c={c}"


# ============================================================
# 1. 自由质点 (MEC-001)
# ============================================================

def free_particle_hamiltonian(state, m):
    """H = p²/(2m)。

    state = [q, p]
    """
    _, p = state
    return p**2 / (2.0 * m)


def free_particle_canonical(t, state, m=1.0):
    """正则方程：q̇ = p/m, ṗ = 0。"""
    _, p = state
    return np.array([p / m, 0.0])


# ============================================================
# 2. 受力质点 (MEC-002)
# ============================================================

def forced_particle_hamiltonian(state, m, F):
    """H = p²/(2m) + Fx。"""
    x, p = state
    return p**2 / (2.0 * m) + F * x


def forced_particle_canonical(t, state, m=1.0, F=1.0):
    """正则方程：q̇ = p/m, ṗ = -F。"""
    _, p = state
    return np.array([p / m, -F])


# ============================================================
# 3. 弹簧振子 (MEC-010)
# ============================================================

def spring_hamiltonian(state, m, k):
    """H = p²/(2m) + ½kx²。"""
    x, p = state
    return p**2 / (2.0 * m) + 0.5 * k * x**2


def spring_canonical(t, state, m=1.0, k=1.0):
    """正则方程：q̇ = p/m, ṗ = -kx。"""
    x, p = state
    return np.array([p / m, -k * x])


# ============================================================
# 4. 胡克力 2D (MEC-006)
# ============================================================

def hooke_hamiltonian_2d(state, m, k):
    """H = (p_x² + p_y²)/(2m) + ½k(x² + y²)。

    state = [x, y, px, py]
    """
    x, y, px, py = state
    return (px**2 + py**2) / (2.0 * m) + 0.5 * k * (x**2 + y**2)


def hooke_canonical_2d(t, state, m=1.0, k=1.0):
    """正则方程：ẋ = px/m, ẏ = py/m, ṗx = -kx, ṗy = -ky。"""
    x, y, px, py = state
    return np.array([px / m, py / m, -k * x, -k * y])


# ============================================================
# 5. 阻尼振子 (MEC-011) — 非保守
# ============================================================

def damped_spring_hamiltonian(state, m, k):
    """H = p²/(2m) + ½kx²（H 不守恒，dH/dt < 0）。"""
    x, p = state
    return p**2 / (2.0 * m) + 0.5 * k * x**2


def damped_spring_canonical(t, state, m=1.0, k=1.0, c=0.1):
    """正则方程 + 非保守力：q̇ = p/m, ṗ = -kx - c·p/m。"""
    x, p = state
    return np.array([p / m, -k * x - c * p / m])


# ============================================================
# 辅助工具
# ============================================================

def legendre_transform(lagrangian_func, state_lagrangian, m, k=None):
    """Legendre 变换：H = p·q̇ - L。

    state_lagrangian: [q, q̇]（拉格朗日坐标）
    返回 (state_hamiltonian, H) 其中 state_hamiltonian = [q, p]
    """
    q, qdot = state_lagrangian
    p = m * qdot  # 共轭动量
    L = lagrangian_func(state_lagrangian, m, k) if k else lagrangian_func(state_lagrangian, m)
    H = p * qdot - L
    return np.array([q, p]), H


def poisson_bracket(f_func, g_func, state, m, k=None):
    """泊松括号 {f, g} = ∂f/∂q · ∂g/∂p - ∂f/∂p · ∂g/∂q。

    数值计算（有限差分）。
    """
    q, p = state
    h = 1e-6

    # 数值偏导
    df_dq = (f_func([q + h, p], m, k) - f_func([q - h, p], m, k)) / (2 * h) if k else \
            (f_func([q + h, p], m) - f_func([q - h, p], m)) / (2 * h)
    df_dp = (f_func([q, p + h], m, k) - f_func([q, p - h], m, k)) / (2 * h) if k else \
            (f_func([q, p + h], m) - f_func([q, p - h], m)) / (2 * h)
    dg_dq = (g_func([q + h, p], m, k) - g_func([q - h, p], m, k)) / (2 * h) if k else \
            (g_func([q + h, p], m) - g_func([q - h, p], m)) / (2 * h)
    dg_dp = (g_func([q, p + h], m, k) - g_func([q, p - h], m, k)) / (2 * h) if k else \
            (g_func([q, p + h], m) - g_func([q, p - h], m)) / (2 * h)

    return df_dq * dg_dp - df_dp * dg_dq


def canonical_commutator(state, m):
    """基本泊松括号 {q, p} = 1。"""
    q_func = lambda s, m: s[0]
    p_func = lambda s, m: s[1]
    return poisson_bracket(q_func, p_func, state, m)


# ============================================================
# 相空间面积（Liouville 定理验证）
# ============================================================

def phase_space_area(states):
    """计算相空间中一组点围成的面积（用于 Liouville 定理验证）。

    states: (N, 2) 数组，每行 (q, p)
    使用 shoelace 公式。
    """
    q = states[:, 0]
    p = states[:, 1]
    N = len(q)
    # Shoelace formula
    area = 0.0
    for i in range(N):
        j = (i + 1) % N
        area += q[i] * p[j] - q[j] * p[i]
    return 0.5 * abs(area)
