"""MEC-080-multibody-dynamics — 模型定义（引擎无关）

多体动力学（multibody dynamics）：综合刚体动力学 + 约束 + 分析力学。
构建 N 连杆平面链的通用动力学框架。

=== 物理系统 ===

  N 刚体通过铰链连接成平面链。
  每个刚体 i：质量 m_i、杆长 l_i、转动惯量 I_i（关于质心）。
  广义坐标 θ_i：各杆相对竖直方向的角度。

  综合 MEC-020 刚体 + MEC-030 约束 + MEC-060 拉格朗日 + MEC-062 广义坐标。

=== 运动方程（标准操作臂形式）===

  M(θ) θ̈ + C(θ, θ̇) θ̇ + G(θ) = 0

  M(θ)    — N×N 质量矩阵（惯性项）
  C(θ,θ̇)  — N×N 科氏/离心力矩阵
  G(θ)    — N×1 重力向量

=== N=1 退化到单摆，N=2 退化到双摆 (MEC-013) ===

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np


def validate_parameters(masses, lengths, inertias, g=9.81):
    """验证物理参数合法性。"""
    N = len(masses)
    assert N >= 1, "至少需要 1 个刚体"
    assert len(lengths) == N, f"长度数组维度({len(lengths)})≠N({N})"
    assert len(inertias) == N, f"惯量数组维度({len(inertias)})≠N({N})"
    for i in range(N):
        assert masses[i] > 0, f"质量 m[{i}] 必须为正"
        assert lengths[i] > 0, f"长度 l[{i}] 必须为正"
        assert inertias[i] >= 0, f"惯量 I[{i}] 必须非负"
    assert g > 0, f"重力加速度 g 必须为正"


# ============================================================
# 质心位置与速度
# ============================================================

def center_of_mass_positions(theta, lengths):
    """各杆质心位置。θ 从竖直向下测量。"""
    N = len(theta)
    x = np.zeros(N)
    y = np.zeros(N)
    cx, cy = 0.0, 0.0
    for i in range(N):
        x[i] = cx + 0.5 * lengths[i] * np.sin(theta[i])
        y[i] = cy - 0.5 * lengths[i] * np.cos(theta[i])
        cx += lengths[i] * np.sin(theta[i])
        cy -= lengths[i] * np.cos(theta[i])
    return x, y


def center_of_mass_velocities(theta, theta_dot, lengths):
    """各杆质心速度。"""
    N = len(theta)
    vx = np.zeros(N)
    vy = np.zeros(N)
    cvx, cvy = 0.0, 0.0
    for i in range(N):
        vx[i] = cvx + 0.5 * lengths[i] * theta_dot[i] * np.cos(theta[i])
        vy[i] = cvy + 0.5 * lengths[i] * theta_dot[i] * np.sin(theta[i])
        cvx += lengths[i] * theta_dot[i] * np.cos(theta[i])
        cvy += lengths[i] * theta_dot[i] * np.sin(theta[i])
    return vx, vy


# ============================================================
# 能量
# ============================================================

def kinetic_energy(theta, theta_dot, masses, lengths, inertias):
    """T = Σ [½ m_i v_ci² + ½ I_i ω_i²。"""
    vx, vy = center_of_mass_velocities(theta, theta_dot, lengths)
    T = 0.0
    for i in range(len(theta)):
        T += 0.5 * masses[i] * (vx[i]**2 + vy[i]**2)
        T += 0.5 * inertias[i] * theta_dot[i]**2
    return T


def potential_energy(theta, masses, lengths, g):
    """V = Σ m_i g y_ci。"""
    _, y = center_of_mass_positions(theta, lengths)
    V = 0.0
    for i in range(len(theta)):
        V += masses[i] * g * y[i]
    return V


def total_energy(theta, theta_dot, masses, lengths, inertias, g):
    """E = T + V。"""
    return (kinetic_energy(theta, theta_dot, masses, lengths, inertias) +
            potential_energy(theta, masses, lengths, g))


# ============================================================
# 质量矩阵 M(θ)（解析公式）
# ============================================================

def mass_matrix(theta, masses, lengths, inertias):
    """N×N 质量矩阵 M(θ)。

    M_ij = Σ_{k=max(i,j)}^{N-1} [m_k · (∂v_ck/∂θ̇_i)·(∂v_ck/∂θ̇_j)] + δ_ij·I_i

    对于质心在杆中点的均匀链，解析推导给出：
    M_ij = Σ_{k=max(i,j)}^{N-1} m_k · r_ki · r_kj · cos(θ_i - θ_j)
    其中 r_ki = l_i (k > i) 或 l_i/2 (k == i)

    加上对角惯量 I_i。
    """
    N = len(theta)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            k_min = max(i, j)
            for k in range(k_min, N):
                # 杆 k 质心位置对 θ̇_i 的偏导系数
                r_ki = lengths[i] if k > i else 0.5 * lengths[i]
                r_kj = lengths[j] if k > j else 0.5 * lengths[j]
                M[i, j] += masses[k] * r_ki * r_kj * np.cos(theta[i] - theta[j])
            if i == j:
                M[i, j] += inertias[i]
    return M


# ============================================================
# 科氏/离心矩阵和重力向量
# ============================================================

def coriolis_vector(theta, theta_dot, masses, lengths, inertias):
    """科氏/离心力向量 C(θ,θ̇)θ̇。

    使用 Christoffel 符号：
    C_i = Σ_jk Γ_ijk θ̇_j θ̇_k
    Γ_ijk = ½(∂M_ij/∂θ_k + ∂M_ik/∂θ_j - ∂M_jk/∂θ_i)
    """
    N = len(theta)
    h = 1e-6

    # 数值计算 ∂M/∂θ_k
    dM = np.zeros((N, N, N))
    for k in range(N):
        d_th = np.zeros(N)
        d_th[k] = h
        M_p = mass_matrix(theta + d_th, masses, lengths, inertias)
        M_m = mass_matrix(theta - d_th, masses, lengths, inertias)
        dM[:, :, k] = (M_p - M_m) / (2 * h)

    # Christoffel 符号和科氏力
    C = np.zeros(N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                gamma_ijk = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
                C[i] += gamma_ijk * theta_dot[j] * theta_dot[k]
    return C


def gravity_vector(theta, masses, lengths, g):
    """重力向量 G_i = ∂V/∂θ_i。"""
    N = len(theta)
    h = 1e-6
    V0 = potential_energy(theta, masses, lengths, g)
    G = np.zeros(N)
    for i in range(N):
        d_th = np.zeros(N)
        d_th[i] = h
        V_p = potential_energy(theta + d_th, masses, lengths, g)
        G[i] = (V_p - V0) / h
    return G


# ============================================================
# 动力学方程
# ============================================================

def dynamics(t, state, masses, lengths, inertias, g=9.81):
    """N 连杆平面链动力学。

    state = [θ1..θN, θ̇1..θ̇N]
    M θ̈ + C θ̇ + G = 0  →  θ̈ = M⁻¹ (-C - G)
    """
    N = len(masses)
    theta = state[:N]
    theta_dot = state[N:]

    M = mass_matrix(theta, masses, lengths, inertias)
    C = coriolis_vector(theta, theta_dot, masses, lengths, inertias)
    G = gravity_vector(theta, masses, lengths, g)

    b = -C - G
    theta_ddot = np.linalg.solve(M, b)
    return np.concatenate([theta_dot, theta_ddot])


def lagrangian(theta, theta_dot, masses, lengths, inertias, g):
    """L = T - V。"""
    return (kinetic_energy(theta, theta_dot, masses, lengths, inertias) -
            potential_energy(theta, masses, lengths, g))
