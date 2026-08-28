"""MEC-080 —— 用 SciPy 求解多体动力学。

涵盖：
1. N=1 单摆验证
2. N=2 双摆验证（与 MEC-060/MEC-013 对照）
3. N=3 三连杆系统
4. 能量守恒验证

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (
    validate_parameters,
    mass_matrix,
    coriolis_vector,
    gravity_vector,
    dynamics,
    kinetic_energy,
    potential_energy,
    total_energy,
    center_of_mass_positions,
    center_of_mass_velocities,
    lagrangian,
)


def solve_chain(masses, lengths, inertias, g, theta0, t_end=10.0, n=5001):
    """数值积分 N 连杆系统。"""
    N = len(masses)
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(dynamics, (0, t_end),
                    np.concatenate([theta0, np.zeros(N)]),
                    args=(masses, lengths, inertias, g),
                    t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol


if __name__ == "__main__":
    g = 9.81

    print("=" * 60)
    print("MEC-080 Multibody Dynamics")
    print("=" * 60)

    # --- 1. N=1 单摆 ---
    print(f"\n{'='*60}")
    print("1. N=1 单摆（退化验证）")
    print(f"{'='*60}")
    m1, l1, I1 = 1.0, 1.0, 0.0
    masses = [m1]; lengths = [l1]; inertias = [I1]
    validate_parameters(masses, lengths, inertias, g)

    omega = np.sqrt(g / l1)
    sol = solve_chain(masses, lengths, inertias, g, [0.1], t_end=5.0, n=501)
    x_ana = 0.1 * np.cos(omega * sol.t)
    err = np.max(np.abs(sol.y[0] - x_ana))
    print(f"  小角度频率: ω = √(g/l) = {omega:.4f}")
    print(f"  数值误差: {err:.3e}")

    energies = np.array([total_energy(sol.y[:1, i], sol.y[1:, i],
                                       masses, lengths, inertias, g)
                         for i in range(sol.y.shape[1])])
    print(f"  能量变化: {np.max(np.abs(energies - energies[0])):.3e}")

    # --- 2. N=2 双摆 ---
    print(f"\n{'='*60}")
    print("2. N=2 双摆（与 MEC-060 对照）")
    print(f"{'='*60}")
    m1, m2 = 1.0, 1.0
    l1, l2 = 1.0, 1.0
    I1, I2 = 0.0, 0.0  # 质点（无杆惯量）
    masses = [m1, m2]; lengths = [l1, l2]; inertias = [I1, I2]

    sol = solve_chain(masses, lengths, inertias, g, [0.5, 0.3], t_end=10.0, n=5001)
    energies = np.array([total_energy(sol.y[:2, i], sol.y[2:, i],
                                       masses, lengths, inertias, g)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / abs(energies[0])
    print(f"  θ₁(0)=0.5, θ₂(0)=0.3")
    print(f"  初始能量: {energies[0]:.6f} J")
    print(f"  最大能量变化: {dE:.3e} ({dE*100:.4f}%)")

    # --- 3. N=3 三连杆 ---
    print(f"\n{'='*60}")
    print("3. N=3 三连杆系统")
    print(f"{'='*60}")
    masses = [1.0, 1.0, 0.5]
    lengths = [1.0, 0.8, 0.6]
    inertias = [0.0, 0.0, 0.0]
    validate_parameters(masses, lengths, inertias, g)

    sol = solve_chain(masses, lengths, inertias, g, [0.3, 0.2, 0.1], t_end=10.0, n=5001)
    energies = np.array([total_energy(sol.y[:3, i], sol.y[3:, i],
                                       masses, lengths, inertias, g)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / abs(energies[0])
    print(f"  m={masses}, l={lengths}")
    print(f"  θ₀ = [0.3, 0.2, 0.1]")
    print(f"  初始能量: {energies[0]:.6f} J")
    print(f"  最大能量变化: {dE:.3e} ({dE*100:.4f}%)")

    # 质量矩阵检查
    M = mass_matrix([0.3, 0.2, 0.1], masses, lengths, inertias)
    print(f"\n  质量矩阵:\n{M}")

    # 正定性
    eigvals = np.linalg.eigvalsh(M)
    print(f"  特征值: {eigvals} (应全正)")

    # --- 4. 静态验证 ---
    print(f"\n{'='*60}")
    print("4. 静态验证（θ=0 时重力应为零）")
    print(f"{'='*60}")
    G = gravity_vector([0.0, 0.0, 0.0], masses, lengths, g)
    print(f"  G(θ=0) = {G} (应≈0)")

    G = gravity_vector([0.5, 0.3, 0.1], masses, lengths, g)
    print(f"  G(θ=[0.5,0.3,0.1]) = {G}")

    print("\n=== 求解完成 ===")
