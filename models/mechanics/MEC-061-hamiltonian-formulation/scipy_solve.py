"""MEC-061 —— 用 SciPy 求解哈密顿正则方程。

涵盖：
1. 自由/受力质点（相空间动力学）
2. 弹簧振子（能量守恒 + 相空间椭圆）
3. 阻尼振子（能量耗散 + 相空间螺旋）
4. Liouville 定理验证
5. 泊松括号验证

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (
    validate_parameters,
    free_particle_hamiltonian,
    free_particle_canonical,
    forced_particle_hamiltonian,
    forced_particle_canonical,
    spring_hamiltonian,
    spring_canonical,
    hooke_hamiltonian_2d,
    hooke_canonical_2d,
    damped_spring_hamiltonian,
    damped_spring_canonical,
    legendre_transform,
    poisson_bracket,
    canonical_commutator,
    phase_space_area,
)


def solve_canonical(dynamics_fn, state0, params, t_end=10.0, n=1001):
    """通用正则方程数值积分。"""
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(dynamics_fn, (0, t_end), state0,
                    args=params, t_eval=t_eval,
                    rtol=1e-10, atol=1e-12)
    return sol


if __name__ == "__main__":
    print("=" * 60)
    print("MEC-061 Hamiltonian Formulation")
    print("=" * 60)

    # --- 1. 自由质点 ---
    print(f"\n{'='*60}")
    print("1. 自由质点 (H = p²/2m)")
    print(f"{'='*60}")
    m = 1.0
    state0 = [0.0, 2.0]  # q=0, p=2mv
    sol = solve_canonical(free_particle_canonical, state0, (m,), t_end=5.0)
    q_ana = 2.0 * sol.t  # q = (p/m)·t = 2t
    p_ana = 2.0 * np.ones_like(sol.t)  # p 守恒
    print(f"  q 误差: {np.max(np.abs(sol.y[0] - q_ana)):.3e}")
    print(f"  p 误差: {np.max(np.abs(sol.y[1] - p_ana)):.3e}")

    # --- 2. 弹簧振子 ---
    print(f"\n{'='*60}")
    print("2. 弹簧振子 (H = p²/2m + ½kx²)")
    print(f"{'='*60}")
    m, k = 1.0, 4.0
    omega = np.sqrt(k / m)
    state0 = [1.0, 0.0]  # x=1, p=0
    sol = solve_canonical(spring_canonical, state0, (m, k), t_end=10.0)

    q_ana = np.cos(omega * sol.t)
    p_ana = -m * omega * np.sin(omega * sol.t)
    print(f"  q 误差: {np.max(np.abs(sol.y[0] - q_ana)):.3e}")
    print(f"  p 误差: {np.max(np.abs(sol.y[1] - p_ana)):.3e}")

    # 能量守恒
    energies = np.array([spring_hamiltonian(sol.y[:, i], m, k)
                         for i in range(sol.y.shape[1])])
    print(f"  H(0) = {energies[0]:.6f}")
    print(f"  能量变化: {np.max(np.abs(energies - energies[0])):.3e}")

    # --- 3. 阻尼振子 ---
    print(f"\n{'='*60}")
    print("3. 阻尼振子 (H 耗散)")
    print(f"{'='*60}")
    m, k, c = 1.0, 4.0, 0.4
    state0 = [1.0, 0.0]
    sol = solve_canonical(damped_spring_canonical, state0, (m, k, c), t_end=20.0)

    energies = np.array([damped_spring_hamiltonian(sol.y[:, i], m, k)
                         for i in range(sol.y.shape[1])])
    print(f"  H(0) = {energies[0]:.6f}")
    print(f"  H(end) = {energies[-1]:.6f}")
    print(f"  能量变化: {energies[-1] - energies[0]:.6f} (应<0)")
    print(f"  单调递减: {np.all(np.diff(energies[::50]) <= 1e-8)}")

    # --- 4. Liouville 定理 ---
    print(f"\n{'='*60}")
    print("4. Liouville 定理（相空间面积守恒）")
    print(f"{'='*60}")
    m, k = 1.0, 1.0
    # 初始矩形/椭圆的边界点
    theta = np.linspace(0, 2 * np.pi, 100)
    q0_boundary = 0.5 * np.cos(theta)  # 椭圆边界
    p0_boundary = m * np.sqrt(k / m) * 0.5 * np.sin(theta)
    states0 = np.column_stack([q0_boundary, p0_boundary])

    area0 = phase_space_area(states0)
    # 积分每个边界点
    t_half = np.pi / (2 * np.sqrt(k / m))  # 1/4 周期
    states_final = np.zeros_like(states0)
    for i in range(len(states0)):
        sol = solve_ivp(spring_canonical, (0, t_half),
                        states0[i], args=(m, k),
                        rtol=1e-10, atol=1e-12)
        states_final[i] = sol.y[:, -1]
    area_final = phase_space_area(states_final)
    print(f"  初始面积: {area0:.6f}")
    print(f"  终止面积: {area_final:.6f}")
    print(f"  面积变化: {abs(area_final - area0) / area0:.3e}")

    # --- 5. 泊松括号 ---
    print(f"\n{'='*60}")
    print("5. 泊松括号 {q, p} = 1")
    print(f"{'='*60}")
    state = [0.5, 1.0]
    bracket = canonical_commutator(state, m=1.0)
    print(f"  {{q, p}} = {bracket:.6f} (应 = 1)")

    print("\n=== 求解完成 ===")
