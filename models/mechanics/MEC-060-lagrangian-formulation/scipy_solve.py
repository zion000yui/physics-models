"""MEC-060 —— 用 SciPy 求解拉格朗日力学公式化。

涵盖：
1. 自由质点 / 受力质点 / 弹簧振子（解析 vs 数值）
2. 阻尼振子（非保守力）
3. 双摆（非线性耦合，能量守恒）
4. 中心力胡克（2D 简谐振动）

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (
    validate_parameters,
    free_particle_lagrangian,
    free_particle_dynamics,
    forced_particle_lagrangian,
    forced_particle_dynamics,
    spring_lagrangian,
    spring_dynamics,
    spring_natural_frequency,
    damped_spring_lagrangian,
    damped_spring_dynamics,
    damping_ratio,
    hooke_lagrangian_2d,
    hooke_dynamics_2d,
    double_pendulum_lagrangian,
    double_pendulum_dynamics,
    double_pendulum_energy,
)


def solve_free_particle(m=1.0, v0=1.0, t_end=5.0, n=1001):
    """数值积分自由质点。"""
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(free_particle_dynamics, (0, t_end), [0.0, v0],
                    args=(m,), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol


def solve_forced_particle(m=1.0, F=1.0, v0=0.0, t_end=5.0, n=1001):
    """数值积分受力质点。"""
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(forced_particle_dynamics, (0, t_end), [0.0, v0],
                    args=(m, F), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol


def solve_spring(m=1.0, k=1.0, x0=1.0, v0=0.0, t_end=10.0, n=1001):
    """数值积分弹簧振子。"""
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(spring_dynamics, (0, t_end), [x0, v0],
                    args=(m, k), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol


def solve_damped_spring(m=1.0, k=1.0, c=0.1, x0=1.0, v0=0.0,
                         t_end=20.0, n=2001):
    """数值积分阻尼振子。"""
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(damped_spring_dynamics, (0, t_end), [x0, v0],
                    args=(m, k, c), t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol


def solve_double_pendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81,
                           th1_0=0.5, th2_0=0.3, t_end=10.0, n=5001):
    """数值积分双摆。"""
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(double_pendulum_dynamics, (0, t_end),
                    [th1_0, th2_0, 0.0, 0.0],
                    args=(m1, m2, l1, l2, g),
                    t_eval=t_eval, rtol=1e-10, atol=1e-12,
                    method='RK45')
    return sol


if __name__ == "__main__":
    print("=" * 60)
    print("MEC-060 Lagrangian Formulation")
    print("=" * 60)

    # --- 1. 自由质点 ---
    print(f"\n{'='*60}")
    print("1. 自由质点 (L = ½mẋ²)")
    print(f"{'='*60}")
    m, v0 = 1.0, 1.0
    sol = solve_free_particle(m, v0, t_end=5.0)
    # 解析: x = v0*t, v = v0
    x_ana = v0 * sol.t
    v_ana = v0 * np.ones_like(sol.t)
    err_x = np.max(np.abs(sol.y[0] - x_ana))
    err_v = np.max(np.abs(sol.y[1] - v_ana))
    print(f"  m={m}, v0={v0}")
    print(f"  x 误差: {err_x:.3e}")
    print(f"  v 误差: {err_v:.3e}")

    # --- 2. 受力质点 ---
    print(f"\n{'='*60}")
    print("2. 受力质点 (L = ½mẋ² - Fx)")
    print(f"{'='*60}")
    m, F = 1.0, 1.0
    sol = solve_forced_particle(m, F, v0=0.0, t_end=5.0)
    x_ana = 0.5 * F / m * sol.t**2
    v_ana = F / m * sol.t
    err_x = np.max(np.abs(sol.y[0] - x_ana))
    err_v = np.max(np.abs(sol.y[1] - v_ana))
    print(f"  m={m}, F={F}")
    print(f"  x 误差: {err_x:.3e}")
    print(f"  v 误差: {err_v:.3e}")

    # --- 3. 弹簧振子 ---
    print(f"\n{'='*60}")
    print("3. 弹簧振子 (L = ½mẋ² - ½kx²)")
    print(f"{'='*60}")
    m, k = 1.0, 4.0
    omega = spring_natural_frequency(m, k)
    sol = solve_spring(m, k, x0=1.0, v0=0.0, t_end=10.0)
    x_ana = 1.0 * np.cos(omega * sol.t)
    v_ana = -omega * np.sin(omega * sol.t)
    err_x = np.max(np.abs(sol.y[0] - x_ana))
    err_v = np.max(np.abs(sol.y[1] - v_ana))
    print(f"  m={m}, k={k}, ω={omega:.4f} rad/s")
    print(f"  x 误差: {err_x:.3e}")
    print(f"  v 误差: {err_v:.3e}")

    # --- 4. 阻尼振子 ---
    print(f"\n{'='*60}")
    print("4. 阻尼振子 (L + 非保守力 Q = -cẋ)")
    print(f"{'='*60}")
    m, k, c = 1.0, 4.0, 0.4
    zeta = damping_ratio(m, k, c)
    omega = spring_natural_frequency(m, k)
    sol = solve_damped_spring(m, k, c, x0=1.0, v0=0.0, t_end=20.0)
    # 欠阻尼解析解
    if zeta < 1:
        omega_d = omega * np.sqrt(1 - zeta**2)
        x_ana = np.exp(-zeta * omega * sol.t) * (
            np.cos(omega_d * sol.t)
            + zeta * omega / omega_d * np.sin(omega_d * sol.t))
        err_x = np.max(np.abs(sol.y[0] - x_ana))
        print(f"  m={m}, k={k}, c={c}, ζ={zeta:.3f} (欠阻尼)")
        print(f"  ω_d = {omega_d:.4f} rad/s")
        print(f"  x 误差: {err_x:.3e}")

    # --- 5. 双摆 ---
    print(f"\n{'='*60}")
    print("5. 双摆 (非线性耦合拉格朗日方程)")
    print(f"{'='*60}")
    m1, m2, l1, l2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    sol = solve_double_pendulum(m1, m2, l1, l2, g,
                                 th1_0=0.5, th2_0=0.3, t_end=10.0)

    # 能量守恒
    energies = np.array([double_pendulum_energy(sol.y[:, i], m1, m2, l1, l2, g)
                         for i in range(sol.y.shape[1])])
    dE = np.max(np.abs(energies - energies[0])) / abs(energies[0])
    print(f"  m1={m1}, m2={m2}, l1={l1}, l2={l2}")
    print(f"  θ₁(0)=0.5, θ₂(0)=0.3")
    print(f"  初始能量: {energies[0]:.6f} J")
    print(f"  最大能量变化: {dE:.3e} ({dE*100:.4f}%)")

    # --- 6. 拉格朗日量计算 ---
    print(f"\n{'='*60}")
    print("6. 拉格朗日量示例")
    print(f"{'='*60}")
    state_spring = [0.5, 0.0]
    L_val = spring_lagrangian(state_spring, m=1.0, k=4.0)
    print(f"  弹簧 L(x=0.5, v=0) = {L_val:.4f}")
    print(f"    (T=0, V=½kx²={0.5*4*0.5**2:.4f}, L=T-V={L_val:.4f})")

    state_double = [0.5, 0.3, 0.0, 0.0]
    L_val = double_pendulum_lagrangian(state_double, m1, m2, l1, l2, g)
    print(f"  双摆 L(θ₁=0.5, θ₂=0.3, ω=0) = {L_val:.6f}")

    print("\n=== 求解完成 ===")
