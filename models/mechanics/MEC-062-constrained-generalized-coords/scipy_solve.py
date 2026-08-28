"""MEC-062 —— 用 SciPy 求解约束系统。

涵盖：
1. 单摆（完整约束消去法）
2. 阿特伍德机（完整约束 + 乘子法求张力）
3. 斜面纯滚动（完整约束 + 拉格朗日重解）
4. 约束力计算与验证

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (
    validate_parameters,
    pendulum_lagrangian,
    pendulum_dynamics,
    pendulum_small_angle_frequency,
    pendulum_energy,
    atwood_lagrangian,
    atwood_dynamics,
    atwood_acceleration,
    atwood_tension,
    rolling_incline_lagrangian,
    rolling_incline_dynamics,
    rolling_incline_acceleration,
    rolling_incline_energy,
    lagrange_multiplier_pendulum,
    constraint_force_pendulum,
    rolling_constraint_force,
    static_friction_required,
    max_incline_angle_for_pure_rolling,
    verify_pendulum_constraint,
    verify_rolling_constraint,
)


if __name__ == "__main__":
    print("=" * 60)
    print("MEC-062 Constrained Generalized Coordinates")
    print("=" * 60)

    # --- 1. 单摆 ---
    print(f"\n{'='*60}")
    print("1. 单摆（完整约束 x²+y²=l² → 广义坐标 θ）")
    print(f"{'='*60}")
    m, g, l = 1.0, 9.81, 1.0
    omega_small = pendulum_small_angle_frequency(g, l)
    print(f"  小角度频率: ω = √(g/l) = {omega_small:.4f} rad/s")

    sol = solve_ivp(pendulum_dynamics, (0, 10.0), [0.5, 0.0],
                    args=(m, g, l), t_eval=np.linspace(0, 10.0, 1001),
                    rtol=1e-10, atol=1e-12)
    energies = np.array([pendulum_energy(sol.y[:, i], m, g, l)
                         for i in range(sol.y.shape[1])])
    print(f"  能量守恒: {np.max(np.abs(energies - energies[0])):.3e}")

    # 约束力
    T_max = max(lagrange_multiplier_pendulum(sol.y[:, i], m, g, l)
                for i in range(sol.y.shape[1]))
    T_min = min(lagrange_multiplier_pendulum(sol.y[:, i], m, g, l)
                for i in range(sol.y.shape[1]))
    print(f"  绳张力范围: [{T_min:.4f}, {T_max:.4f}] N")
    print(f"  静止时张力: {m*g:.4f} N")

    # --- 2. 阿特伍德机 ---
    print(f"\n{'='*60}")
    print("2. 阿特伍德机（完整约束 x1+x2=l）")
    print(f"{'='*60}")
    m1, m2 = 2.0, 1.0
    a = atwood_acceleration(m1, m2, g)
    T = atwood_tension(m1, m2, g)
    print(f"  m1={m1}, m2={m2}, g={g}")
    print(f"  加速度: a = (m1-m2)g/(m1+m2) = {a:.4f} m/s²")
    print(f"  绳张力: T = 2m1m2g/(m1+m2) = {T:.4f} N")

    sol = solve_ivp(atwood_dynamics, (0, 5.0), [0.0, 0.0],
                    args=(m1, m2, g, 2.0), t_eval=np.linspace(0, 5.0, 501),
                    rtol=1e-10, atol=1e-12)
    x_ana = 0.5 * a * sol.t**2
    v_ana = a * sol.t
    print(f"  x 误差: {np.max(np.abs(sol.y[0] - x_ana)):.3e}")
    print(f"  v 误差: {np.max(np.abs(sol.y[1] - v_ana)):.3e}")

    # --- 3. 斜面纯滚动 ---
    print(f"\n{'='*60}")
    print("3. 斜面纯滚动（完整约束 x=Rφ → 广义坐标 x）")
    print(f"{'='*60}")
    m, g, R, k = 1.0, 9.81, 0.5, 0.4
    theta = 30.0
    a_roll = rolling_incline_acceleration(g, k, theta)
    a_slide = g * np.sin(np.radians(theta))  # 无滚动
    print(f"  m={m}, R={R}, k={k} (球), θ={theta}°")
    print(f"  纯滚动加速度: a = g·sinθ/(1+k) = {a_roll:.4f} m/s²")
    print(f"  纯滑动加速度: a = g·sinθ = {a_slide:.4f} m/s²")
    print(f"  比值 a_roll/a_slide = {a_roll/a_slide:.4f} = 1/(1+k) = {1/(1+k):.4f}")

    sol = solve_ivp(rolling_incline_dynamics, (0, 5.0), [0.0, 0.0],
                    args=(m, g, R, k, theta),
                    t_eval=np.linspace(0, 5.0, 501),
                    rtol=1e-10, atol=1e-12)
    x_ana = 0.5 * a_roll * sol.t**2
    v_ana = a_roll * sol.t
    print(f"  x 误差: {np.max(np.abs(sol.y[0] - x_ana)):.3e}")

    energies = np.array([rolling_incline_energy(sol.y[:, i], m, g, R, k, theta)
                         for i in range(sol.y.shape[1])])
    print(f"  能量守恒: {np.max(np.abs(energies - energies[0])):.3e}")

    # --- 4. 约束力 ---
    print(f"\n{'='*60}")
    print("4. 约束力（拉格朗日乘子）")
    print(f"{'='*60}")
    f_friction = static_friction_required(m, g, k, theta)
    N = m * g * np.cos(np.radians(theta))
    mu_s_needed = f_friction / N
    theta_max = max_incline_angle_for_pure_rolling(mu_s=0.5, k=k)
    print(f"  纯滚动静摩擦力: f_s = {f_friction:.4f} N")
    print(f"  法向力: N = {N:.4f} N")
    print(f"  所需摩擦系数: μ_s ≥ {mu_s_needed:.4f}")
    print(f"  μ_s=0.5 时最大角度: θ_max = {theta_max:.1f}°")

    # 约束验证
    print(f"\n  单摆约束验证: {verify_pendulum_constraint([0.5, 0.0], l)}")
    ok_x, ok_v = verify_rolling_constraint([0.3, 1.0], R)
    print(f"  纯滚动约束验证: x={ok_x}, v={ok_v}")

    print("\n=== 求解完成 ===")
