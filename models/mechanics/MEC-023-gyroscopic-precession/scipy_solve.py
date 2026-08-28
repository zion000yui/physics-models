"""MEC-023 —— 用 SciPy 数值求解陀螺慢进动运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, validate_parameters,
                   steady_state_precession, exact_steady_state_precession,
                   conjugate_momentum_phi, effective_energy)


def simulate(theta0=0.7854, theta_dot0=0.0, phi0=0.0, phi_dot0=None,
             m=1.0, l=0.5, I1=0.2, I3=0.1, omega_s=50.0, g=9.81,
             t_end=10.0, n_points=101):
    """数值积分陀螺轨迹，返回 (t, theta, theta_dot, phi, phi_dot)。"""
    if phi_dot0 is None:
        phi_dot0 = steady_state_precession(m, l, I1, I3, omega_s, g, theta0)
    initial_state = np.array([theta0, theta_dot0, phi0, phi_dot0],
                              dtype=float)
    validate_parameters(m=m, l=l, I1=I1, I3=I3, omega_s=omega_s, g=g)

    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(dynamics, (0.0, t_end), initial_state,
                    t_eval=t_eval, args=(m, l, I1, I3, omega_s, g),
                    rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


if __name__ == "__main__":
    m, l, I1, I3, omega_s, g = 1.0, 0.5, 0.2, 0.1, 50.0, 9.81
    theta_0 = np.pi / 4

    omega_p_approx = steady_state_precession(m, l, I1, I3, omega_s, g, theta_0)
    omega_slow, omega_fast = exact_steady_state_precession(
        m, l, I1, I3, omega_s, g, theta_0)

    print(f"参数: m={m}, l={l}, I1={I1}, I3={I3}, omega_s={omega_s}, g={g}")
    print(f"稳态倾斜角 θ₀ = {theta_0:.4f} rad ({np.degrees(theta_0):.1f}°)")
    print(f"慢进动近似 Ω_p = mgl/(I₃ω_s) = {omega_p_approx:.6f} rad/s")
    print(f"精确慢根 = {omega_slow:.6f} rad/s")
    print(f"精确快根 = {omega_fast:.6f} rad/s")
    print(f"近似误差 = {abs(omega_p_approx - omega_slow)/omega_slow:.3e}")
    print(f"Ω_p/ω_s = {omega_p_approx/omega_s:.3e} (应 << 1)")

    # 使用精确稳态初始条件
    t_end = 10.0
    t, theta_n, theta_dot_n, phi_n, phi_dot_n = simulate(
        theta0=theta_0, phi_dot0=omega_slow,
        m=m, l=l, I1=I1, I3=I3, omega_s=omega_s, g=g, t_end=t_end)

    # 守恒量检查
    p_phi = np.array([conjugate_momentum_phi(
        [theta_n[i], theta_dot_n[i], phi_n[i], phi_dot_n[i]],
        I1, I3, omega_s) for i in range(len(t))])
    E_eff = np.array([effective_energy(
        [theta_n[i], theta_dot_n[i], phi_n[i], phi_dot_n[i]],
        m, l, I1, I3, omega_s, g) for i in range(len(t))])

    print(f"时间点数       : {len(t)}")
    print(f"终止时间       : {t_end:.2f} s")
    print(f"θ 最大波动     : {np.max(np.abs(theta_n - theta_0)):.3e}")
    print(f"φ̇ 末点         : {phi_dot_n[-1]:.6f} (稳态 {omega_slow:.6f})")
    print(f"p_φ 标准差     : {np.std(p_phi):.3e}")
    print(f"E_eff 标准差   : {np.std(E_eff):.3e}")
