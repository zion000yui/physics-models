"""MEC-031 —— 用 SciPy 数值求解曲柄滑块机构。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, slider_position, rod_angle,
                   slider_velocity_ratio, rod_angular_velocity_ratio,
                   velocity_analysis, acceleration_analysis,
                   equivalent_inertia, mechanical_energy,
                   toggle_positions, validate_parameters)


def simulate(r=0.3, l=1.0, m_crank=1.0, m_rod=1.0, m_sl=1.0,
             theta0=0.5, omega0=1.0, tau=0.0, g=0.0,
             t_end=10.0, n_points=501):
    """数值积分曲柄滑块动力学。"""
    validate_parameters(r=r, l=l, m_crank=m_crank, m_rod=m_rod, m_sl=m_sl, g=g, tau=tau)

    r_cm, l_cm = r / 2, l / 2
    I_crank = m_crank * r**2 / 12
    I_rod = m_rod * l**2 / 12

    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=[theta0, omega0],
        t_eval=t_eval,
        args=(r, l, m_crank, m_rod, m_sl, r_cm, l_cm, I_crank, I_rod, g, tau),
        rtol=1e-10, atol=1e-12,
    )
    return sol


if __name__ == "__main__":
    r, l = 0.3, 1.0
    m_crank, m_rod, m_sl = 1.0, 1.0, 1.0
    theta0, omega0 = 0.5, 1.0

    # 均匀杆参数
    r_cm, l_cm = r / 2, l / 2
    I_crank = m_crank * r**2 / 12
    I_rod = m_rod * l**2 / 12

    # 极限位置
    x_tdc, x_bdc, theta_tdc, theta_bdc = toggle_positions(r, l)
    print(f"上止点 (TDC): θ=0°, x={x_tdc:.4f} m")
    print(f"下止点 (BDC): θ=180°, x={x_bdc:.4f} m")
    print(f"行程        : {x_tdc - x_bdc:.4f} m (= 2r = {2*r:.4f})")

    # 初始运动学
    x0 = slider_position(theta0, r, l)
    phi0 = rod_angle(theta0, r, l)
    vx0, phidot0 = velocity_analysis(theta0, omega0, r, l)
    print(f"\n初始 θ={np.degrees(theta0):.2f}°")
    print(f"  x    = {x0:.6f} m")
    print(f"  φ    = {np.degrees(phi0):.2f}°")
    print(f"  ẋ    = {vx0:.6f} m/s")
    print(f"  φ̇   = {phidot0:.6f} rad/s")

    I_eff_0 = equivalent_inertia(theta0, r, l, m_crank, m_rod, m_sl,
                                 r_cm, l_cm, I_crank, I_rod)
    print(f"  I_eff = {I_eff_0:.6f} kg·m²")
    print(f"  E_0   = {0.5 * I_eff_0 * omega0**2:.6f} J")

    # 数值积分
    sol = simulate(r, l, m_crank, m_rod, m_sl, theta0, omega0,
                   tau=0.0, g=0.0, t_end=10.0, n_points=501)

    theta = sol.y[0]
    omega = sol.y[1]

    # 运动学后处理
    x_arr = np.array([slider_position(t, r, l) for t in theta])
    phi_arr = np.array([rod_angle(t, r, l) for t in theta])

    # 能量
    E = np.array([mechanical_energy(
        [theta[i], omega[i]], r, l, m_crank, m_rod, m_sl,
        r_cm, l_cm, I_crank, I_rod, g=0.0) for i in range(len(theta))])

    # I_eff 范围
    I_eff_arr = np.array([equivalent_inertia(
        t, r, l, m_crank, m_rod, m_sl, r_cm, l_cm, I_crank, I_rod)
        for t in theta])

    print(f"\n=== 数值积分结果 ===")
    print(f"时间点数        : {len(sol.t)}")
    print(f"θ 范围          : [{np.min(theta):.4f}, {np.max(theta):.4f}] rad")
    print(f"ω 范围          : [{np.min(omega):.6f}, {np.max(omega):.6f}] rad/s")
    print(f"x 范围          : [{np.min(x_arr):.6f}, {np.max(x_arr):.6f}] m")
    print(f"φ 范围          : [{np.degrees(np.min(phi_arr)):.2f}°, {np.degrees(np.max(phi_arr)):.2f}°]")
    print(f"I_eff 范围      : [{np.min(I_eff_arr):.6f}, {np.max(I_eff_arr):.6f}] kg·m²")
    print(f"能量漂移        : {np.max(np.abs(E - E[0])):.3e} J")
