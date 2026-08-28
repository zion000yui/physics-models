"""MEC-030 —— 用 SciPy 数值求解四连杆机构。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, position_analysis, velocity_analysis,
                   acceleration_analysis, equivalent_inertia,
                   mechanical_energy, grashof_criterion,
                   toggle_positions, validate_parameters)


def simulate(l1=2.0, l2=0.5, l3=1.5, l4=1.2,
             m2=1.0, m3=1.0, m4=1.0,
             theta2_0=0.5, omega2_0=1.0,
             tau=0.0, g=0.0, config='open',
             t_end=10.0, n_points=501):
    """数值积分四连杆动力学，返回完整结果。"""
    validate_parameters(l1, l2, l3, l4, m2, m3, m4, g=g, tau=tau)

    # 均匀杆默认值
    r2, r3, r4 = l2 / 2, l3 / 2, l4 / 2
    I2 = m2 * l2**2 / 12
    I3 = m3 * l3**2 / 12
    I4 = m4 * l4**2 / 12

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=[theta2_0, omega2_0],
        t_eval=t_eval,
        args=(l1, l2, l3, l4, m2, m3, m4, r2, r3, r4, I2, I3, I4, g, tau, config),
        rtol=1e-10,
        atol=1e-12,
    )

    return sol


if __name__ == "__main__":
    l1, l2, l3, l4 = 2.0, 0.5, 1.5, 1.2
    m2, m3, m4 = 1.0, 1.0, 1.0
    theta2_0, omega2_0 = 0.5, 1.0
    config = 'open'

    # 均匀杆参数
    r2, r3, r4 = l2 / 2, l3 / 2, l4 / 2
    I2 = m2 * l2**2 / 12
    I3 = m3 * l3**2 / 12
    I4 = m4 * l4**2 / 12

    # Grashof 类型
    g_type = grashof_criterion(l1, l2, l3, l4)
    print(f"Grashof 类型    : {g_type}")

    # 极限位置
    t4_ext, t4_fold, t2_ext, t2_fold = toggle_positions(l1, l2, l3, l4)
    print(f"伸展极限 θ₄    : {np.degrees(t4_ext):.2f}°  (θ₂={np.degrees(t2_ext):.2f}°)")
    print(f"折叠极限 θ₄    : {np.degrees(t4_fold):.2f}°  (θ₂={np.degrees(t2_fold):.2f}°)")

    # 初始运动学
    t3_0, t4_0 = position_analysis(theta2_0, l1, l2, l3, l4, config)
    print(f"\n初始 θ₂={np.degrees(theta2_0):.2f}° ({config})")
    print(f"  θ₃ = {np.degrees(t3_0):.2f}°")
    print(f"  θ₄ = {np.degrees(t4_0):.2f}°")

    I_eff_0 = equivalent_inertia(theta2_0, l1, l2, l3, l4, m2, m3, m4,
                                 r2, r3, r4, I2, I3, I4, config=config)
    print(f"  I_eff = {I_eff_0:.6f} kg·m²")
    print(f"  E_0   = {0.5 * I_eff_0 * omega2_0**2:.6f} J")

    # 数值积分
    sol = simulate(l1, l2, l3, l4, m2, m3, m4,
                   theta2_0, omega2_0, tau=0.0, g=0.0,
                   config=config, t_end=10.0, n_points=501)

    # 运动学后处理
    theta2 = sol.y[0]
    omega2 = sol.y[1]

    t3_arr = np.empty_like(theta2)
    t4_arr = np.empty_like(theta2)
    for i in range(len(theta2)):
        t3_arr[i], t4_arr[i] = position_analysis(theta2[i], l1, l2, l3, l4, config)

    # 能量
    E = np.array([mechanical_energy(
        [theta2[i], omega2[i]], l1, l2, l3, l4, m2, m3, m4,
        r2, r3, r4, I2, I3, I4, g=0.0, config=config)
        for i in range(len(theta2))])

    # 闭环约束残差
    residual = np.sqrt(
        (l2 * np.cos(theta2) + l3 * np.cos(t3_arr)
         - l4 * np.cos(t4_arr) - l1)**2
        + (l2 * np.sin(theta2) + l3 * np.sin(t3_arr)
           - l4 * np.sin(t4_arr))**2
    )

    # I_eff 范围
    I_eff_arr = np.array([equivalent_inertia(
        theta2[i], l1, l2, l3, l4, m2, m3, m4,
        r2, r3, r4, I2, I3, I4, config=config)
        for i in range(len(theta2))])

    print(f"\n=== 数值积分结果 ===")
    print(f"时间点数        : {len(sol.t)}")
    print(f"θ₂ 范围         : [{np.min(theta2):.4f}, {np.max(theta2):.4f}] rad")
    print(f"ω₂ 范围         : [{np.min(omega2):.6f}, {np.max(omega2):.6f}] rad/s")
    print(f"θ₄ 范围         : [{np.degrees(np.min(t4_arr)):.2f}°, {np.degrees(np.max(t4_arr)):.2f}°]")
    print(f"I_eff 范围      : [{np.min(I_eff_arr):.6f}, {np.max(I_eff_arr):.6f}] kg·m²")
    print(f"能量漂移        : {np.max(np.abs(E - E[0])):.3e} J")
    print(f"闭环约束残差    : {np.max(residual):.3e} m")
