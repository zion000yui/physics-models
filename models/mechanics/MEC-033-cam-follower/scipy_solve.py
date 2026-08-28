"""MEC-033 —— 用 SciPy 数值求解凸轮从动件动力学。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, follower_displacement, follower_velocity_ratio,
                   follower_acceleration_ratio, equivalent_inertia,
                   equivalent_inertia_derivative, contact_force,
                   pressure_angle, mechanical_energy, validate_parameters)

# 默认参数
H = 0.01
B_RISE = np.pi / 2
B_DWELL1 = np.pi / 4
B_RETURN = np.pi / 2
I_CAM = 0.001
M_F = 0.1
K = 100.0
R_B = 0.03
PROFILE = 'cycloidal'


def simulate(h=H, beta_rise=B_RISE, beta_dwell1=B_DWELL1, beta_return=B_RETURN,
             I_cam=I_CAM, m_f=M_F, k=K, tau=0.0, r_b=R_B, profile=PROFILE,
             theta0=0.0, omega0=10.0, t_end=2.0, n_points=501):
    """数值积分凸轮从动件动力学。"""
    validate_parameters(h=h, beta_rise=beta_rise, beta_dwell1=beta_dwell1,
                        beta_return=beta_return, I_cam=I_cam, m_f=m_f, k=k, r_b=r_b)
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=[theta0, omega0],
        t_eval=t_eval,
        args=(h, beta_rise, beta_dwell1, beta_return, I_cam, m_f, k, tau, r_b, profile),
        rtol=1e-10, atol=1e-12,
    )
    return sol


if __name__ == "__main__":
    theta0, omega0 = 0.0, 10.0

    b2 = B_RISE + B_DWELL1
    b3 = b2 + B_RETURN
    b4 = 2 * np.pi
    print(f"DRRD: Rise[0,{B_RISE:.4f}] Dwell1[{B_RISE:.4f},{b2:.4f}] "
          f"Return[{b2:.4f},{b3:.4f}] Dwell2[{b3:.4f},{b4:.4f}]")
    print(f"轮廓: {PROFILE}, 升程: {H*1000:.1f} mm, 弹簧: {K:.0f} N/m")

    # 初始运动学
    y0 = follower_displacement(theta0, H, B_RISE, B_DWELL1, B_RETURN, PROFILE)
    yp0 = follower_velocity_ratio(theta0, H, B_RISE, B_DWELL1, B_RETURN, PROFILE)
    I_eff0 = equivalent_inertia(theta0, H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, PROFILE)
    E0 = 0.5 * I_eff0 * omega0**2 + 0.5 * K * y0**2
    print(f"\n初始 θ={np.degrees(theta0):.1f}°")
    print(f"  y = {y0*1000:.4f} mm, dy/dθ = {yp0:.6f}")
    print(f"  I_eff = {I_eff0:.8f} kg·m², E_0 = {E0:.6f} J")

    # 数值积分
    sol = simulate(theta0=theta0, omega0=omega0, t_end=2.0)
    theta = sol.y[0]
    omega = sol.y[1]

    # 运动学后处理
    y_arr = np.array([follower_displacement(t, H, B_RISE, B_DWELL1, B_RETURN, PROFILE) for t in theta])
    I_eff_arr = np.array([equivalent_inertia(t, H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, PROFILE) for t in theta])

    # 能量
    E = np.array([mechanical_energy([t, w], H, B_RISE, B_DWELL1, B_RETURN, I_CAM, M_F, K, PROFILE)
                  for t, w in zip(theta, omega)])

    # 接触力
    from model import follower_acceleration_ratio
    F_arr = np.array([contact_force(t, w, dynamics(0, [t, w], H, B_RISE, B_DWELL1, B_RETURN,
                                     I_CAM, M_F, K, 0, R_B, PROFILE)[1],
                                     H, B_RISE, B_DWELL1, B_RETURN, M_F, K, PROFILE)
                      for t, w in zip(theta, omega)])

    # 压力角
    pa_arr = np.array([np.degrees(pressure_angle(t, H, B_RISE, B_DWELL1, B_RETURN, R_B, PROFILE))
                       for t in theta])

    print(f"\n=== 数值积分结果 ===")
    print(f"时间点数        : {len(sol.t)}")
    print(f"θ 范围          : [{np.min(theta):.4f}, {np.max(theta):.4f}] rad")
    print(f"ω 范围          : [{np.min(omega):.6f}, {np.max(omega):.6f}] rad/s")
    print(f"y 范围          : [{np.min(y_arr)*1000:.4f}, {np.max(y_arr)*1000:.4f}] mm")
    print(f"I_eff 范围      : [{np.min(I_eff_arr):.8f}, {np.max(I_eff_arr):.8f}] kg·m²")
    print(f"能量漂移        : {np.max(np.abs(E - E[0])):.3e} J")
    print(f"min(F_contact)  : {np.min(F_arr):.4f} N ({'保持接触' if np.min(F_arr) > 0 else '跳脱!'})")
    print(f"max(压力角)     : {np.max(pa_arr):.2f}°")
