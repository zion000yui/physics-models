"""MEC-032 —— 用 SciPy 数值求解齿轮传动动力学。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, analytical, equivalent_inertia,
                   transmission_ratio, output_kinematics,
                   contact_force, contact_force_from_output,
                   power_flow, mechanical_energy, validate_parameters)


def simulate(r1=0.1, r2=0.2, I1=0.01, I2=0.04,
             theta0=0.0, omega0=1.0, tau_in=1.0, tau_load=0.0,
             t_end=5.0, n_points=501):
    """数值积分齿轮动力学。"""
    validate_parameters(r1=r1, r2=r2, I1=I1, I2=I2, tau_in=tau_in, tau_load=tau_load)
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=[theta0, omega0],
        t_eval=t_eval,
        args=(r1, r2, I1, I2, tau_in, tau_load),
        rtol=1e-10, atol=1e-12,
    )
    return sol


if __name__ == "__main__":
    r1, r2 = 0.1, 0.2
    I1, I2 = 0.01, 0.04
    tau_in, tau_load = 1.0, 0.0
    theta0, omega0 = 0.0, 1.0

    i = transmission_ratio(r1, r2)
    I_eq = equivalent_inertia(I1, I2, r1, r2)
    alpha_const = (tau_in - i * tau_load) / I_eq

    print(f"传动比 i = r1/r2 = {i:.4f}")
    print(f"等效惯量 I_eq = I1 + i²·I2 = {I_eq:.6f} kg·m²")
    print(f"角加速度 α₁ = (τ_in - i·τ_load)/I_eq = {alpha_const:.4f} rad/s²")
    print()

    # 数值积分
    sol = simulate(r1, r2, I1, I2, theta0, omega0, tau_in, tau_load, t_end=5.0)

    # 解析解对照
    t = sol.t
    theta1_num, omega1_num = sol.y[0], sol.y[1]
    theta1_ana, omega1_ana = analytical(t, [theta0, omega0], r1, r2, I1, I2, tau_in, tau_load)

    err_theta = np.max(np.abs(theta1_num - theta1_ana))
    err_omega = np.max(np.abs(omega1_num - omega1_ana))

    # 输出齿轮运动学
    t2, w2, a2 = output_kinematics(theta1_num[-1], omega1_num[-1], alpha_const, r1, r2)

    # 接触力
    F1 = contact_force(tau_in, I1, alpha_const, r1)
    F2 = contact_force_from_output(tau_load, I2, alpha_const, r1, r2)

    # 功率
    P_in, P_out = power_flow(tau_in, omega1_num[-1], tau_load, r1, r2)

    # 能量
    E0 = mechanical_energy([theta0, omega0], r1, r2, I1, I2)
    E_end = mechanical_energy([theta1_num[-1], omega1_num[-1]], r1, r2, I1, I2)

    print(f"=== 数值积分结果 ===")
    print(f"时间点数            : {len(t)}")
    print(f"θ₁(末) 数值/解析    : {theta1_num[-1]:.6f} / {theta1_ana[-1]:.6f}")
    print(f"ω₁(末) 数值/解析    : {omega1_num[-1]:.6f} / {omega1_ana[-1]:.6f}")
    print(f"位置误差            : {err_theta:.3e}")
    print(f"速度误差            : {err_omega:.3e}")
    print()
    print(f"输出齿轮 θ₂(末)     : {t2:.6f} rad")
    print(f"输出齿轮 ω₂(末)     : {w2:.6f} rad/s")
    print(f"输出齿轮 α₂         : {a2:.6f} rad/s²")
    print()
    print(f"接触力 F (齿轮1)    : {F1:.6f} N")
    print(f"接触力 F (齿轮2)    : {F2:.6f} N")
    print(f"两者差异             : {abs(F1-F2):.3e}")
    print()
    print(f"输入功率 P_in       : {P_in:.6f} W")
    print(f"输出功率 P_out      : {P_out:.6f} W")
    print(f"动能变化率 dT/dt    : {I_eq * alpha_const * omega1_num[-1]:.6f} W")
    print(f"P_in - P_out        : {P_in - P_out:.6f} (应 = dT/dt)")
    print()
    print(f"E(0)                : {E0:.6f} J")
    print(f"E(末)               : {E_end:.6f} J")
    print(f"ΔE                  : {E_end - E0:.6f} J")
    print(f"τ_eff·Δθ            : {(tau_in - i*tau_load) * (theta1_num[-1] - theta0):.6f} J")
