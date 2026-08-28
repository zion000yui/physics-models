"""MEC-041 —— 用 SciPy 数值求解库仑摩擦动力学。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, friction_force, mechanical_energy,
                   analytical_constant_force, validate_parameters,
                   normal_force, is_sliding)


def simulate(m=1.0, g=9.81, mu_s=0.3, mu_k=0.25, F_ext=5.0,
             x0=0.0, v0=0.0, t_end=3.0, n_points=1001):
    """数值积分库仑摩擦动力学。"""
    validate_parameters(m=m, g=g, mu_s=mu_s, mu_k=mu_k)
    t_eval = np.linspace(0.0, t_end, n_points)

    # F_ext 作为常数通过 args 传入
    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=[x0, v0],
        t_eval=t_eval,
        args=(m, g, mu_s, mu_k, F_ext),
        rtol=1e-10, atol=1e-12,
    )
    return sol


if __name__ == "__main__":
    m, g = 1.0, 9.81
    mu_s, mu_k = 0.3, 0.25
    F_ext = 5.0
    v0 = 0.0

    N = normal_force(m, g)
    F_max_s = mu_s * N
    F_k = mu_k * N

    print(f"质点: m={m}, g={g}, N={N:.2f} N")
    print(f"摩擦: μ_s={mu_s}, μ_k={mu_k}")
    print(f"力:   F_ext={F_ext}, F_max_static={F_max_s:.4f}, F_kinetic={F_k:.4f}")
    print(f"状态: {'动摩擦(超静摩擦)' if abs(F_ext) > F_max_s else '静摩擦(静止)'}")

    # 解析解
    x_ana, v_ana = analytical_constant_force(
        np.linspace(0, 3, 1001), v0, m, g, mu_s, mu_k, F_ext)
    if abs(F_ext) > F_max_s:
        a_net = (F_ext - mu_k * N * np.sign(F_ext)) / m
        print(f"净加速度: a = (F_ext - μ_k·N·sign)/m = {a_net:.4f} m/s²")

    # 数值积分
    sol = simulate(m=m, g=g, mu_s=mu_s, mu_k=mu_k, F_ext=F_ext, v0=v0, t_end=3.0)

    x = sol.y[0]
    v = sol.y[1]
    t = sol.t

    # 能量
    E = np.array([mechanical_energy([x[i], v[i]], m) for i in range(len(t))])

    # 解析解对照
    x_a, v_a = analytical_constant_force(t, v0, m, g, mu_s, mu_k, F_ext)
    err_x = np.max(np.abs(x - x_a))
    err_v = np.max(np.abs(v - v_a))

    print(f"\n=== 数值积分结果 ===")
    print(f"时间点数        : {len(t)}")
    print(f"x 范围          : [{np.min(x):.4f}, {np.max(x):.4f}] m")
    print(f"v 范围          : [{np.min(v):.4f}, {np.max(v):.4f}] m/s")
    print(f"解析解误差      : x={err_x:.3e}, v={err_v:.3e}")
    print(f"E(0)={E[0]:.6f}, E(end)={E[-1]:.6f}")

    # 摩擦耗散
    if abs(F_ext) > F_max_s:
        W_ext = F_ext * (x[-1] - x[0])
        W_friction = -mu_k * N * abs(x[-1] - x[0])
        dE = E[-1] - E[0]
        print(f"外力做功 W_ext   : {W_ext:.6f} J")
        print(f"摩擦耗散 W_fric  : {W_friction:.6f} J")
        print(f"W_ext + W_fric   : {W_ext + W_friction:.6f} J (应 = ΔE = {dE:.6f})")

    # 静摩擦场景（F_ext < F_max_s）
    F_small = 1.0
    print(f"\n=== 静摩擦场景 (F_ext={F_small} < F_max_s={F_max_s:.2f}) ===")
    sol_s = simulate(m=m, g=g, mu_s=mu_s, mu_k=mu_k, F_ext=F_small, v0=0.0, t_end=2.0)
    print(f"x(end) = {sol_s.y[0][-1]:.6e} (应≈0，静摩擦)")
    print(f"v(end) = {sol_s.y[1][-1]:.6e} (应≈0)")

    # 自由滑动减速场景（有初速度，无外力）
    print(f"\n=== 自由减速场景 (v0=5, F_ext=0) ===")
    sol_d = simulate(m=m, g=g, mu_s=mu_s, mu_k=mu_k, F_ext=0.0, v0=5.0, t_end=3.0)
    x_d = sol_d.y[0]
    v_d = sol_d.y[1]
    print(f"v(end) = {v_d[-1]:.6f} (应≈0，摩擦减速至停止)")
    print(f"x(end) = {x_d[-1]:.4f} m")
    # 解析：减速时间 t_stop = v0/(μ_k·g), 停止距离 = v0²/(2·μ_k·g)
    t_stop = 5.0 / (mu_k * g)
    x_stop = 5.0**2 / (2 * mu_k * g)
    print(f"解析: t_stop={t_stop:.4f}s, x_stop={x_stop:.4f}m")
