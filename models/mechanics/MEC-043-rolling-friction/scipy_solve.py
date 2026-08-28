"""MEC-043 —— 用 SciPy 数值求解滚动摩擦动力学。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, analytical, mechanical_energy,
                   effective_mass, rolling_friction_force,
                   validate_parameters)


def simulate(m=1.0, R=0.5, I=None, g=9.81, mu_r=0.01,
             v0=3.0, t_end=5.0, n_points=1001):
    """数值积分滚动摩擦动力学。"""
    if I is None:
        I = 0.4 * m * R**2
    validate_parameters(m=m, R=R, I=I, g=g, mu_r=mu_r)
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=[0.0, v0],
        t_eval=t_eval,
        args=(m, R, I, g, mu_r),
        rtol=1e-10, atol=1e-12)
    return sol


if __name__ == "__main__":
    m, R = 1.0, 0.5
    I = 0.4 * m * R**2  # 实心球
    g, mu_r = 9.81, 0.01
    v0 = 3.0

    m_eff = effective_mass(m, I, R)
    a_decel = mu_r * m * g / m_eff
    t_stop = abs(v0) / a_decel
    x_stop = v0**2 / (2 * a_decel)

    print(f"刚体: m={m}, R={R}, I={I:.6f} (实心球)")
    print(f"滚动摩擦: μ_r={mu_r}")
    print(f"有效质量: m_eff={m_eff:.6f} kg")
    print(f"减速度: a={a_decel:.6f} m/s²")
    print(f"停止时间: t_stop={t_stop:.4f} s")
    print(f"停止距离: x_stop={x_stop:.4f} m")

    # 数值积分
    sol = simulate(m=m, R=R, I=I, g=g, mu_r=mu_r, v0=v0, t_end=t_stop*2)

    x = sol.y[0]
    v = sol.y[1]
    t = sol.t

    # 解析解
    x_ana, v_ana = analytical(t, v0, m, R, I, g, mu_r)
    err_x = np.max(np.abs(x - x_ana))
    err_v = np.max(np.abs(v - v_ana))

    # 能量
    E = np.array([mechanical_energy([x[i], v[i]], m, R, I) for i in range(len(t))])

    print(f"\n=== 数值积分结果 ===")
    print(f"解析误差: x={err_x:.3e}, v={err_v:.3e}")
    print(f"E(0)={E[0]:.6f}, E(end)={E[-1]:.6f}, ΔE={E[-1]-E[0]:.6f}")
    print(f"能量单调递减: {np.all(np.diff(E[::50]) <= 1e-8)}")
    print(f"v(end)={v[-1]:.6f} (应≈0)")
    print(f"x(end)={x[-1]:.4f} (应≈{x_stop:.4f})")
