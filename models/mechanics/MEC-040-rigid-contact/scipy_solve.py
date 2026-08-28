"""MEC-040 —— 用 SciPy 数值求解刚性接触动力学。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, contact_force, mechanical_energy,
                   analytical_free_flight, validate_parameters)


def simulate(m=1.0, g=9.81, k_c=1e4, c_c=0.0,
             y0=1.0, v0=0.0, t_end=3.0, n_points=2001):
    """数值积分接触动力学。"""
    validate_parameters(m=m, g=g, k_c=k_c, c_c=c_c)
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=[y0, v0],
        t_eval=t_eval,
        args=(m, g, k_c, c_c),
        rtol=1e-10, atol=1e-12,
        method='RK45',
    )
    return sol


if __name__ == "__main__":
    m, g = 1.0, 9.81
    k_c, c_c = 1e4, 0.0  # 无阻尼（弹跳）
    y0, v0 = 1.0, 0.0

    print(f"质点: m={m} kg, y0={y0} m, v0={v0} m/s")
    print(f"接触: k_c={k_c:.0f} N/m, c_c={c_c:.1f} N·s/m")

    # 自由下落时间
    t_fall = np.sqrt(2 * y0 / g)
    v_impact = -g * t_fall
    print(f"预计落地: t={t_fall:.4f} s, v={v_impact:.4f} m/s")

    # 数值积分
    sol = simulate(m=m, g=g, k_c=k_c, c_c=c_c, y0=y0, v0=v0, t_end=3.0)

    y = sol.y[0]
    v = sol.y[1]
    t = sol.t

    # 运动学分析
    contact_times = []
    for i in range(1, len(t)):
        if y[i-1] > 0 and y[i] <= 0:
            contact_times.append(t[i])

    # 能量
    E = np.array([mechanical_energy([y[i], v[i]], m, g, k_c) for i in range(len(t))])

    # 自由飞行段解析解对照（第一次落地前）
    t_pre = t[t <= t_fall * 1.05]
    y_pre = y[:len(t_pre)]
    v_pre = v[:len(t_pre)]
    y_ana, v_ana = analytical_free_flight(t_pre, y0, v0, g)

    err_y = np.max(np.abs(y_pre - y_ana)) if len(t_pre) > 0 else 0
    err_v = np.max(np.abs(v_pre - v_ana)) if len(t_pre) > 0 else 0

    print(f"\n=== 数值积分结果 ===")
    print(f"时间点数            : {len(t)}")
    print(f"y 范围              : [{np.min(y):.6f}, {np.max(y):.6f}] m")
    print(f"v 范围              : [{np.min(v):.6f}, {np.max(v):.6f}] m/s")
    print(f"接触次数            : {len(contact_times)}")
    print(f"自由飞行段解析误差  : y={err_y:.3e}, v={err_v:.3e}")
    print(f"能量漂移(无阻尼)    : {np.max(np.abs(E - E[0])):.3e} J")
    print(f"最大穿透深度        : {np.min(y):.6f} m ({abs(np.min(y))*1000:.3f} mm)")

    # 有阻尼对比
    sol_d = simulate(m=m, g=g, k_c=k_c, c_c=10.0, y0=y0, v0=v0, t_end=3.0)
    y_d = sol_d.y[0]
    v_d = sol_d.y[1]
    E_d = np.array([mechanical_energy([y_d[i], v_d[i]], m, g, k_c) for i in range(len(y_d))])

    print(f"\n=== 有阻尼 (c_c=10) ===")
    print(f"y 范围              : [{np.min(y_d):.6f}, {np.max(y_d):.6f}] m")
    print(f"v 范围              : [{np.min(v_d):.6f}, {np.max(v_d):.6f}] m/s")
    print(f"能量变化            : E(0)={E_d[0]:.6f}, E(end)={E_d[-1]:.6f}, ΔE={E_d[-1]-E_d[0]:.6f}")
    print(f"能量单调递减        : {np.all(np.diff(E_d) <= 1e-10)}")
