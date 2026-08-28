"""MEC-042 —— 用 SciPy 数值求解碰撞动力学。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (dynamics, solve_collision, analytical_motion,
                   kinetic_energy_before, kinetic_energy_after,
                   energy_loss, momentum_before, momentum_after,
                   reduced_mass, validate_parameters)


def simulate(m1=1.0, m2=1.0, e=0.8, v1_before=3.0, v2_before=-2.0,
             t_collision=1.0, x1_0=0.0, x2_0=5.0, t_end=3.0, n_points=1001):
    """数值积分碰撞动力学。"""
    validate_parameters(m1=m1, m2=m2, e=e)
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=[x1_0, v1_before, x2_0, v2_before],
        t_eval=t_eval,
        args=(m1, m2, e, t_collision),
        rtol=1e-10, atol=1e-12,
        dense_output=False,
    )
    return sol


if __name__ == "__main__":
    m1, m2 = 1.0, 2.0
    e = 0.8
    v1b, v2b = 3.0, -2.0
    t_col = 1.0
    x1_0, x2_0 = 0.0, 5.0  # x2 > x1，质点1追质点2

    print(f"碰撞: m1={m1}, m2={m2}, e={e}")
    print(f"碰前: v1={v1b}, v2={v2b}")
    print(f"约化质量: {reduced_mass(m1, m2):.6f}")

    # 解析解
    v1a, v2a = solve_collision(v1b, v2b, m1, m2, e)
    print(f"碰后: v1={v1a:.6f}, v2={v2a:.6f}")

    # 动量守恒
    p_before = momentum_before(v1b, v2b, m1, m2)
    p_after = momentum_after(v1a, v2a, m1, m2)
    print(f"动量: 前={p_before:.6f}, 后={p_after:.6f}, 守恒={abs(p_before-p_after)<1e-12}")

    # 能量
    T_before = kinetic_energy_before(v1b, v2b, m1, m2)
    T_after = kinetic_energy_after(v1a, v2a, m1, m2)
    dT_formula = energy_loss(v1b, v2b, m1, m2, e)
    print(f"动能: 前={T_before:.6f}, 后={T_after:.6f}, ΔT={T_after-T_before:.6f}")
    print(f"公式: ΔT={dT_formula:.6f}, 一致={abs((T_after-T_before)-dT_formula)<1e-12}")

    # 弹性碰撞对比
    v1_elastic, v2_elastic = solve_collision(v1b, v2b, m1, m2, 1.0)
    T_elastic = kinetic_energy_after(v1_elastic, v2_elastic, m1, m2)
    print(f"\n弹性碰撞 (e=1): v1={v1_elastic:.6f}, v2={v2_elastic:.6f}, T={T_elastic:.6f} (守恒={abs(T_elastic-T_before)<1e-12})")

    # 塑性碰撞对比
    v1_plastic, v2_plastic = solve_collision(v1b, v2b, m1, m2, 0.0)
    print(f"塑性碰撞 (e=0): v1={v1_plastic:.6f}, v2={v2_plastic:.6f} (粘在一起)")

    # 数值积分
    sol = simulate(m1=m1, m2=m2, e=e, v1_before=v1b, v2_before=v2b,
                   t_collision=t_col, x1_0=x1_0, x2_0=x2_0, t_end=3.0)

    # 解析运动对照
    x1_ana, v1_ana, x2_ana, v2_ana = analytical_motion(
        sol.t, v1b, v2b, m1, m2, e, t_col, x1_0, x2_0)

    err_x1 = np.max(np.abs(sol.y[0] - x1_ana))
    err_v1 = np.max(np.abs(sol.y[1] - v1_ana))

    print(f"\n=== 数值积分结果 ===")
    print(f"解析解误差: x1={err_x1:.3e}, v1={err_v1:.3e}")
    print(f"v1(end) = {sol.y[1][-1]:.6f} (解析={v1a:.6f})")
    print(f"v2(end) = {sol.y[3][-1]:.6f} (解析={v2a:.6f})")
