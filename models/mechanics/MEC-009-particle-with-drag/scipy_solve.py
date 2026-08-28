"""MEC-009 —— 用 SciPy 数值求解速度相关阻力下的质点运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, terminal_velocity


def simulate(x0=0.0, y0=0.0, vx0=10.0, vy0=15.0,
             g=9.81, b=0.5, c=0.0, m=1.0,
             t_end=5.0, n_points=101):
    """数值积分阻力运动轨迹，返回 (t, x, y, vx, vy) 序列。

    参数
    ----
    x0, y0 : float
        初始位置（水平/垂直）。
    vx0, vy0 : float
        初始速度（水平/垂直）。
    g : float, optional
        重力加速度（默认 9.81 m/s²）。
    b : float, optional
        线性阻力系数（默认 0.5）。
    c : float, optional
        二次阻力系数（默认 0.0，即无二次阻力）。
    m : float, optional
        质量（默认 1.0 kg）。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, x, y, vx, vy) : (np.ndarray, ...) 五元组
    """
    initial_state = np.array([x0, y0, vx0, vy0], dtype=float)
    validate_parameters(g=g, b=b, c=c, m=m)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(g, b, c, m),
        rtol=1e-9, atol=1e-12,
    )

    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


if __name__ == "__main__":
    # 线性阻力示例
    g, b, c, m = 9.81, 0.5, 0.0, 1.0
    x0, y0, vx0, vy0 = 0.0, 0.0, 10.0, 15.0
    t_end = 5.0
    t, x_num, y_num, vx_num, vy_num = simulate(
        x0=x0, y0=y0, vx0=vx0, vy0=vy0,
        g=g, b=b, c=c, m=m, t_end=t_end)

    # 解析解对照（线性阻力有闭式解）
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], g=g, b=b, c=c, m=m)

    err_x = np.max(np.abs(x_num - x_ana))
    err_y = np.max(np.abs(y_num - y_ana))
    err_vx = np.max(np.abs(vx_num - vx_ana))
    err_vy = np.max(np.abs(vy_num - vy_ana))

    vt = terminal_velocity(g=g, b=b, c=c, m=m)

    # 机械能（m=1）：E = 0.5*(vx²+vy²) + g*y
    E_num = 0.5 * (vx_num ** 2 + vy_num ** 2) + g * y_num

    print("=== 线性阻力（b=0.5, c=0）===")
    print(f"参数         : g={g}, b={b}, c={c}, m={m}")
    print(f"初始状态     : [{x0}, {y0}, {vx0}, {vy0}]")
    print(f"终态速度     : {vt:.4f} m/s" if vt else "无终态速度")
    print(f"时间点数     : {len(t)}")
    print(f"终止时间     : {t_end:.2f} s")
    print(f"末点数值 [x,y]: [{x_num[-1]:.6f}, {y_num[-1]:.6f}]")
    print(f"末点解析 [x,y]: [{x_ana[-1]:.6f}, {y_ana[-1]:.6f}]")
    print(f"最大 x 误差  : {err_x:.3e}")
    print(f"最大 y 误差  : {err_y:.3e}")
    print(f"最大 vx 误差 : {err_vx:.3e}")
    print(f"最大 vy 误差 : {err_vy:.3e}")
    print(f"机械能初值   : {E_num[0]:.6f}")
    print(f"机械能末值   : {E_num[-1]:.6f}")
    print(f"机械能耗散   : {E_num[0]-E_num[-1]:.6f} J")
