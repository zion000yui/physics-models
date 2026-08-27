"""MEC-003 —— 用 SciPy 数值求解抛体运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics


def simulate(x0=0.0, y0=10.0, vx0=10.0, vy0=15.0, g=9.81,
             t_end=5.0, n_points=101):
    """数值积分抛体运动，返回 (t, x, y, vx, vy) 序列。

    参数
    ----
    x0, y0 : float
        初始位置（水平/垂直）。
    vx0, vy0 : float
        初始速度（水平/垂直）。
    g : float, optional
        重力加速度（默认 9.81 m/s²）。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, x, y, vx, vy) : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,          # 模型方程，来自 model.py
        (0.0, t_end),      # 时间区间 [t0, t_end]
        y0=[x0, y0, vx0, vy0],  # 初始状态 [x, y, vx, vy]
        t_eval=t_eval,     # 指定输出时刻
        args=(g,),         # 传递给 dynamics 的额外参数 g
    )

    # sol.y 形状为 (4, n_points)
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


if __name__ == "__main__":
    x0, y0, vx0, vy0, g, t_end = 0.0, 10.0, 10.0, 15.0, 9.81, 5.0
    t, x_num, y_num, vx_num, vy_num = simulate(
        x0=x0, y0=y0, vx0=vx0, vy0=vy0, g=g, t_end=t_end
    )

    # 解析解对照
    x_ana, y_ana, vx_ana, vy_ana = analytical(t,
                                                [x0, y0, vx0, vy0], g=g)

    err_x = np.max(np.abs(x_num - x_ana))
    err_y = np.max(np.abs(y_num - y_ana))
    err_vx = np.max(np.abs(vx_num - vx_ana))
    err_vy = np.max(np.abs(vy_num - vy_ana))

    print(f"时间点数      : {len(t)}")
    print(f"初始状态      : [{x_num[0]:.6f}, {y_num[0]:.6f}, "
          f"{vx_num[0]:.6f}, {vy_num[0]:.6f}]")
    print(f"末点数值 [x,y]: [{x_num[-1]:.6f}, {y_num[-1]:.6f}]")
    print(f"末点解析 [x,y]: [{x_ana[-1]:.6f}, {y_ana[-1]:.6f}]")
    print(f"末点数值 [vx,vy]: [{vx_num[-1]:.6f}, {vy_num[-1]:.6f}]")
    print(f"末点解析 [vx,vy]: [{vx_ana[-1]:.6f}, {vy_ana[-1]:.6f}]")
    print(f"最大 x 误差  : {err_x:.3e}")
    print(f"最大 y 误差  : {err_y:.3e}")
    print(f"最大 vx 误差 : {err_vx:.3e}")
    print(f"最大 vy 误差 : {err_vy:.3e}")
