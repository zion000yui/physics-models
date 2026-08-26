"""MEC-002 —— 用 SciPy 数值求解受恒定外力作用的质点轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics


def simulate(x0=0.0, v0=2.0, t_end=10.0, n_points=101, F=1.0, m=1.0):
    """数值积分受恒定外力作用的质点，返回 (t, x, v) 序列。

    参数
    ----
    x0, v0 : float
        初始位置 / 初始速度。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。
    F : float, optional
        恒定外力（默认 1.0）。
    m : float, optional
        质点质量（默认 1.0）。

    返回
    ----
    (t, x, v) : (np.ndarray, np.ndarray, np.ndarray)
    """
    t_eval = np.linspace(0.0, t_end, n_points)

    # solve_ivp：对一阶 ODE 做数值积分（默认 RK45，带误差控制）
    sol = solve_ivp(
        dynamics,          # 模型方程，来自 model.py
        (0.0, t_end),      # 时间区间 [t0, t_end]
        y0=[x0, v0],       # 初始状态 [x0, v0]
        t_eval=t_eval,     # 指定输出时刻
        args=(F, m),       # 传递给 dynamics 的额外参数 (F, m)
    )

    # sol.y 形状为 (2, n_points)：sol.y[0] 是位置，sol.y[1] 是速度
    return sol.t, sol.y[0], sol.y[1]


if __name__ == "__main__":
    x0, v0, t_end, F, m = 0.0, 2.0, 10.0, 1.0, 1.0
    t, x_num, v_num = simulate(x0=x0, v0=v0, t_end=t_end, F=F, m=m)

    # 解析解对照
    x_ana, v_ana = analytical(t, x0, v0, F, m)

    err_x = np.max(np.abs(x_num - x_ana))
    err_v = np.max(np.abs(v_num - v_ana))

    print(f"时间点数      : {len(t)}")
    print(f"初始状态 [x,v]: [{x_num[0]:.6f}, {v_num[0]:.6f}]")
    print(f"末点数值 x    : {x_num[-1]:.6f}")
    print(f"末点解析 x    : {x_ana[-1]:.6f}")
    print(f"末点数值 v    : {v_num[-1]:.6f}")
    print(f"末点解析 v    : {v_ana[-1]:.6f}")
    print(f"最大位置误差  : {err_x:.3e}")
    print(f"最大速度误差  : {err_v:.3e}")
