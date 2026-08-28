"""MEC-006 —— 用 SciPy 数值求解胡克型中心力场运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters


def simulate(x0=1.0, y0=0.0, vx0=0.0, vy0=1.0, k=1.0, m=1.0,
             t_end=6.28318530718, n_points=101):
    """数值积分胡克型中心力场运动，返回 (t, x, y, vx, vy) 序列。

    参数
    ----
    x0, y0 : float
        初始位置（水平/垂直）。
    vx0, vy0 : float
        初始速度（水平/垂直）。
    k : float, optional
        弹性系数（默认 1.0 N/m）。
    m : float, optional
        质量（默认 1.0 kg）。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, x, y, vx, vy) : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    initial_state = np.array([x0, y0, vx0, vy0], dtype=float)
    validate_parameters(k=k, m=m)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,          # 模型方程，来自 model.py
        (0.0, t_end),      # 时间区间 [t0, t_end]
        y0=initial_state,  # 初始状态 [x, y, vx, vy]
        t_eval=t_eval,     # 指定输出时刻
        args=(k, m),       # 传递给 dynamics 的额外参数
        rtol=1e-9, atol=1e-12,
    )

    # sol.y 形状为 (4, n_points)
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


if __name__ == "__main__":
    x0, y0, vx0, vy0, k, m = 1.0, 0.0, 0.0, 1.0, 1.0, 1.0
    t_end = 2.0 * np.pi  # 一个完整周期
    t, x_num, y_num, vx_num, vy_num = simulate(
        x0=x0, y0=y0, vx0=vx0, vy0=vy0, k=k, m=m, t_end=t_end
    )

    # 解析解对照
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], k=k, m=m
    )

    err_x = np.max(np.abs(x_num - x_ana))
    err_y = np.max(np.abs(y_num - y_ana))
    err_vx = np.max(np.abs(vx_num - vx_ana))
    err_vy = np.max(np.abs(vy_num - vy_ana))

    # 守恒量验证
    from model import angular_momentum, mechanical_energy
    L_num = np.array([angular_momentum([x, y, vx, vy], m=m)
                      for x, y, vx, vy in zip(x_num, y_num, vx_num, vy_num)])
    E_num = np.array([mechanical_energy([x, y, vx, vy], k=k, m=m)
                      for x, y, vx, vy in zip(x_num, y_num, vx_num, vy_num)])

    print(f"时间点数      : {len(t)}")
    print(f"终止时间      : {t_end:.6f} s")
    print(f"初始状态      : [{x_num[0]:.6f}, {y_num[0]:.6f}, "
          f"{vx_num[0]:.6f}, {vy_num[0]:.6f}]")
    print(f"末点数值 [x,y]: [{x_num[-1]:.6f}, {y_num[-1]:.6f}]")
    print(f"末点解析 [x,y]: [{x_ana[-1]:.6f}, {y_ana[-1]:.6f}]")
    print(f"最大 x 误差   : {err_x:.3e}")
    print(f"最大 y 误差   : {err_y:.3e}")
    print(f"最大 vx 误差  : {err_vx:.3e}")
    print(f"最大 vy 误差  : {err_vy:.3e}")
    print(f"角动量均值    : {np.mean(L_num):.6f}，标准差 {np.std(L_num):.3e}")
    print(f"机械能均值    : {np.mean(E_num):.6f}，标准差 {np.std(E_num):.3e}")
