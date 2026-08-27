"""MEC-005 —— 用 SciPy 数值求解非匀速圆周运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_initial_state


def simulate(x0=1.0, y0=0.0, vx0=0.0, vy0=1.0, R=1.0, omega0=1.0,
             alpha=0.0, xc=0.0, yc=0.0, t_end=6.28318530718, n_points=101):
    """数值积分非匀速圆周运动，返回 (t, x, y, vx, vy) 序列。

    参数
    ----
    x0, y0 : float
        初始位置（水平/垂直）。
    vx0, vy0 : float
        初始速度（水平/垂直）。
    R : float, optional
        圆周半径（默认 1.0 m）。
    omega0 : float, optional
        初始角速度（默认 1.0 rad/s）。
    alpha : float, optional
        常数角加速度（默认 0.0 rad/s²）。
    xc, yc : float, optional
        圆心坐标（默认原点）。
    t_end : float
        仿真终止时间。默认 2π，即一个完整周期（仅当 alpha=0 时为完整周期）。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, x, y, vx, vy) : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    initial_state = np.array([x0, y0, vx0, vy0], dtype=float)

    # 入口验证：确保初始状态满足圆周运动条件
    validate_initial_state(initial_state, R, omega0, xc, yc)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,          # 模型方程，来自 model.py
        (0.0, t_end),      # 时间区间 [t0, t_end]
        y0=initial_state,  # 初始状态 [x, y, vx, vy]
        t_eval=t_eval,     # 指定输出时刻
        args=(R, omega0, alpha, xc, yc),  # 传递给 dynamics 的额外参数
        rtol=1e-9, atol=1e-12,
    )

    # sol.y 形状为 (4, n_points)
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


if __name__ == "__main__":
    x0, y0, vx0, vy0 = 1.0, 0.0, 0.0, 1.0
    R, omega0, alpha, xc, yc = 1.0, 1.0, 0.0, 0.0, 0.0
    t_end = 2.0 * np.pi  # 一个完整周期（仅 alpha=0 时闭合）
    t, x_num, y_num, vx_num, vy_num = simulate(
        x0=x0, y0=y0, vx0=vx0, vy0=vy0,
        R=R, omega0=omega0, alpha=alpha, xc=xc, yc=yc, t_end=t_end
    )

    # 解析解对照
    x_ana, y_ana, vx_ana, vy_ana = analytical(
        t, [x0, y0, vx0, vy0], R=R, omega0=omega0, alpha=alpha, xc=xc, yc=yc
    )

    err_x = np.max(np.abs(x_num - x_ana))
    err_y = np.max(np.abs(y_num - y_ana))
    err_vx = np.max(np.abs(vx_num - vx_ana))
    err_vy = np.max(np.abs(vy_num - vy_ana))

    omega_t = omega0 + alpha * t
    speed_num = np.hypot(vx_num, vy_num)
    expected_speed = R * np.abs(omega_t)
    speed_err = np.max(np.abs(speed_num - expected_speed))

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
    print(f"速率最大误差  : {speed_err:.3e}")
