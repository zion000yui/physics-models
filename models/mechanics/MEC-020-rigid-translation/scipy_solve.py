"""MEC-020 —— 用 SciPy 数值求解刚体平动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, momentum


def simulate(x0=0.0, y0=0.0, vx0=10.0, vy0=15.0,
             m=1.0, Fx=0.0, Fy=0.0,
             t_end=5.0, n_points=101):
    """数值积分刚体平动轨迹，返回 (t, x, y, vx, vy) 序列。

    参数
    ----
    x0, y0 : float
        初始质心位置。
    vx0, vy0 : float
        初始质心速度。
    m : float, optional
        刚体质量（默认 1.0 kg）。
    Fx, Fy : float, optional
        合外力分量（默认 0.0 N）。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, x, y, vx, vy) : (np.ndarray, ...) 五元组
    """
    initial_state = np.array([x0, y0, vx0, vy0], dtype=float)
    validate_parameters(m=m, Fx=Fx, Fy=Fy)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(m, Fx, Fy),
        rtol=1e-9, atol=1e-12,
    )

    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


if __name__ == "__main__":
    # 恒力+重力示例
    m = 2.0
    Fx = 5.0
    Fy = -m * 9.81  # 重力
    x0, y0, vx0, vy0 = 0.0, 0.0, 10.0, 15.0

    print(f"参数: m={m}, Fx={Fx}, Fy={Fy:.2f}")
    print(f"初始状态: x0={x0}, y0={y0}, vx0={vx0}, vy0={vy0}")

    t_end = 3.0
    t, x_n, y_n, vx_n, vy_n = simulate(
        x0, y0, vx0, vy0, m=m, Fx=Fx, Fy=Fy, t_end=t_end)

    x_a, y_a, vx_a, vy_a = analytical(
        t, [x0, y0, vx0, vy0], m=m, Fx=Fx, Fy=Fy)

    err_x = np.max(np.abs(x_n - x_a))
    err_y = np.max(np.abs(y_n - y_a))
    err_vx = np.max(np.abs(vx_n - vx_a))
    err_vy = np.max(np.abs(vy_n - vy_a))

    # 动量检查
    P_num = np.array([momentum(
        [x_n[i], y_n[i], vx_n[i], vy_n[i]], m=m)
        for i in range(len(t))])

    print(f"时间点数      : {len(t)}")
    print(f"终止时间      : {t_end:.2f} s")
    print(f"最大 x 误差   : {err_x:.3e}")
    print(f"最大 y 误差   : {err_y:.3e}")
    print(f"最大 vx 误差  : {err_vx:.3e}")
    print(f"最大 vy 误差  : {err_vy:.3e}")
    print(f"动量 Px 标准差 : {np.std(P_num[:, 0]):.3e}")
    print(f"动量 Py 标准差 : {np.std(P_num[:, 1]):.3e}")
