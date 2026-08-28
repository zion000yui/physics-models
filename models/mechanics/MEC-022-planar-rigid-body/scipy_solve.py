"""MEC-022 —— 用 SciPy 数值求解平面刚体运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (analytical, dynamics, validate_parameters,
                   torque_from_force, momentum, angular_momentum,
                   mechanical_energy)


def simulate(x0=0.0, y0=0.0, vx0=0.0, vy0=0.0,
             theta0=0.0, omega0=0.0,
             m=1.0, I=1.0, Fx=0.0, Fy=4.0, rx=0.5, ry=0.0,
             t_end=5.0, n_points=101):
    """数值积分平面刚体轨迹，返回 (t, x, y, vx, vy, theta, omega)。"""
    initial_state = np.array([x0, y0, vx0, vy0, theta0, omega0],
                              dtype=float)
    validate_parameters(m=m, I=I)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(m, I, Fx, Fy, rx, ry),
        rtol=1e-9, atol=1e-12,
    )

    return (sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3],
            sol.y[4], sol.y[5])


if __name__ == "__main__":
    m, I = 1.0, 2.0
    Fx, Fy = 0.0, 4.0
    rx, ry = 0.5, 0.0
    x0, y0, vx0, vy0 = 0.0, 0.0, 0.0, 0.0
    theta0, omega0 = 0.0, 0.0

    tau = torque_from_force(Fx, Fy, rx, ry)

    print(f"参数: m={m}, I={I}, F=({Fx},{Fy}), r=({rx},{ry})")
    print(f"力矩 tau = rx*Fy - ry*Fx = {tau:.4f} N·m")
    print(f"质心加速度: a=({Fx/m:.4f}, {Fy/m:.4f}) m/s²")
    print(f"角加速度: alpha={tau/I:.4f} rad/s²")

    t_end = 3.0
    (t, x_n, y_n, vx_n, vy_n, th_n, w_n) = simulate(
        x0, y0, vx0, vy0, theta0, omega0,
        m, I, Fx, Fy, rx, ry, t_end=t_end)

    x_a, y_a, vx_a, vy_a, th_a, w_a = analytical(
        t, [x0, y0, vx0, vy0, theta0, omega0],
        m, I, Fx, Fy, rx, ry)

    err_x = np.max(np.abs(x_n - x_a))
    err_y = np.max(np.abs(y_n - y_a))
    err_th = np.max(np.abs(th_n - th_a))
    err_w = np.max(np.abs(w_n - w_a))

    E_num = np.array([mechanical_energy(
        [x_n[i], y_n[i], vx_n[i], vy_n[i], th_n[i], w_n[i]],
        m, I, Fx, Fy, rx, ry) for i in range(len(t))])

    print(f"时间点数      : {len(t)}")
    print(f"终止时间      : {t_end:.2f} s")
    print(f"最大 x 误差   : {err_x:.3e}")
    print(f"最大 y 误差   : {err_y:.3e}")
    print(f"最大 θ 误差   : {err_th:.3e}")
    print(f"最大 ω 误差   : {err_w:.3e}")
    print(f"机械能标准差   : {np.std(E_num):.3e}")
