"""MEC-021 —— 用 SciPy 数值求解定轴转动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, angular_momentum


def simulate(theta0=1.0, omega0=0.0, I=1.0, tau=2.0,
             t_end=5.0, n_points=101):
    """数值积分定轴转动轨迹，返回 (t, theta, omega) 序列。

    参数
    ----
    theta0 : float
        初始角位移。
    omega0 : float
        初始角速度。
    I : float, optional
        转动惯量（默认 1.0 kg·m²）。
    tau : float, optional
        恒外力矩（默认 2.0 N·m）。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, theta, omega) : (np.ndarray, np.ndarray, np.ndarray)
    """
    initial_state = np.array([theta0, omega0], dtype=float)
    validate_parameters(I=I)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(I, tau),
        rtol=1e-9, atol=1e-12,
    )

    return sol.t, sol.y[0], sol.y[1]


if __name__ == "__main__":
    I, tau = 2.0, 3.0
    theta0, omega0 = 1.0, 0.5
    alpha = tau / I

    print(f"转动惯量 I    : {I}")
    print(f"外力矩 τ      : {tau}")
    print(f"角加速度 α    : {alpha:.6f} rad/s²")
    print(f"初始状态      : θ0={theta0}, ω0={omega0}")

    t_end = 5.0
    t, theta_n, omega_n = simulate(
        theta0=theta0, omega0=omega0, I=I, tau=tau, t_end=t_end)

    theta_a, omega_a = analytical(
        t, [theta0, omega0], I=I, tau=tau)

    err_theta = np.max(np.abs(theta_n - theta_a))
    err_omega = np.max(np.abs(omega_n - omega_a))

    # 角动量检查
    L_num = np.array([angular_momentum([th, om], I=I)
                      for th, om in zip(theta_n, omega_n)])

    print(f"时间点数      : {len(t)}")
    print(f"终止时间      : {t_end:.2f} s")
    print(f"最大 θ 误差   : {err_theta:.3e}")
    print(f"最大 ω 误差   : {err_omega:.3e}")
    print(f"角动量均值    : {np.mean(L_num):.6f}，标准差 {np.std(L_num):.3e}")
