"""MEC-024 —— 用 SciPy 数值求解纯滚动运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (analytical, dynamics, validate_parameters,
                   acceleration, effective_mass, mechanical_energy)


def simulate(x0=0.0, theta0=0.0, v0=0.0, omega0=0.0,
             m=1.0, I=1.0, R=1.0, g=9.81, alpha=0.5236,
             t_end=3.0, n_points=101):
    """数值积分纯滚动轨迹，返回 (t, x, theta, v, omega) 序列。

    参数
    ----
    x0, theta0, v0, omega0 : float
        初始位移、转角、速度、角速度。
    m, I, R, g, alpha : float
        物理参数。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, x, theta, v, omega) : 五元组
    """
    initial_state = np.array([x0, theta0, v0, omega0], dtype=float)
    validate_parameters(m=m, I=I, R=R, g=g, alpha=alpha)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(m, I, R, g, alpha),
        rtol=1e-9, atol=1e-12,
    )

    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


if __name__ == "__main__":
    g = 9.81
    alpha = np.radians(30)
    m, R = 1.0, 0.5
    I = 0.4 * m * R ** 2  # 实心球

    a = acceleration(g, m, I, R, alpha)
    m_eff = effective_mass(m, I, R)
    I_ratio = I / (m * R ** 2)

    print(f"参数: m={m}, I={I:.4f}, R={R}, g={g}, α={np.degrees(alpha):.1f}°")
    print(f"I/(mR²) = {I_ratio:.4f}（实心球 = 2/5）")
    print(f"有效质量 m_eff = {m_eff:.6f}")
    print(f"加速度 a = {a:.6f} m/s² = {a/(g*np.sin(alpha)):.4f}·g·sinα")
    print(f"理论: a = (5/7)·g·sinα = {5/7*g*np.sin(alpha):.6f}")

    t_end = 3.0
    t, x_n, th_n, v_n, w_n = simulate(
        m=m, I=I, R=R, g=g, alpha=alpha, t_end=t_end)

    x_a, th_a, v_a, w_a = analytical(
        t, [0, 0, 0, 0], m, I, R, g, alpha)

    err_x = np.max(np.abs(x_n - x_a))
    err_th = np.max(np.abs(th_n - th_a))
    err_v = np.max(np.abs(v_n - v_a))
    err_w = np.max(np.abs(w_n - w_a))

    # 纯滚动约束检查
    constraint_err = np.max(np.abs(v_n - R * w_n))

    # 机械能守恒检查
    E_num = np.array([mechanical_energy(
        [x_n[i], th_n[i], v_n[i], w_n[i]], m, I, R, g, alpha)
        for i in range(len(t))])

    print(f"时间点数       : {len(t)}")
    print(f"终止时间       : {t_end:.2f} s")
    print(f"最大 x 误差   : {err_x:.3e}")
    print(f"最大 θ 误差   : {err_th:.3e}")
    print(f"最大 v 误差    : {err_v:.3e}")
    print(f"最大 ω 误差   : {err_w:.3e}")
    print(f"约束 |v-Rω|    : {constraint_err:.3e}")
    print(f"机械能标准差   : {np.std(E_num):.3e}")
