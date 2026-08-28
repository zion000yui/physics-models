"""MEC-013 —— 用 SciPy 数值求解双摆运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import dynamics, validate_parameters, mechanical_energy


def simulate(theta1_0=1.5, theta2_0=0.5, omega1_0=0.0, omega2_0=0.0,
             m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81,
             t_end=10.0, n_points=101):
    """数值积分双摆轨迹，返回 (t, theta1, theta2, omega1, omega2) 序列。

    参数
    ----
    theta1_0, theta2_0, omega1_0, omega2_0 : float
        初始角度和角速度。
    m1, m2, L1, L2, g : float
        物理参数。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, theta1, theta2, omega1, omega2) : 五元组
    """
    initial_state = np.array([theta1_0, theta2_0, omega1_0, omega2_0],
                              dtype=float)
    validate_parameters(m1=m1, m2=m2, L1=L1, L2=L2, g=g)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(m1, m2, L1, L2, g),
        rtol=1e-10, atol=1e-12,
    )

    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


if __name__ == "__main__":
    m1, m2, L1, L2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    theta1_0, theta2_0 = np.pi / 2, 0.0
    omega1_0, omega2_0 = 0.0, 0.0

    state0 = [theta1_0, theta2_0, omega1_0, omega2_0]
    E0 = mechanical_energy(state0, m1, m2, L1, L2, g)

    print(f"参数: m1={m1}, m2={m2}, L1={L1}, L2={L2}, g={g}")
    print(f"初始角度: θ1={theta1_0:.4f} ({np.degrees(theta1_0):.1f}°), "
          f"θ2={theta2_0:.4f} ({np.degrees(theta2_0):.1f}°)")
    print(f"初始机械能: {E0:.6f} J")

    t_end = 10.0
    t, t1_n, t2_n, w1_n, w2_n = simulate(
        theta1_0, theta2_0, omega1_0, omega2_0,
        m1, m2, L1, L2, g, t_end=t_end)

    E_num = np.array([mechanical_energy(
        [t1_n[i], t2_n[i], w1_n[i], w2_n[i]], m1, m2, L1, L2, g)
        for i in range(len(t))])

    print(f"时间点数       : {len(t)}")
    print(f"终止时间       : {t_end:.2f} s")
    print(f"机械能均值     : {np.mean(E_num):.6f}")
    print(f"机械能标准差   : {np.std(E_num):.3e}")
    print(f"机械能最大偏差 : {np.max(np.abs(E_num - E0)):.3e}")
