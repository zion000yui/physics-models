"""MEC-008 —— 用 SciPy 数值求解二体问题运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    total_momentum, total_angular_momentum, total_energy, \
    reduced_mass, gravitational_parameter, center_of_mass, \
    relative_orbital_elements


def simulate(x1=1.0, y1=0.0, vx1=0.0, vy1=0.3,
             x2=-1.0, y2=0.0, vx2=0.0, vy2=-0.3,
             G=1.0, m1=1.0, m2=1.0,
             t_end=5.0, n_points=101):
    """数值积分二体问题，返回 (t, x1, y1, vx1, vy1, x2, y2, vx2, vy2) 序列。

    参数
    ----
    x1, y1, vx1, vy1 : float
        质点 1 的初始位置和速度。
    x2, y2, vx2, vy2 : float
        质点 2 的初始位置和速度。
    G : float, optional
        引力常数（默认 1.0）。
    m1, m2 : float, optional
        两个质点的质量（默认 1.0）。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, x1, y1, vx1, vy1, x2, y2, vx2, vy2) : tuple of np.ndarray
    """
    initial_state = np.array([x1, y1, vx1, vy1, x2, y2, vx2, vy2],
                             dtype=float)
    validate_parameters(G=G, m1=m1, m2=m2)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(G, m1, m2),
        rtol=1e-9, atol=1e-12,
    )

    return (sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3],
            sol.y[4], sol.y[5], sol.y[6], sol.y[7])


if __name__ == "__main__":
    # 椭圆轨道示例：等质量，质心静止
    G, m1, m2 = 1.0, 1.0, 1.0
    x1, y1, vx1, vy1 = 1.0, 0.0, 0.0, 0.3
    x2, y2, vx2, vy2 = -1.0, 0.0, 0.0, -0.3

    state0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    elem = relative_orbital_elements(state0, G, m1, m2)
    mu = gravitational_parameter(G, m1, m2)
    mu_red = reduced_mass(m1, m2)
    T = 2.0 * np.pi / elem['n']

    print(f"引力参数 mu  : {mu:.6f} (= G*(m1+m2))")
    print(f"约化质量     : {mu_red:.6f}")
    print(f"轨道类型     : {elem['orbit_type']}")
    print(f"半长轴 a     : {elem['a']:.6f}")
    print(f"偏心率 e     : {elem['e']:.6f}")
    print(f"近心点幅角 ω : {elem['omega']:.6f}")
    print(f"平均运动 n   : {elem['n']:.6f} rad/s")
    print(f"轨道周期 T   : {T:.6f} s")

    t_end = T
    (t, x1_n, y1_n, vx1_n, vy1_n,
     x2_n, y2_n, vx2_n, vy2_n) = simulate(
        x1=x1, y1=y1, vx1=vx1, vy1=vy1,
        x2=x2, y2=y2, vx2=vx2, vy2=vy2,
        G=G, m1=m1, m2=m2, t_end=t_end)

    # 解析解对照
    (x1_a, y1_a, vx1_a, vy1_a,
     x2_a, y2_a, vx2_a, vy2_a) = analytical(
        t, state0, G=G, m1=m1, m2=m2)

    # 误差统计
    err_x1 = np.max(np.abs(x1_n - x1_a))
    err_y1 = np.max(np.abs(y1_n - y1_a))
    err_x2 = np.max(np.abs(x2_n - x2_a))
    err_y2 = np.max(np.abs(y2_n - y2_a))

    # 守恒量验证
    P_num = np.array([total_momentum(
        [x1_n[i], y1_n[i], vx1_n[i], vy1_n[i],
         x2_n[i], y2_n[i], vx2_n[i], vy2_n[i]], m1, m2)
        for i in range(len(t))])
    L_num = np.array([total_angular_momentum(
        [x1_n[i], y1_n[i], vx1_n[i], vy1_n[i],
         x2_n[i], y2_n[i], vx2_n[i], vy2_n[i]], m1, m2)
        for i in range(len(t))])
    E_num = np.array([total_energy(
        [x1_n[i], y1_n[i], vx1_n[i], vy1_n[i],
         x2_n[i], y2_n[i], vx2_n[i], vy2_n[i]], G, m1, m2)
        for i in range(len(t))])

    # 质心运动
    CM = np.array([center_of_mass(
        [x1_n[i], y1_n[i], vx1_n[i], vy1_n[i],
         x2_n[i], y2_n[i], vx2_n[i], vy2_n[i]], m1, m2)
        for i in range(len(t))])

    print(f"时间点数       : {len(t)}")
    print(f"终止时间       : {t_end:.6f} s")
    print(f"最大 x1 误差   : {err_x1:.3e}")
    print(f"最大 y1 误差   : {err_y1:.3e}")
    print(f"最大 x2 误差   : {err_x2:.3e}")
    print(f"最大 y2 误差   : {err_y2:.3e}")
    print(f"动量 Px 标准差  : {np.std(P_num[:, 0]):.3e}")
    print(f"动量 Py 标准差  : {np.std(P_num[:, 1]):.3e}")
    print(f"角动量标准差    : {np.std(L_num):.3e}")
    print(f"机械能标准差    : {np.std(E_num):.3e}")
    print(f"质心 X 标准差   : {np.std(CM[:, 0]):.3e}")
    print(f"质心 Y 标准差   : {np.std(CM[:, 1]):.3e}")
    print(f"质心 Vx 标准差  : {np.std(CM[:, 2]):.3e}")
    print(f"质心 Vy 标准差  : {np.std(CM[:, 3]):.3e}")
    print(f"周期后 x1 闭合  : {abs(x1_n[-1]-x1):.3e}")
    print(f"周期后 x2 闭合  : {abs(x2_n[-1]-x2):.3e}")
