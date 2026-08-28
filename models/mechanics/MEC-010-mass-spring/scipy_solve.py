"""MEC-010 —— 用 SciPy 数值求解质量—弹簧简谐振子运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    angular_frequency, period, amplitude, mechanical_energy


def simulate(x0=1.0, v0=0.0, k=1.0, m=1.0,
             t_end=6.28318530718, n_points=101):
    """数值积分简谐振动轨迹，返回 (t, x, v) 序列。

    参数
    ----
    x0 : float
        初始位移。
    v0 : float
        初始速度。
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
    (t, x, v) : (np.ndarray, np.ndarray, np.ndarray)
    """
    initial_state = np.array([x0, v0], dtype=float)
    validate_parameters(k=k, m=m)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(k, m),
        rtol=1e-9, atol=1e-12,
    )

    return sol.t, sol.y[0], sol.y[1]


if __name__ == "__main__":
    x0, v0, k, m = 1.0, 0.0, 1.0, 1.0
    omega0 = angular_frequency(k, m)
    T = period(k, m)
    A = amplitude(x0, v0, k, m)
    E0 = mechanical_energy([x0, v0], k, m)

    print(f"弹性系数 k    : {k}")
    print(f"质量 m        : {m}")
    print(f"角频率 ω₀     : {omega0:.6f} rad/s")
    print(f"周期 T        : {T:.6f} s")
    print(f"振幅 A        : {A:.6f} m")
    print(f"机械能 E      : {E0:.6f} J")

    t_end = T  # 一个完整周期
    t, x_num, v_num = simulate(x0=x0, v0=v0, k=k, m=m, t_end=t_end)

    # 解析解对照
    x_ana, v_ana = analytical(t, [x0, v0], k=k, m=m)

    err_x = np.max(np.abs(x_num - x_ana))
    err_v = np.max(np.abs(v_num - v_ana))

    # 守恒量验证
    E_num = np.array([mechanical_energy([x, v], k=k, m=m)
                      for x, v in zip(x_num, v_num)])

    print(f"时间点数      : {len(t)}")
    print(f"终止时间      : {t_end:.6f} s")
    print(f"初始状态      : [x={x_num[0]:.6f}, v={v_num[0]:.6f}]")
    print(f"末点数值 [x,v]: [{x_num[-1]:.6f}, {v_num[-1]:.6f}]")
    print(f"末点解析 [x,v]: [{x_ana[-1]:.6f}, {v_ana[-1]:.6f}]")
    print(f"最大 x 误差   : {err_x:.3e}")
    print(f"最大 v 误差   : {err_v:.3e}")
    print(f"机械能均值    : {np.mean(E_num):.6f}，标准差 {np.std(E_num):.3e}")
    print(f"周期后闭合误差: x={abs(x_num[-1]-x0):.3e}, v={abs(v_num[-1]-v0):.3e}")
