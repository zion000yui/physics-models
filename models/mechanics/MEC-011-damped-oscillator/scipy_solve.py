"""MEC-011 —— 用 SciPy 数值求解阻尼振子运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    damping_ratio, natural_frequency, damped_frequency, mechanical_energy


def simulate(x0=1.0, v0=0.0, k=1.0, m=1.0, b=0.4,
             t_end=10.0, n_points=101):
    """数值积分阻尼振子轨迹，返回 (t, x, v) 序列。

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
    b : float, optional
        阻尼系数（默认 0.4，欠阻尼）。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, x, v) : (np.ndarray, np.ndarray, np.ndarray)
    """
    initial_state = np.array([x0, v0], dtype=float)
    validate_parameters(k=k, m=m, b=b)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(k, m, b),
        rtol=1e-9, atol=1e-12,
    )

    return sol.t, sol.y[0], sol.y[1]


if __name__ == "__main__":
    # 欠阻尼示例
    k, m, b = 1.0, 1.0, 0.4
    x0, v0 = 1.0, 0.0
    zeta = damping_ratio(k, m, b)
    omega0 = natural_frequency(k, m)

    print(f"弹性系数 k    : {k}")
    print(f"质量 m        : {m}")
    print(f"阻尼系数 b    : {b}")
    print(f"阻尼比 ζ      : {zeta:.4f}")
    print(f"固有角频率 ω₀  : {omega0:.6f}")
    if zeta < 1.0:
        omega_d = damped_frequency(k, m, b)
        print(f"阻尼角频率 ω_d : {omega_d:.6f}")
    print(f"状态          : {'欠阻尼' if zeta < 1 else ('临界阻尼' if zeta == 1 else '过阻尼')}")

    t_end = 10.0
    t, x_num, v_num = simulate(x0=x0, v0=v0, k=k, m=m, b=b, t_end=t_end)

    x_ana, v_ana = analytical(t, [x0, v0], k=k, m=m, b=b)

    err_x = np.max(np.abs(x_num - x_ana))
    err_v = np.max(np.abs(v_num - v_ana))

    E_num = np.array([mechanical_energy([x, v], k=k, m=m)
                      for x, v in zip(x_num, v_num)])

    print(f"时间点数      : {len(t)}")
    print(f"终止时间      : {t_end:.2f} s")
    print(f"初始状态      : [x={x_num[0]:.6f}, v={v_num[0]:.6f}]")
    print(f"末点数值 [x,v]: [{x_num[-1]:.6f}, {v_num[-1]:.6f}]")
    print(f"末点解析 [x,v]: [{x_ana[-1]:.6f}, {v_ana[-1]:.6f}]")
    print(f"最大 x 误差   : {err_x:.3e}")
    print(f"最大 v 误差   : {err_v:.3e}")
    print(f"机械能初值    : {E_num[0]:.6f}")
    print(f"机械能末值    : {E_num[-1]:.6f}")
    print(f"机械能耗散    : {E_num[0]-E_num[-1]:.6f} J")
