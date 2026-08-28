"""MEC-012 —— 用 SciPy 数值求解受迫阻尼振子运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    natural_frequency, damping_ratio, \
    steady_state_amplitude, steady_state_phase, \
    resonance_frequency, mechanical_energy


def simulate(x0=1.0, v0=0.0, k=1.0, m=1.0, b=0.4,
             F0=1.0, omega=0.8,
             t_end=30.0, n_points=301):
    """数值积分受迫阻尼振子轨迹，返回 (t, x, v) 序列。

    参数
    ----
    x0, v0 : float
        初始位移和速度。
    k, m, b : float
        弹性系数、质量、阻尼系数。
    F0 : float
        驱动力幅值。
    omega : float
        驱动角频率。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, x, v) : (np.ndarray, np.ndarray, np.ndarray)
    """
    initial_state = np.array([x0, v0], dtype=float)
    validate_parameters(k=k, m=m, b=b, F0=F0, omega=omega)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(k, m, b, F0, omega),
        rtol=1e-9, atol=1e-12,
    )

    return sol.t, sol.y[0], sol.y[1]


if __name__ == "__main__":
    # 欠阻尼受迫振动示例
    k, m, b = 1.0, 1.0, 0.4
    F0, omega = 1.0, 0.8
    x0, v0 = 1.0, 0.0

    zeta = damping_ratio(k, m, b)
    omega0 = natural_frequency(k, m)
    A_ss = steady_state_amplitude(k, m, b, F0, omega)
    delta = steady_state_phase(k, m, b, omega)
    omega_r = resonance_frequency(k, m, b)

    print(f"弹性系数 k    : {k}")
    print(f"质量 m        : {m}")
    print(f"阻尼系数 b    : {b}")
    print(f"阻尼比 ζ      : {zeta:.4f}")
    print(f"固有角频率 ω₀  : {omega0:.6f}")
    print(f"驱动力 F0     : {F0}")
    print(f"驱动频率 ω    : {omega}")
    print(f"稳态振幅 A_ss  : {A_ss:.6f}")
    print(f"相位滞后 δ    : {delta:.6f}")
    print(f"共振频率 ω_max: {omega_r:.6f}" if omega_r else "共振频率: 无（ζ ≥ 1/√2）")

    t_end = 30.0  # 足够长以进入稳态
    t, x_num, v_num = simulate(x0=x0, v0=v0, k=k, m=m, b=b,
                                F0=F0, omega=omega, t_end=t_end)

    x_ana, v_ana = analytical(t, [x0, v0], k=k, m=m, b=b,
                              F0=F0, omega=omega)

    err_x = np.max(np.abs(x_num - x_ana))
    err_v = np.max(np.abs(v_num - v_ana))

    # 守恒量验证（机械能不守恒，仅展示）
    E_num = np.array([mechanical_energy([x, v], k=k, m=m)
                      for x, v in zip(x_num, v_num)])

    # 稳态段振幅验证（取最后几个周期）
    T_drive = 2.0 * np.pi / omega
    steady_start = t_end - 5 * T_drive
    mask = t >= steady_start
    if np.any(mask):
        x_steady = x_num[mask]
        A_steady_numerical = (np.max(x_steady) - np.min(x_steady)) / 2.0
    else:
        A_steady_numerical = None

    print(f"时间点数      : {len(t)}")
    print(f"终止时间      : {t_end:.2f} s")
    print(f"最大 x 误差   : {err_x:.3e}")
    print(f"最大 v 误差   : {err_v:.3e}")
    print(f"机械能初值    : {E_num[0]:.6f}")
    print(f"机械能末值    : {E_num[-1]:.6f}")
    if A_steady_numerical is not None:
        print(f"稳态振幅(数值): {A_steady_numerical:.6f}")
        print(f"稳态振幅(理论): {A_ss:.6f}")
        print(f"振幅误差      : {abs(A_steady_numerical - A_ss):.3e}")
