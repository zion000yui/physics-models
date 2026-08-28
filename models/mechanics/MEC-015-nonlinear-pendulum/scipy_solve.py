"""MEC-015 —— 用 SciPy 数值求解非线性单摆运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import (analytical, dynamics, validate_parameters,
                   natural_frequency, small_angle_period, nonlinear_period,
                   mechanical_energy)


def simulate(theta0=1.0, omega0=0.0, g=9.81, L=1.0, m=1.0,
             t_end=5.0, n_points=101):
    """数值积分非线性单摆轨迹，返回 (t, theta, omega) 序列。

    参数
    ----
    theta0 : float
        初始角度（rad）。
    omega0 : float
        初始角速度（rad/s）。
    g, L, m : float
        重力加速度、摆长、质量。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, theta, omega) : (np.ndarray, np.ndarray, np.ndarray)
    """
    initial_state = np.array([theta0, omega0], dtype=float)
    validate_parameters(g=g, L=L, m=m)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(g, L, m),
        rtol=1e-9, atol=1e-12,
    )

    return sol.t, sol.y[0], sol.y[1]


if __name__ == "__main__":
    g, L, m = 9.81, 1.0, 1.0
    omega_0 = natural_frequency(g, L)
    T0 = small_angle_period(g, L)

    print(f"参数: g={g}, L={L}, m={m}")
    print(f"小角度固有角频率 ω₀ : {omega_0:.6f} rad/s")
    print(f"小角度周期 T₀       : {T0:.6f} s")

    # 大角度非线性摆动
    theta0 = 1.5  # ~86°
    T_nonlin = nonlinear_period(g, L, theta0)
    print(f"大角度振幅 θ_max     : {theta0} rad ({np.degrees(theta0):.1f}°)")
    print(f"非线性周期 T        : {T_nonlin:.6f} s (T/T₀={T_nonlin/T0:.4f})")

    t_end = T_nonlin
    t, theta_num, omega_num = simulate(
        theta0=theta0, omega0=0.0, g=g, L=L, m=m, t_end=t_end)

    # 小角度解析解对照（仅小角度时有效）
    theta_small, omega_small = analytical(
        t, [theta0, 0.0], g=g, L=L, m=m)

    err_small = np.max(np.abs(theta_num - theta_small))

    # 机械能守恒
    E_num = np.array([mechanical_energy([th, om], g=g, L=L, m=m)
                      for th, om in zip(theta_num, omega_num)])

    print(f"时间点数             : {len(t)}")
    print(f"终止时间             : {t_end:.6f} s")
    print(f"初始角度             : {theta_num[0]:.6f} rad")
    print(f"末点角度             : {theta_num[-1]:.6f} rad")
    print(f"小角度解误差（大角度）: {err_small:.3e}（预期不匹配）")
    print(f"机械能均值           : {np.mean(E_num):.6f}，标准差 {np.std(E_num):.3e}")
    print(f"周期后角度闭合误差   : {abs(theta_num[-1]-theta0):.3e}")
