"""MEC-014 —— 用 SciPy 数值求解耦合振子运动轨迹。

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics, validate_parameters, \
    normal_modes, mechanical_energy


def simulate(x1_0=1.0, x2_0=0.0, v1_0=0.0, v2_0=0.0,
             m1=1.0, m2=1.0, k1=1.0, k2=1.0, kc=0.5,
             t_end=10.0, n_points=101):
    """数值积分耦合振子轨迹，返回 (t, x1, x2, v1, v2) 序列。

    参数
    ----
    x1_0, x2_0, v1_0, v2_0 : float
        初始位移和速度。
    m1, m2, k1, k2, kc : float
        物理参数。
    t_end : float
        仿真终止时间。
    n_points : int
        输出的时间点数。

    返回
    ----
    (t, x1, x2, v1, v2) : 五元组
    """
    initial_state = np.array([x1_0, x2_0, v1_0, v2_0], dtype=float)
    validate_parameters(m1=m1, m2=m2, k1=k1, k2=k2, kc=kc)

    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        dynamics,
        (0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        args=(m1, m2, k1, k2, kc),
        rtol=1e-9, atol=1e-12,
    )

    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


if __name__ == "__main__":
    m1, m2, k1, k2, kc = 1.0, 1.0, 1.0, 1.0, 0.5
    x1_0, x2_0, v1_0, v2_0 = 1.0, 0.0, 0.0, 0.0

    modes = normal_modes(m1, m2, k1, k2, kc)
    print(f"参数: m1={m1}, m2={m2}, k1={k1}, k2={k2}, kc={kc}")
    print(f"模态 1: ω={modes[0]['omega']:.6f}, mode={modes[0]['mode']}")
    print(f"模态 2: ω={modes[1]['omega']:.6f}, mode={modes[1]['mode']}")
    print(f"初始状态: x1={x1_0}, x2={x2_0}, v1={v1_0}, v2={v2_0}")

    t_end = 10.0
    t, x1_n, x2_n, v1_n, v2_n = simulate(
        x1_0, x2_0, v1_0, v2_0, m1, m2, k1, k2, kc, t_end=t_end)

    x1_a, x2_a, v1_a, v2_a = analytical(
        t, [x1_0, x2_0, v1_0, v2_0], m1, m2, k1, k2, kc)

    err_x1 = np.max(np.abs(x1_n - x1_a))
    err_x2 = np.max(np.abs(x2_n - x2_a))
    err_v1 = np.max(np.abs(v1_n - v1_a))
    err_v2 = np.max(np.abs(v2_n - v2_a))

    E_num = np.array([mechanical_energy(
        [x1_n[i], x2_n[i], v1_n[i], v2_n[i]], m1, m2, k1, k2, kc)
        for i in range(len(t))])

    print(f"时间点数       : {len(t)}")
    print(f"终止时间       : {t_end:.2f} s")
    print(f"最大 x1 误差   : {err_x1:.3e}")
    print(f"最大 x2 误差   : {err_x2:.3e}")
    print(f"最大 v1 误差   : {err_v1:.3e}")
    print(f"最大 v2 误差   : {err_v2:.3e}")
    print(f"机械能均值     : {np.mean(E_num):.6f}，标准差 {np.std(E_num):.3e}")
