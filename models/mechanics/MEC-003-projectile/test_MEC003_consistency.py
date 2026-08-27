"""MEC-003 —— 一致性测试：数值解 vs 解析解。

验证：
- 数值结果与解析解误差
- 水平方向速度守恒
- 机械能守恒（m=1）
- 边界条件正确

运行方法（在本文件所在目录执行）：
    python test_MEC003_consistency.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from model import analytical, dynamics

TOL = 1e-6


def _solve(x0=0.0, y0=10.0, vx0=10.0, vy0=15.0, g=9.81,
           t_end=5.0, n=101):
    """小工具：跑一次数值积分，返回 (t, x, y, vx, vy)。"""
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end),
                    [x0, y0, vx0, vy0], t_eval=t_eval, args=(g,))
    return t_eval, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def test_position_matches_analytical():
    """位置数值解应与解析解一致。"""
    x0, y0, vx0, vy0, g = 0.0, 10.0, 10.0, 15.0, 9.81
    t_end, n = 4.0, 81
    t, x_num, y_num, vx_num, vy_num = _solve(x0, y0, vx0, vy0, g, t_end, n)
    x_ana, y_ana, vx_ana, vy_ana = analytical(t, [x0, y0, vx0, vy0], g=g)
    err_x = np.max(np.abs(x_num - x_ana))
    err_y = np.max(np.abs(y_num - y_ana))
    assert err_x < TOL, f"x 误差 {err_x:.3e} 超出容差 {TOL}"
    assert err_y < TOL, f"y 误差 {err_y:.3e} 超出容差 {TOL}"


def test_horizontal_velocity_constant():
    """水平方向速度应恒等于初速度（无空气阻力）。"""
    x0, y0, vx0, vy0, g = 0.0, 5.0, 12.0, 8.0, 9.81
    t, _, _, vx_num, _ = _solve(x0, y0, vx0, vy0, g, t_end=6.0, n=121)
    assert np.allclose(vx_num, vx0, atol=TOL), "水平速度不守恒"


def test_mechanical_energy_conserved():
    """机械能 E = 0.5*(vx^2 + vy^2) + g*y（m=1）应保持恒定。"""
    x0, y0, vx0, vy0, g = 2.0, 10.0, 5.0, 20.0, 9.81
    t, x_num, y_num, vx_num, vy_num = _solve(x0, y0, vx0, vy0, g,
                                               t_end=4.5, n=91)
    E_num = 0.5 * (vx_num ** 2 + vy_num ** 2) + g * y_num
    assert np.allclose(E_num, E_num[0], atol=TOL), \
        f"机械能不守恒，波动 {np.max(np.abs(E_num - E_num[0])):.3e}"


def test_landing_time():
    """从 y0>0 抛出后，落地时刻的数值解应与解析值一致。"""
    x0, y0, vx0, vy0, g = 0.0, 10.0, 15.0, 10.0, 9.81
    t_end = 5.0

    # 解析落地时间：解 y0 + vy0*t - 0.5*g*t^2 = 0 的正根
    t_land_ana = brentq(lambda t: y0 + vy0 * t - 0.5 * g * t ** 2,
                        0.0, t_end)

    # 数值落地时间：找 y 穿越 0 的时刻（提高分辨率以保证插值精度）
    t, _, y_num, _, _ = _solve(x0, y0, vx0, vy0, g, t_end=t_end, n=2001)
    # 寻找最后一个非负 y 的索引
    above = np.where(y_num >= 0)[0]
    if len(above) == 0:
        raise AssertionError("数值解未落地")
    i = above[-1]
    if i < len(t) - 1:
        # 线性插值求落地时刻
        t_land_num = t[i] + (t[i + 1] - t[i]) * (-y_num[i]) / (
            y_num[i + 1] - y_num[i])
    else:
        t_land_num = t[i]

    err = abs(t_land_num - t_land_ana)
    assert err < TOL, f"落地时间误差 {err:.3e} 超出容差 {TOL}"


if __name__ == "__main__":
    test_position_matches_analytical()
    test_horizontal_velocity_constant()
    test_mechanical_energy_conserved()
    test_landing_time()
    print("OK: MEC-003 数值解与解析解一致")
