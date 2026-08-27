"""MEC-002 —— 一致性测试：数值解 vs 解析解。

运行方法（在本文件所在目录执行）：
    python test_consistency.py

通过则打印 "OK"，任一条失败会抛出 AssertionError。
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics

TOL = 1e-6  # 允许的最大误差


def _solve(x0, v0, t_end, n, F=1.0, m=1.0):
    """小工具：跑一次数值积分，返回 (t, x, v)。"""
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), [x0, v0], t_eval=t_eval, args=(F, m))
    return t_eval, sol.y[0], sol.y[1]


def test_x_matches_analytical():
    """位置数值解应与解析解一致。"""
    x0, v0, t_end, F, m = 1.0, -0.5, 5.0, 2.0, 0.5
    t, x_num, _ = _solve(x0, v0, t_end, 51, F=F, m=m)
    x_ana, _ = analytical(t, x0, v0, F, m)
    err = np.max(np.abs(x_num - x_ana))
    assert err < TOL, f"位置误差 {err:.3e} 超出容差 {TOL}"


def test_v_matches_analytical():
    """速度数值解应与解析解一致。"""
    x0, v0, t_end, F, m = 0.0, 1.0, 6.0, 3.0, 2.0
    t, _, v_num = _solve(x0, v0, t_end, 61, F=F, m=m)
    _, v_ana = analytical(t, x0, v0, F, m)
    err = np.max(np.abs(v_num - v_ana))
    assert err < TOL, f"速度误差 {err:.3e} 超出容差 {TOL}"


def test_acceleration_is_constant():
    """受恒定外力时加速度应恒为 F/m。"""
    x0, v0, t_end, F, m = 0.0, 0.0, 8.0, 2.0, 0.5
    t, _, v_num = _solve(x0, v0, t_end, 101, F=F, m=m)
    a = np.diff(v_num) / np.diff(t)
    a_expected = F / m
    assert np.allclose(a, a_expected, atol=TOL), f"加速度不恒为 {a_expected}"


if __name__ == "__main__":
    test_x_matches_analytical()
    test_v_matches_analytical()
    test_acceleration_is_constant()
    print("OK: MEC-002 数值解与解析解一致")
