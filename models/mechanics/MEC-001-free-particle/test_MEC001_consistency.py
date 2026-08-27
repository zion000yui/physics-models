"""MEC-001 —— 一致性测试：数值解 vs 解析解。

运行方法（在本文件所在目录执行）：
    python test_consistency.py

通过则打印 "OK"，任一条失败会抛出 AssertionError。
"""

import numpy as np
from scipy.integrate import solve_ivp

from model import analytical, dynamics

TOL = 1e-6  # 允许的最大误差


def _solve(x0, v0, t_end, n):
    """小工具：跑一次数值积分，返回 (t, x, v)。"""
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(dynamics, (0.0, t_end), [x0, v0], t_eval=t_eval)
    return t_eval, sol.y[0], sol.y[1]


def test_x_matches_analytical():
    """位置数值解应与解析解一致。"""
    x0, v0, t_end = 1.0, -0.5, 5.0
    t, x_num, _ = _solve(x0, v0, t_end, 51)
    x_ana, _ = analytical(t, x0, v0)
    err = np.max(np.abs(x_num - x_ana))
    assert err < TOL, f"位置误差 {err:.3e} 超出容差 {TOL}"


def test_v_is_constant():
    """无外力时速度应恒等于初速度（动能守恒）。"""
    x0, v0, t_end = 0.0, 3.0, 8.0
    _, _, v_num = _solve(x0, v0, t_end, 101)
    assert np.allclose(v_num, v0, atol=TOL), "速度不守恒"


if __name__ == "__main__":
    test_x_matches_analytical()
    test_v_is_constant()
    print("OK: MEC-001 数值解与解析解一致")