"""MEC-090 —— 用 SciPy 求解非线性力学。

涵盖：
1. 非线性单摆周期-振幅关系
2. 受驱阻尼摆（分岔前兆）
3. 庞加莱截面
4. Lyapunov 指数
5. 双摆轨迹发散

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk

from model import (
    validate_parameters,
    pendulum_dynamics,
    pendulum_linear_frequency,
    pendulum_period_analytical,
    pendulum_period_series,
    pendulum_energy,
    pendulum_energy_threshold,
    driven_damped_pendulum,
    solve_driven_pendulum,
    poincare_section,
    estimate_lyapunov_exponent,
    double_pendulum_divergence,
)


if __name__ == "__main__":
    g, l = 9.81, 1.0

    print("=" * 60)
    print("MEC-090 Nonlinear Mechanics")
    print("=" * 60)

    # --- 1. 非线性单摆周期 ---
    print(f"\n{'='*60}")
    print("1. 非线性单摆周期-振幅关系")
    print(f"{'='*60}")

    omega0 = pendulum_linear_frequency(g, l)
    T0 = 2 * np.pi / omega0
    print(f"  线性周期 T₀ = 2π√(l/g) = {T0:.6f} s")

    print(f"\n  {'θ₀':>8s}  {'T(椭圆)':>10s}  {'T(级数)':>10s}  {'T/T₀':>8s}")
    for deg in [1, 10, 30, 60, 90, 120, 150]:
        th0 = np.radians(deg)
        T_ell = pendulum_period_analytical(th0, g, l)
        T_ser = pendulum_period_series(th0, g, l)
        ratio = T_ell / T0
        print(f"  {deg:7d}°  {T_ell:10.6f}  {T_ser:10.6f}  {ratio:8.4f}")

    # --- 2. 能量阈值 ---
    print(f"\n  旋转阈值 E_c = 2mgl = {pendulum_energy_threshold(g, l):.4f} J")
    print(f"  E < E_c: 振荡运动")
    print(f"  E > E_c: 旋转运动")

    # --- 3. 受驱阻尼摆 ---
    print(f"\n{'='*60}")
    print("3. 受驱阻尼摆")
    print(f"{'='*60}")

    c, A, omega_d = 0.5, 1.2, 2.0 / 3.0
    print(f"  c={c}, A={A}, ω_d={omega_d:.4f}")

    sol = solve_driven_pendulum(g, l, c, A, omega_d,
                                  th0=0.2, w0=0.0, t_end=50.0, n=5001)
    print(f"  θ(t=0) = {sol.y[0, 0]:.4f}")
    print(f"  θ(t=50) = {sol.y[0, -1]:.4f}")

    # 庞加莱截面
    psec = poincare_section(sol, omega_d, t_start=10.0)
    print(f"  庞加莱截面点数: {len(psec)}")
    if len(psec) > 0:
        print(f"  θ 范围: [{psec[:, 0].min():.3f}, {psec[:, 0].max():.3f}]")

    # --- 4. Lyapunov 指数 ---
    print(f"\n{'='*60}")
    print("4. Lyapunov 指数")
    print(f"{'='*60}")

    # 小驱动（周期运动）
    lam_small = estimate_lyapunov_exponent(g, l, 0.5, 0.5, omega_d,
                                             t_end=100.0, n=10001)
    print(f"  小驱动 (A=0.5): λ ≈ {lam_small:.4f} (应≈0 或负, 周期运动)")

    # 大驱动（可能混沌）
    lam_large = estimate_lyapunov_exponent(g, l, 0.5, 1.2, omega_d,
                                            t_end=200.0, n=20001)
    print(f"  大驱动 (A=1.2): λ ≈ {lam_large:.4f}")

    # --- 5. 双摆轨迹发散 ---
    print(f"\n{'='*60}")
    print("5. 双摆轨迹发散（混沌特征）")
    print(f"{'='*60}")

    # 使用更大的初始角度引发混沌
    t_arr, d_arr = double_pendulum_divergence(th1=2.0, th2=2.0,
                                                t_end=15.0, n=15001)
    # 初始差距
    d0 = d_arr[1]
    # 后期差距
    d_end = d_arr[-1]
    print(f"  θ₁(0)=2.0, θ₂(0)=2.0 (大角度混沌区域)")
    print(f"  初始差距: {d0:.3e}")
    print(f"  t=5 时差距: {d_arr[5000]:.3e}")
    print(f"  终止差距: {d_end:.3e}")
    print(f"  发散倍数: {d_end / max(d0, 1e-20):.1f}")
    if d_end > 100 * d0:
        print(f"  → 轨迹显著发散（混沌特征）")
    else:
        print(f"  → 发散不显著")

    print("\n=== 求解完成 ===")
