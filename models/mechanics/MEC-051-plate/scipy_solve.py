"""MEC-051 —— 用 SciPy 求解 Kirchhoff-Love 薄板。

涵盖：
1. 静态弯曲：Navier 级数解
2. 动态模态动力学：solve_ivp
3. 有限差分固有频率：广义特征值问题

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

from model import (
    validate_parameters,
    plate_stiffness,
    static_simply_supported_navier,
    navier_load_coeff,
    max_deflection_simply_supported,
    natural_frequencies_plate,
    plate_mode_shape,
    plate_mode_laplacian,
    modal_dynamics_plate,
    modal_energy_plate,
    verify_plate_orthogonality,
    plate_modal_mass,
    plate_modal_stiffness,
    fd_plate_natural_frequencies,
    strain_energy_plate,
)


if __name__ == "__main__":
    # --- 参数 ---
    E = 2.0e11     # 钢
    h = 0.01       # 板厚 1cm
    nu = 0.3       # 泊松比
    rho = 7850.0
    a = 1.0        # 板长
    b = 1.0        # 板宽
    q = 1000.0     # 均布载荷 (Pa)

    validate_parameters(E=E, h=h, nu=nu, rho=rho, a=a, b=b, q_load=q)
    D = plate_stiffness(E, h, nu)

    print("=" * 60)
    print("MEC-051 Kirchhoff-Love Plate")
    print("=" * 60)
    print(f"参数: E={E:.2e} Pa, h={h} m, ν={nu}, ρ={rho} kg/m³")
    print(f"      a={a} m, b={b} m, q={q} Pa")
    print(f"板抗弯刚度 D={D:.4f} N·m")

    # --- 1. 静态 Navier 解 ---
    print(f"\n{'='*60}")
    print("1. 静态弯曲（简支板 Navier 解）")
    print(f"{'='*60}")

    x = np.linspace(0, a, 101)
    y = np.linspace(0, b, 101)
    w, Mx, My, Mxy = static_simply_supported_navier(x, y, q, a, b, E, h, nu,
                                                     n_terms=30)

    w_max = max_deflection_simply_supported(q, a, b, E, h, nu, n_terms=50)
    w_center = w[50, 50]  # 中心点
    print(f"中心挠度（Navier）: w(a/2, b/2) = {w_center:.6e} m = {w_center*1e3:.4f} mm")
    print(f"中心挠度（级数）:   w_max = {w_max:.6e} m = {w_max*1e3:.4f} mm")
    print(f"误差: {abs(w_center - w_max):.3e}")

    # 挠度系数 α: w_max = α q a⁴ / D
    alpha = w_max * D / (q * a**4)
    print(f"挠度系数 α = {alpha:.6f} (方板理论值≈0.00406)")

    # 边界条件
    print(f"\n边界 w: w(0,0)={w[0,0]:.2e}, w(a,0)={w[-1,0]:.2e}, "
          f"w(0,b)={w[0,-1]:.2e}, w(a,b)={w[-1,-1]:.2e}")
    print(f"边界 Mx: Mx(0,b/2)={Mx[0,50]:.4f}, Mx(a,b/2)={Mx[-1,50]:.4f}")
    print(f"边界 My: My(a/2,0)={My[50,0]:.4f}, My(a/2,b)={My[50,-1]:.4f}")

    # --- 2. 固有频率 ---
    print(f"\n{'='*60}")
    print("2. 固有频率")
    print(f"{'='*60}")

    omegas = natural_frequencies_plate(10, a, b, E, h, nu, rho)
    print(f"\n  {'i':>3s}  {'ω (rad/s)':>14s}  {'f (Hz)':>14s}")
    for i, (w_freq) in enumerate(omegas):
        print(f"  {i+1:3d}  {w_freq:14.4f}  {w_freq/(2*np.pi):14.4f}")

    # 前 5 阶应有解析公式
    # ω_mn = π²(m²/a² + n²/b²) √(D/(ρh))
    factor = np.sqrt(D / (rho * h))
    print(f"\n  验证（前 5 阶）:")
    expected = [
        (np.pi**2 * (1 + 1) * factor, 1, 1),
        (np.pi**2 * (4 + 1) * factor, 2, 1),
        (np.pi**2 * (1 + 4) * factor, 1, 2),
        (np.pi**2 * (4 + 4) * factor, 2, 2),
        (np.pi**2 * (9 + 1) * factor, 3, 1),
    ]
    for i, (exp_val, m, n) in enumerate(expected):
        print(f"    ({m},{n}): ω={omegas[i]:.4f}, 预期={exp_val:.4f}, "
              f"误差={abs(omegas[i]-exp_val)/exp_val*100:.4f}%")

    # --- 3. 有限差分 vs 解析 ---
    print(f"\n{'='*60}")
    print("3. 有限差分 vs 解析（简支方板）")
    print(f"{'='*60}")

    for N in [15, 21, 31]:
        omegas_fd = fd_plate_natural_frequencies(N, N, a, b, E, h, nu, rho,
                                                  n_modes=3)
        errs = np.abs(omegas_fd[:3] - omegas[:3]) / omegas[:3] * 100
        print(f"  N={N:2d}  误差: {errs}")

    # --- 4. 模态动力学 ---
    print(f"\n{'='*60}")
    print("4. 模态动力学（简支板自由振动）")
    print(f"{'='*60}")

    omegas_3 = natural_frequencies_plate(3, a, b, E, h, nu, rho)
    q0 = np.array([1e-4, 0, 0])
    qdot0 = np.array([0, 0, 0])
    t_end = 2 * np.pi / omegas_3[0] * 3

    sol = solve_ivp(
        modal_dynamics_plate, (0, t_end), np.concatenate([q0, qdot0]),
        args=(omegas_3, None),
        t_eval=np.linspace(0, t_end, 501),
        rtol=1e-10, atol=1e-12,
    )

    energies = np.array([modal_energy_plate(sol.y[:, i], omegas_3)
                         for i in range(sol.y.shape[1])])
    print(f"初始能量: {energies[0]:.6e} J")
    print(f"终止能量: {energies[-1]:.6e} J")
    print(f"能量变化: {abs(energies[-1]-energies[0])/energies[0]*100:.4f}%")

    # 频率验证
    q1 = sol.y[0, :]
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(q1)
    if len(peaks) >= 2:
        period_num = np.mean(np.diff(sol.t[peaks]))
        omega_num = 2 * np.pi / period_num
        print(f"模态 (1,1) 数值频率: {omega_num:.4f} rad/s (解析: {omegas_3[0]:.4f})")

    # --- 5. 模态正交性 ---
    print(f"\n{'='*60}")
    print("5. 模态正交性")
    print(f"{'='*60}")

    modes = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for i, (m1, n1) in enumerate(modes):
        for j, (m2, n2) in enumerate(modes[i:], i):
            result = verify_plate_orthogonality(m1, n1, m2, n2, rho, h, a, b)
            if (m1, n1) == (m2, n2):
                print(f"  <φ({m1},{n1}), φ({m2},{n2})> = {result:.6e} (模态质量)")
            else:
                print(f"  <φ({m1},{n1}), φ({m2},{n2})> = {result:.6e} (应≈0)")

    # --- 6. ω² = k/m ---
    print(f"\n{'='*60}")
    print("6. 模态质量/刚度验证 ω² = k_mn / m_mn")
    print(f"{'='*60}")

    omegas_full = natural_frequencies_plate(4, a, b, E, h, nu, rho)
    mode_pairs = [(1, 1), (2, 1), (1, 2), (2, 2)]
    for i, (m, n) in enumerate(mode_pairs):
        m_mn = plate_modal_mass(m, n, rho, h, a, b)
        k_mn = plate_modal_stiffness(m, n, D, a, b, nu)
        omega_sq = k_mn / m_mn
        print(f"  ({m},{n}): m={m_mn:.6e}, k={k_mn:.6e}, "
              f"sqrt(k/m)={np.sqrt(omega_sq):.4f}, ω_ana={omegas_full[i]:.4f}")

    print("\n=== 求解完成 ===")
