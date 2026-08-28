"""MEC-052 —— 用 SciPy 求解圆柱壳力学。

涵盖：
1. 薄膜理论：内压下的薄膜应力
2. 弯曲理论：轴对称 Donnell 方程（BVP + 解析解）
3. 固有频率：解析 + 有限差分
4. 退化验证：R→∞ 退化为板

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.linalg import eigh

from model import (
    validate_parameters,
    bending_stiffness,
    membrane_stiffness,
    characteristic_length,
    decay_constant,
    membrane_forces_internal_pressure,
    membrane_stresses_internal_pressure,
    hoop_to_axial_ratio,
    axial_bending_ode_coefficients,
    axial_bending_analytical,
    axial_bending_max_deflection,
    natural_frequencies_cylindrical,
    membrane_frequency,
    bending_frequency_limit,
    modal_dynamics_shell,
    modal_energy_shell,
    degradation_to_plate_check,
    fd_shell_natural_frequencies,
)


if __name__ == "__main__":
    # --- 参数 ---
    E = 2.0e11
    h = 0.005    # 5mm 壁厚
    nu = 0.3
    rho = 7850.0
    R = 0.5      # 半径 0.5m
    L = 2.0      # 长度 2m
    p = 1.0e6    # 内压 1 MPa

    validate_parameters(E=E, h=h, nu=nu, rho=rho, R=R, L=L, p=p)
    D = bending_stiffness(E, h, nu)
    k = membrane_stiffness(E, h, R)
    lam = characteristic_length(E, h, nu, R)
    alpha = decay_constant(E, h, nu, R)

    print("=" * 60)
    print("MEC-052 Cylindrical Shell")
    print("=" * 60)
    print(f"参数: E={E:.2e} Pa, h={h} m, ν={nu}, ρ={rho} kg/m³")
    print(f"      R={R} m, L={L} m, p={p:.2e} Pa")
    print(f"抗弯刚度 D={D:.4f} N·m")
    print(f"薄膜刚度 k=Eh/R²={k:.4f} N/m²")
    print(f"衰减长度 λ={lam:.4f} m")
    print(f"衰减常数 α={alpha:.4f} 1/m")
    print(f"L/λ={L/lam:.2f} (应>>1 长壳近似有效)")

    # --- 1. 薄膜理论 ---
    print(f"\n{'='*60}")
    print("1. 薄膜理论（内压容器）")
    print(f"{'='*60}")

    N_x, N_theta = membrane_forces_internal_pressure(p, R)
    sigma_x, sigma_theta = membrane_stresses_internal_pressure(p, R, h)
    print(f"环向薄膜力: N_θ = {N_theta:.2f} N/m")
    print(f"轴向薄膜力: N_x = {N_x:.2f} N/m")
    print(f"环向应力: σ_θ = {sigma_theta:.2e} Pa = {sigma_theta/1e6:.2f} MPa")
    print(f"轴向应力: σ_x = {sigma_x:.2e} Pa = {sigma_x/1e6:.2f} MPa")
    print(f"应力比 σ_θ/σ_x = {sigma_theta/sigma_x:.1f} (理论值: {hoop_to_axial_ratio()})")

    # --- 2. 弯曲理论 ---
    print(f"\n{'='*60}")
    print("2. 弯曲理论（轴对称 Donnell 方程）")
    print(f"{'='*60}")

    w_max = axial_bending_max_deflection(p, E, h, R)
    print(f"最大挠度（远端）: w_max = pR²/(Eh) = {w_max:.6e} m = {w_max*1e3:.4f} mm")

    x = np.linspace(0, 5 * lam, 200)
    w_canti = axial_bending_analytical(x, p, E, h, nu, R, L, bc='long_cantilever')
    w_ss = axial_bending_analytical(x, p, E, h, nu, R, L, bc='long_simply_supported')

    print(f"悬臂端部挠度: w(0) = {w_canti[0]:.6e} (应≈0)")
    print(f"铰支端部挠度: w(0) = {w_ss[0]:.6e} (应≈0)")
    print(f"远端挠度: w(L>>λ) = {w_canti[-1]:.6e} (应≈{w_max:.6e})")

    # --- 3. BVP 求解 ---
    print(f"\n{'='*60}")
    print("3. BVP 求解轴对称弯曲")
    print(f"{'='*60}")

    D, k = axial_bending_ode_coefficients(E, h, nu, R)
    x_bvp = np.linspace(0, L, 101)
    y_init = np.zeros((4, len(x_bvp)))
    y_init[0] = w_max * (1 - np.exp(-alpha * x_bvp) * np.cos(alpha * x_bvp))

    def ode(x, y):
        """D w'''' + k w = p → 4 个 1 阶 ODE。"""
        dy = np.zeros_like(y)
        dy[0] = y[1]
        dy[1] = y[2]
        dy[2] = y[3]
        dy[3] = (p - k * y[0]) / D * np.ones_like(x)
        return dy

    def bc_cantilever(ya, yb):
        """固定端 x=0: w=0, w'=0; 自由端 x=L: M=0(w''=0), V=0(w'''=0)。"""
        return np.array([ya[0], ya[1], yb[2], yb[3]])

    sol = solve_bvp(ode, bc_cantilever, x_bvp, y_init, tol=1e-8, max_nodes=10000)
    if sol.success:
        w_bvp = sol.sol(x_bvp)[0]
        w_ana = axial_bending_analytical(x_bvp, p, E, h, nu, R, L, bc='long_cantilever')
        err = np.max(np.abs(w_bvp - w_ana))
        print(f"BVP 求解成功，与解析解误差: {err:.3e} m")
    else:
        print(f"BVP 求解失败: {sol.message}")

    # --- 4. 固有频率 ---
    print(f"\n{'='*60}")
    print("4. 固有频率")
    print(f"{'='*60}")

    omegas = natural_frequencies_cylindrical(5, E, h, nu, rho, R, L)
    omega_mem = membrane_frequency(E, rho, R)
    print(f"\n  {'n':>3s}  {'ω (rad/s)':>14s}  {'f (Hz)':>14s}  {'弯曲极限':>14s}")
    for i, omega in enumerate(omegas):
        omega_bend = bending_frequency_limit(i + 1, E, h, nu, rho, L)
        print(f"  {i+1:3d}  {omega:14.4f}  {omega/(2*np.pi):14.4f}  {omega_bend:14.4f}")
    print(f"\n  薄膜频率极限: ω_0 = {omega_mem:.4f} rad/s")
    print(f"  高 n 时 ω_n → 弯曲极限")

    # --- 5. 有限差分 ---
    print(f"\n{'='*60}")
    print("5. 有限差分 vs 解析（轴对称固有频率）")
    print(f"{'='*60}")

    for N in [31, 51, 101]:
        omegas_fd = fd_shell_natural_frequencies(N, L, E, h, nu, rho, R, n_modes=3)
        errs = np.abs(omegas_fd[:3] - omegas[:3]) / omegas[:3] * 100
        print(f"  N={N:3d}  误差: {errs}")

    # --- 6. 模态动力学 ---
    print(f"\n{'='*60}")
    print("6. 模态动力学（自由振动）")
    print(f"{'='*60}")

    omegas_3 = natural_frequencies_cylindrical(3, E, h, nu, rho, R, L)
    q0 = np.array([1e-4, 0, 0])
    qdot0 = np.array([0, 0, 0])
    t_end = 2 * np.pi / omegas_3[0] * 3

    sol_dyn = solve_ivp(
        modal_dynamics_shell, (0, t_end), np.concatenate([q0, qdot0]),
        args=(omegas_3, None),
        t_eval=np.linspace(0, t_end, 501),
        rtol=1e-10, atol=1e-12,
    )
    energies = np.array([modal_energy_shell(sol_dyn.y[:, i], omegas_3)
                         for i in range(sol_dyn.y.shape[1])])
    print(f"初始能量: {energies[0]:.6e} J")
    print(f"终止能量: {energies[-1]:.6e} J")
    print(f"能量变化: {abs(energies[-1]-energies[0])/energies[0]*100:.4f}%")

    # --- 7. 退化验证 ---
    print(f"\n{'='*60}")
    print("7. 退化验证：R→∞ 退化为板")
    print(f"{'='*60}")

    omega_shell, omega_plate, err = degradation_to_plate_check(E, h, nu, rho, L)
    print(f"壳频率（R→∞）: {omega_shell:.4f} rad/s")
    print(f"板频率:        {omega_plate:.4f} rad/s")
    print(f"误差: {err:.3e}")

    print("\n=== 求解完成 ===")
