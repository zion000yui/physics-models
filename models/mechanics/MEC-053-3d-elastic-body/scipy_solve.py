"""MEC-053 —— 用 SciPy 求解三维弹性体。

涵盖：
1. 弹性常数换算验证
2. 三种经典均匀应力状态
3. 弹性波速
4. 退化与不可压缩极限

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np

from model import (
    validate_parameters,
    shear_modulus,
    bulk_modulus,
    lame_first,
    lame_second,
    all_elastic_constants,
    inverse_from_GK,
    compliance_matrix,
    stiffness_matrix,
    stress_from_strain,
    strain_from_stress,
    uniaxial_tension,
    hydrostatic_compression,
    pure_shear,
    volumetric_strain,
    mean_stress,
    deviatoric_stress,
    strain_energy_density,
    p_wave_speed,
    s_wave_speed,
    wave_speed_ratio,
    degradation_uncoupled,
    incompressibility_limit,
    check_incompressibility,
)


if __name__ == "__main__":
    E = 2.0e11
    nu = 0.3
    rho = 7850.0

    validate_parameters(E=E, nu=nu, rho=rho)

    print("=" * 60)
    print("MEC-053 3D Elastic Body")
    print("=" * 60)
    print(f"参数: E={E:.2e} Pa, ν={nu}, ρ={rho} kg/m³")

    # --- 1. 弹性常数 ---
    print(f"\n{'='*60}")
    print("1. 弹性常数换算")
    print(f"{'='*60}")

    G, K, lam, mu = all_elastic_constants(E, nu)
    print(f"  剪切模量 G = E/[2(1+ν)] = {G:.4e} Pa = {G/1e9:.4f} GPa")
    print(f"  体积模量 K = E/[3(1-2ν)] = {K:.4e} Pa = {K/1e9:.4f} GPa")
    print(f"  Lamé 第一参数 λ = {lam:.4e} Pa = {lam/1e9:.4f} GPa")
    print(f"  Lamé 第二参数 μ = G = {mu:.4e} Pa = {mu/1e9:.4f} GPa")

    # 逆推
    E_back, nu_back = inverse_from_GK(G, K)
    print(f"\n  从 G,K 逆推: E={E_back:.4e} (误差 {abs(E_back-E)/E:.3e})")
    print(f"                ν={nu_back:.6f} (误差 {abs(nu_back-nu)/nu:.3e})")

    # --- 2. 本构矩阵 ---
    print(f"\n{'='*60}")
    print("2. 本构矩阵")
    print(f"{'='*60}")

    C = stiffness_matrix(E, nu)
    S = compliance_matrix(E, nu)
    print("  刚度矩阵 C (GPa):")
    print(f"  {C/1e9}")
    print(f"\n  柔度矩阵 S (×10⁻¹² Pa⁻¹):")
    print(f"  {S*1e12}")

    # 验证 C @ S = I
    I_check = C @ S
    err = np.max(np.abs(I_check - np.eye(6)))
    print(f"\n  C @ S = I 误差: {err:.3e}")

    # --- 3. 经典应力状态 ---
    print(f"\n{'='*60}")
    print("3. 经典均匀应力状态")
    print(f"{'='*60}")

    sigma0 = 100e6  # 100 MPa

    # 单轴拉伸
    eps_uni, U_uni = uniaxial_tension(sigma0, E, nu)
    print(f"\n  单轴拉伸 σ={sigma0/1e6:.0f} MPa:")
    print(f"    ε_xx = {eps_uni[0]:.6e} (应 = σ/E = {sigma0/E:.6e})")
    print(f"    ε_yy = ε_zz = {eps_uni[1]:.6e} (应 = -νσ/E = {-nu*sigma0/E:.6e})")
    print(f"    θ = {volumetric_strain(eps_uni):.6e} (应 = σ(1-2ν)/E)")
    print(f"    U = {U_uni:.4e} J/m³ (应 = σ²/(2E) = {sigma0**2/(2*E):.4e})")

    # 静水压缩
    p = 100e6
    eps_hyd, U_hyd = hydrostatic_compression(p, E, nu)
    theta_hyd = volumetric_strain(eps_hyd)
    print(f"\n  静水压缩 p={p/1e6:.0f} MPa:")
    print(f"    ε_xx = ε_yy = ε_zz = {eps_hyd[0]:.6e}")
    print(f"    θ = {theta_hyd:.6e} (应 = -p/K = {-p/K:.6e})")
    print(f"    U = {U_hyd:.4e} J/m³ (应 = p²/(2K) = {p**2/(2*K):.4e})")

    # 纯剪切
    tau = 50e6
    eps_shr, U_shr = pure_shear(tau, E, nu)
    print(f"\n  纯剪切 τ={tau/1e6:.0f} MPa:")
    print(f"    γ_xy = {eps_shr[5]:.6e} (应 = τ/G = {tau/G:.6e})")
    print(f"    U = {U_shr:.4e} J/m³ (应 = τ²/(2G) = {tau**2/(2*G):.4e})")

    # --- 4. 弹性波速 ---
    print(f"\n{'='*60}")
    print("4. 弹性波速")
    print(f"{'='*60}")

    c_p = p_wave_speed(E, nu, rho)
    c_s = s_wave_speed(E, nu, rho)
    ratio = wave_speed_ratio(nu)
    print(f"  纵波速度 c_p = {c_p:.2f} m/s")
    print(f"  横波速度 c_s = {c_s:.2f} m/s")
    print(f"  波速比 c_p/c_s = {c_p/c_s:.4f} (理论: {ratio:.4f})")
    print(f"  c_p > c_s: {c_p > c_s} (P 波始终快于 S 波)")
    print(f"  c_p/c_s > √2: {c_p/c_s > np.sqrt(2):.4f} (不可压缩极限 ν→0.5 时 → √2)")

    # --- 5. 退化与极限 ---
    print(f"\n{'='*60}")
    print("5. 退化与极限行为")
    print(f"{'='*60}")

    # ν → 0
    s01 = degradation_uncoupled(E, nu_zero=0.0)
    print(f"  ν→0: S[0,1] = {s01:.3e} (应≈0, 各方向独立)")

    # ν → 0.5 (不可压缩)
    nus, Ks = incompressibility_limit()
    print(f"\n  ν→0.5 不可压缩极限:")
    for n, K in zip(nus, Ks):
        print(f"    ν={n:.4f}: K={K:.4e} Pa ({'→∞' if K > 1e15 else ''})")

    # 体积应变趋近零
    err_inc = check_incompressibility(0.4999)
    print(f"\n  ν=0.4999 时体积应变误差: {err_inc:.3e} (应趋近零)")

    # --- 6. 与已有模型的对应 ---
    print(f"\n{'='*60}")
    print("6. 与已有 MEC 模型的对应")
    print(f"{'='*60}")

    # 单轴拉伸 → 弹簧 (MEC-010): F = EA/L · Δx, k = EA/L
    A_cross = 1e-4  # 1cm²
    L_bar = 1.0
    k_spring = E * A_cross / L_bar
    eps_uni_val, _ = uniaxial_tension(sigma0, E, nu)
    delta = eps_uni_val[0] * L_bar
    F = sigma0 * A_cross
    print(f"  单轴拉伸 → 弹簧: k=EA/L={k_spring:.2f} N/m")
    print(f"    σ={sigma0/1e6:.0f} MPa → Δx=σε·L={delta*1e3:.4f} mm")
    print(f"    F=σA={F:.2f} N, k·Δx={k_spring*delta:.2f} N (应一致)")

    # 板抗弯刚度: D = Eh³/[12(1-ν²)]
    h_plate = 0.01
    D_plate = E * h_plate**3 / (12 * (1 - nu**2))
    print(f"\n  板抗弯刚度 D = Eh³/[12(1-ν²)] = {D_plate:.4f} N·m")
    print(f"    (MEC-051 的 D 依赖 ν, 1D 梁 EI = E·bh³/12 不含 ν)")

    print("\n=== 求解完成 ===")
