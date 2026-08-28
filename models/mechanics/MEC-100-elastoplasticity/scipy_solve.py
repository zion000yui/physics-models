"""MEC-100 —— 用 SciPy 求解弹塑性力学。

涵盖：
1. 单轴弹塑性应力-应变关系
2. 屈服准则（Tresca / von Mises）
3. 弹塑性能量
4. 卸载与残余应变
5. 退化到纯弹性

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np

from model import (
    validate_parameters,
    tresca_yield,
    tresca_check,
    von_mises_stress,
    von_mises_check,
    von_mises_from_deviator,
    stress_deviator,
    uniaxial_stress_strain,
    yield_strain,
    plastic_strain,
    elastic_energy,
    plastic_dissipation,
    total_energy,
    unload_stress,
    bauschinger_effect,
    degradation_to_elastic,
    check_pure_elastic_when_below_yield,
)


if __name__ == "__main__":
    E = 2.0e11
    nu = 0.3
    sigma_y = 250e6  # 250 MPa
    H = 5e9         # 线性硬化模量 5 GPa

    validate_parameters(E=E, nu=nu, sigma_y=sigma_y, H=H)

    print("=" * 60)
    print("MEC-100 Elastoplasticity")
    print("=" * 60)
    print(f"参数: E={E:.2e} Pa, ν={nu}, σ_y={sigma_y/1e6:.0f} MPa, H={H/1e9:.1f} GPa")

    # --- 1. 单轴应力-应变 ---
    print(f"\n{'='*60}")
    print("1. 单轴弹塑性应力-应变")
    print(f"{'='*60}")

    eps_y = yield_strain(E, sigma_y)
    print(f"  屈服应变: ε_y = σ_y/E = {eps_y:.6e}")

    eps = np.linspace(0, 5 * eps_y, 100)
    sigma, is_pl = uniaxial_stress_strain(eps, E, sigma_y, H)

    # 弹性段
    sigma_elastic = E * eps
    err_elastic = np.max(np.abs(sigma[~is_pl] - sigma_elastic[~is_pl]))
    print(f"  弹性段最大误差: {err_elastic:.3e}")

    # 塑性段
    idx_pl = np.where(is_pl)[0]
    if len(idx_pl) > 0:
        i = idx_pl[-1]
        eps_p = eps[i] - eps_y
        sigma_expected = sigma_y + H * eps_p
        print(f"  塑性段末点: σ={sigma[i]/1e6:.2f} MPa, "
              f"预期={sigma_expected/1e6:.2f} MPa")

    # 理想塑性
    sigma_ideal, _ = uniaxial_stress_strain(eps, E, sigma_y, H=0.0)
    pl_idx = np.where(np.abs(eps) > eps_y)[0]
    if len(pl_idx) > 0:
        print(f"  理想塑性末点: σ={sigma_ideal[pl_idx[-1]]/1e6:.2f} MPa "
              f"(应={sigma_y/1e6:.2f} MPa)")

    # --- 2. 屈服准则 ---
    print(f"\n{'='*60}")
    print("2. 屈服准则")
    print(f"{'='*60}")

    # 单轴应力状态
    sigma_v = np.array([100e6, 0, 0, 0, 0, 0])
    vm = von_mises_stress(sigma_v)
    tr = tresca_yield(sigma_v)
    print(f"  单轴 σ=100 MPa:")
    print(f"    von Mises: {vm/1e6:.2f} MPa")
    print(f"    Tresca:    {tr/1e6:.2f} MPa")
    print(f"    屈服(σ_y=250): VM={von_mises_check(sigma_v, sigma_y)}, "
          f"Tr={tresca_check(sigma_v, sigma_y)}")

    # 静水压缩
    sigma_h = np.array([-100e6, -100e6, -100e6, 0, 0, 0])
    vm_h = von_mises_stress(sigma_h)
    tr_h = tresca_yield(sigma_h)
    print(f"\n  静水压缩 p=100 MPa:")
    print(f"    von Mises: {vm_h/1e6:.6f} MPa (应=0)")
    print(f"    Tresca:    {tr_h/1e6:.6f} MPa (应=0)")

    # 纯剪切
    sigma_s = np.array([0, 0, 0, 0, 0, 100e6])
    vm_s = von_mises_stress(sigma_s)
    tr_s = tresca_yield(sigma_s)
    print(f"\n  纯剪切 τ=100 MPa:")
    print(f"    von Mises: {vm_s/1e6:.2f} MPa (应=√3·τ={np.sqrt(3)*100:.2f})")
    print(f"    Tresca:    {tr_s/1e6:.2f} MPa (应=τ=100)")

    # --- 3. 能量 ---
    print(f"\n{'='*60}")
    print("3. 弹塑性能量")
    print(f"{'='*60}")

    eps_test = np.array([3 * eps_y])
    sigma_test, _ = uniaxial_stress_strain(eps_test, E, sigma_y, H)
    U_e = elastic_energy(sigma_test, E)[0]
    U_p = plastic_dissipation(eps_test, E, sigma_y, H)[0]
    U_total = total_energy(eps_test, E, sigma_y, H)[0]
    print(f"  ε = 3ε_y = {eps_test[0]:.6e}")
    print(f"  σ = {sigma_test[0]/1e6:.2f} MPa")
    print(f"  弹性应变能: {U_e:.4e} J/m³")
    print(f"  塑性耗散:   {U_p:.4e} J/m³")
    print(f"  总能量:     {U_total:.4e} J/m³")
    print(f"  塑性/总:    {U_p/U_total:.4f}")

    # --- 4. 卸载 ---
    print(f"\n{'='*60}")
    print("4. 卸载与残余应变")
    print(f"{'='*60}")

    eps_max = 3 * eps_y
    eps_res, sigma_max = unload_stress(eps_max, E, sigma_y, H)
    print(f"  最大应变: {eps_max:.6e}")
    print(f"  最大应力: {sigma_max/1e6:.2f} MPa")
    print(f"  残余应变: {eps_res:.6e} (卸载后永久变形)")
    print(f"  回弹应变: {eps_max - eps_res:.6e} (弹性恢复)")

    # Bauschinger 效应
    sigma_rev = bauschinger_effect(eps_max, E, sigma_y, H)
    print(f"\n  Bauschinger 反向屈服: {sigma_rev/1e6:.2f} MPa")
    print(f"  (理想塑性应 = -{sigma_y/1e6:.2f} MPa)")

    # --- 5. 退化 ---
    print(f"\n{'='*60}")
    print("5. 退化验证")
    print(f"{'='*60}")

    err_deg = degradation_to_elastic(E, sigma_y, H)
    print(f"  σ_y→∞ 退化到纯弹性: 误差 {err_deg:.3e}")

    check_pure_elastic_when_below_yield(E, sigma_y)
    print(f"  低于屈服时纯弹性: ✓")

    print("\n=== 求解完成 ===")
