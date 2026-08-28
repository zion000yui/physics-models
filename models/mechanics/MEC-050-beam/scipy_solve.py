"""MEC-050 —— 用 SciPy 求解欧拉-伯努利梁。

涵盖：
1. 静态弯曲：solve_bvp 求解 4 阶 ODE
2. 动态模态动力学：solve_ivp 求解模态坐标 ODE
3. 有限差分固有频率：广义特征值问题

运行方法（在本文件所在目录执行）：
    python scipy_solve.py
"""

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.linalg import eigh

from model import (
    validate_parameters,
    static_cantilever_uniform_load,
    static_simply_supported_uniform_load,
    static_cantilever_tip_load,
    max_deflection_cantilever,
    max_deflection_simply_supported,
    natural_frequencies,
    mode_shape,
    mode_shape_second_derivative,
    modal_dynamics,
    modal_energy,
    reconstruct_displacement,
    modal_mass,
    modal_stiffness,
    verify_orthogonality,
    fd_natural_frequencies,
    beam_pde_rhs,
    bending_stiffness,
    mass_per_length,
)


# ============================================================
# 1. 静态 BVP 求解
# ============================================================

def solve_static_bvp_cantilever(q_load, L, E, I, n_points=101):
    """用 solve_bvp 求解悬臂梁静态弯曲。

    将 4 阶 ODE EI w'''' = q 降为 4 个 1 阶 ODE：
      y1 = w,    y1' = y2
      y2 = θ,    y2' = y3
      y3 = M/EI, y3' = y4  (注意符号：M = -EI w''，这里直接用 w'' = y3)
      y4 = w'''， y4' = q/EI

    实际上更简洁：
      y1 = w,  y1' = y2
      y2 = w', y2' = y3
      y3 = w'', y3' = y4
      y4 = w''', y4' = q/EI

    边界条件（悬臂）：
      x=0: w=0 (y1=0), w'=0 (y2=0)
      x=L: w''=0 (y3=0), w'''=0 (y4=0)
    """
    EI = E * I

    x = np.linspace(0, L, n_points)
    # 初始猜测：简单抛物线
    y_init = np.zeros((4, n_points))
    y_init[0] = q_load * x**2 * (6 * L**2 - 4 * L * x + x**2) / (24 * EI)

    def ode(x, y):
        """w'''' = q/EI → 4 个 1 阶 ODE。"""
        dy = np.zeros_like(y)
        dy[0] = y[1]        # w' = θ
        dy[1] = y[2]        # θ' = w''
        dy[2] = y[3]        # w''' = V/(-EI)
        dy[3] = q_load / EI * np.ones_like(x)  # w'''' = q/EI
        return dy

    def bc(ya, yb):
        """悬臂边界条件。"""
        # x=0: w=0, w'=0
        # x=L: w''=0, w'''=0
        return np.array([ya[0], ya[1], yb[2], yb[3]])

    sol = solve_bvp(ode, bc, x, y_init, tol=1e-8, max_nodes=5000)
    return sol


def solve_static_bvp_simply_supported(q_load, L, E, I, n_points=101):
    """用 solve_bvp 求解简支梁静态弯曲。

    边界条件（简支）：
      x=0: w=0 (y1=0), w''=0 (y3=0)
      x=L: w=0 (y1=0), w''=0 (y3=0)
    """
    EI = E * I
    x = np.linspace(0, L, n_points)
    y_init = np.zeros((4, n_points))
    y_init[0] = q_load * x * (L**3 - 2 * L * x**2 + x**3) / (24 * EI)

    def ode(x, y):
        dy = np.zeros_like(y)
        dy[0] = y[1]
        dy[1] = y[2]
        dy[2] = y[3]
        dy[3] = q_load / EI * np.ones_like(x)
        return dy

    def bc(ya, yb):
        """简支边界条件。"""
        return np.array([ya[0], ya[2], yb[0], yb[2]])

    sol = solve_bvp(ode, bc, x, y_init, tol=1e-8, max_nodes=5000)
    return sol


# ============================================================
# 2. 动态模态动力学
# ============================================================

def solve_modal_dynamics(omegas, q0, qdot0, t_end, forces_fn=None,
                         n_points=1001):
    """用 solve_ivp 求解模态坐标 ODE。

    q0, qdot0: 初始模态位移和速度
    """
    N = len(omegas)
    state0 = np.concatenate([q0, qdot0])

    sol = solve_ivp(
        modal_dynamics, (0, t_end), state0,
        args=(omegas, forces_fn, N),
        t_eval=np.linspace(0, t_end, n_points),
        rtol=1e-10, atol=1e-12,
    )
    return sol


# ============================================================
# 3. 有限差分固有频率
# ============================================================

def compute_fd_frequencies(N, L, E, I, rho, A, n_modes=5):
    """用有限差分法计算简支梁固有频率。"""
    return fd_natural_frequencies(N, L, E, I, rho, A, n_modes)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # --- 参数 ---
    E = 2.0e11    # 钢材杨氏模量 (Pa)
    I = 1.0e-8    # 截面惯性矩 (m⁴) — 1cm × 1cm 方截面: I = bh³/12 = 0.01*0.01³/12
    rho = 7850.0  # 钢材密度 (kg/m³)
    A = 1.0e-4    # 截面积 (m²) — 1cm × 1cm
    L = 1.0       # 梁长 (m)
    q = 100.0     # 均布载荷 (N/m)

    validate_parameters(E=E, I=I, rho=rho, A=A, L=L, q_load=q)

    print("=" * 60)
    print("MEC-050 Euler-Bernoulli Beam")
    print("=" * 60)
    print(f"参数: E={E:.2e} Pa, I={I:.2e} m⁴, ρ={rho} kg/m³, A={A:.2e} m²")
    print(f"      L={L} m, q={q} N/m")
    print(f"抗弯刚度 EI={E*I:.4f} N·m²")
    print(f"线密度 μ={rho*A:.6f} kg/m")

    # --- 1. 静态悬臂梁 ---
    print(f"\n{'='*60}")
    print("1. 悬臂梁静态弯曲（均布载荷）")
    print(f"{'='*60}")

    x = np.linspace(0, L, 201)
    w_ana, theta_ana, M_ana, V_ana = static_cantilever_uniform_load(x, q, L, E, I)
    w_max_ana = max_deflection_cantilever(q, L, E, I)
    print(f"解析最大挠度: w(L) = {w_max_ana:.6e} m = {w_max_ana*1e3:.4f} mm")

    sol_bvp = solve_static_bvp_cantilever(q, L, E, I)
    if sol_bvp.success:
        w_bvp = sol_bvp.sol(x)[0]
        err_w = np.max(np.abs(w_bvp - w_ana))
        print(f"BVP 求解成功，最大误差: {err_w:.3e} m")
    else:
        print(f"BVP 求解失败: {sol_bvp.message}")

    # --- 2. 静态简支梁 ---
    print(f"\n{'='*60}")
    print("2. 简支梁静态弯曲（均布载荷）")
    print(f"{'='*60}")

    w_ana_s, theta_ana_s, M_ana_s, V_ana_s = static_simply_supported_uniform_load(x, q, L, E, I)
    w_max_s = max_deflection_simply_supported(q, L, E, I)
    print(f"解析最大挠度: w(L/2) = {w_max_s:.6e} m = {w_max_s*1e3:.4f} mm")

    sol_bvp_s = solve_static_bvp_simply_supported(q, L, E, I)
    if sol_bvp_s.success:
        w_bvp_s = sol_bvp_s.sol(x)[0]
        err_w_s = np.max(np.abs(w_bvp_s - w_ana_s))
        print(f"BVP 求解成功，最大误差: {err_w_s:.3e} m")
    else:
        print(f"BVP 求解失败: {sol_bvp_s.message}")

    # --- 3. 固有频率 ---
    print(f"\n{'='*60}")
    print("3. 固有频率")
    print(f"{'='*60}")

    for bc_name in ['cantilever', 'simply_supported']:
        omegas = natural_frequencies(5, bc_name, E, I, rho, A, L)
        freqs_hz = omegas / (2 * np.pi)
        print(f"\n  {bc_name}:")
        print(f"  {'n':>3s}  {'ω (rad/s)':>14s}  {'f (Hz)':>14s}")
        for i, (w, f) in enumerate(zip(omegas, freqs_hz)):
            print(f"  {i+1:3d}  {w:14.4f}  {f:14.4f}")

    # --- 4. 有限差分 vs 解析固有频率 ---
    print(f"\n{'='*60}")
    print("4. 有限差分 vs 解析（简支梁）")
    print(f"{'='*60}")

    omegas_ana = natural_frequencies(5, 'simply_supported', E, I, rho, A, L)
    for N in [51, 101, 201]:
        omegas_fd = compute_fd_frequencies(N, L, E, I, rho, A, n_modes=5)
        errs = np.abs(omegas_fd[:5] - omegas_ana[:5]) / omegas_ana[:5] * 100
        print(f"  N={N:3d}  误差: {errs}")
    print(f"  解析值:  {omegas_ana}")

    # --- 5. 模态动力学（自由振动）---
    print(f"\n{'='*60}")
    print("5. 模态动力学（简支梁自由振动）")
    print(f"{'='*60}")

    omegas_ss = natural_frequencies(3, 'simply_supported', E, I, rho, A, L)
    # 初始条件：仅第一模态，振幅 1e-3 m（形状归一化）
    q0 = np.array([1e-3, 0, 0])
    qdot0 = np.array([0, 0, 0])
    t_end = 2 * np.pi / omegas_ss[0] * 5  # 5 个第一模态周期

    sol_dyn = solve_modal_dynamics(omegas_ss, q0, qdot0, t_end, n_points=501)

    # 能量
    energies = np.array([modal_energy(sol_dyn.y[:, i], omegas_ss)
                         for i in range(sol_dyn.y.shape[1])])
    print(f"初始能量: {energies[0]:.6e} J")
    print(f"终止能量: {energies[-1]:.6e} J")
    print(f"能量变化: {abs(energies[-1] - energies[0]):.3e} ({abs(energies[-1]-energies[0])/energies[0]*100:.2f}%)")

    # 检查模态 1 振荡频率
    q1 = sol_dyn.y[0, :]
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(q1)
    if len(peaks) >= 2:
        period_num = np.mean(np.diff(sol_dyn.t[peaks]))
        omega_num = 2 * np.pi / period_num
        print(f"模态 1 数值频率: {omega_num:.4f} rad/s (解析: {omegas_ss[0]:.4f})")
    else:
        print(f"峰值不足，无法计算频率")

    # --- 6. 模态正交性 ---
    print(f"\n{'='*60}")
    print("6. 模态正交性（悬臂梁）")
    print(f"{'='*60}")

    for i in range(1, 4):
        for j in range(i, 4):
            orth = verify_orthogonality(i, j, 'cantilever', rho, A, L)
            if i == j:
                print(f"  <φ{i}, φ{j}> = {orth:.6e} (应为非零，模态质量)")
            else:
                print(f"  <φ{i}, φ{j}> = {orth:.6e} (应≈0)")

    # --- 7. 模态质量/刚度验证 ω² = k/m ---
    print(f"\n{'='*60}")
    print("7. 模态质量/刚度验证 ω² = k_n / m_n（悬臂梁）")
    print(f"{'='*60}")

    omegas_c = natural_frequencies(3, 'cantilever', E, I, rho, A, L)
    for n in range(1, 4):
        m_n = modal_mass(n, 'cantilever', rho, A, L)
        k_n = modal_stiffness(n, 'cantilever', E, I, L)
        omega_sq = k_n / m_n
        print(f"  模态 {n}: m_n={m_n:.6e}, k_n={k_n:.6e}, "
              f"√(k/m)={np.sqrt(omega_sq):.4f}, ω_ana={omegas_c[n-1]:.4f}")

    print("\n=== 求解完成 ===")
