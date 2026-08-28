"""MEC-090-nonlinear-mechanics — 模型定义（引擎无关）

非线性力学：分岔、混沌、庞加莱截面。
复用 MEC-013 双摆和 MEC-015 非线性单摆的数据，展示非线性系统特征。

=== 物理系统 ===

  1) 非线性单摆（大角度单摆，MEC-015）
     θ̈ + (g/l)sinθ = 0  （非线性，非简谐）

     特征：
     - 周期 T 依赖振幅（与线性单摆 T=2π√(l/g) 不同）
     - 振幅 → π 时周期 → ∞（同宿轨道）
     - 相空间有不动点 (0,0) 和 (π,0)（鞍点）

  2) 受驱阻尼摆（Duffing 类型）
     θ̈ + cθ̇ + (g/l)sinθ = A·cos(ω_d·t)

     特征：
     - 有驱动力 → 非自治系统
     - 驱动参数变化可导致分岔和混沌
     - 庞加莱截面可揭示混沌结构

  3) 双摆（MEC-013）
     非线性耦合二自由度系统，能量足够大时混沌。

=== 非线性特征量 ===

  - 庞加莱截面（Poincaré section）：在驱动周期 Ts = 2π/ω_d 处采样
  - 分岔图：稳态振幅 vs 驱动参数
  - Lyapunov 指数：衡量轨迹发散率

=== 周期振幅依赖性（非线性单摆）===

  小角度：T₀ = 2π√(l/g)（线性，振幅无关）
  大角度：
    T = T₀ [1 + (1/16)θ₀² + (11/3072)θ₀⁴ + ...]
    或 T = 4√(l/g) K(sin²(θ₀/2))  （第一类完全椭圆积分）

  θ₀ → π: T → ∞

=== 与已有 MEC 模型的关系 ===

  MEC-015 非线性单摆 → MEC-090 周期-振幅关系
  MEC-013 双摆 → MEC-090 混沌行为
  MEC-060 拉格朗日 → MEC-090 非线性方程推导
  MEC-080 多体 → MEC-090 多体混沌

说明：本文件不依赖任何求解引擎，只描述物理本身。
"""

import numpy as np
from scipy.integrate import solve_ivp


def validate_parameters(g=9.81, l=1.0, c=0.0, A=0.0, omega_d=0.0):
    """验证物理参数合法性。"""
    assert g > 0, f"重力加速度 g 必须为正，当前 g={g}"
    assert l > 0, f"长度 l 必须为正，当前 l={l}"
    assert c >= 0, f"阻尼系数 c 必须非负，当前 c={c}"
    assert A >= 0, f"驱动力振幅 A 必须非负，当前 A={A}"


# ============================================================
# 非线性单摆（无驱动无阻尼）
# ============================================================

def pendulum_dynamics(t, state, g=9.81, l=1.0):
    """非线性单摆：θ̈ = -(g/l)sinθ。"""
    th, w = state
    return np.array([w, -g / l * np.sin(th)])


def pendulum_linear_frequency(g, l):
    """线性化频率 ω₀ = √(g/l)。"""
    return np.sqrt(g / l)


def pendulum_period_analytical(th0, g=9.81, l=1.0):
    """非线性单摆周期（椭圆积分精确公式）。

    T = 4√(l/g) · K(sin²(θ₀/2))
    其中 K 为第一类完全椭圆积分。
    """
    from scipy.special import ellipk
    m = np.sin(th0 / 2)**2  # 椭圆积分参数
    if m >= 1.0:
        return float('inf')  # θ₀ = π 时周期无穷
    return 4 * np.sqrt(l / g) * ellipk(m)


def pendulum_period_series(th0, g=9.81, l=1.0):
    """周期振幅依赖性的级数展开。

    T = T₀ [1 + θ₀²/16 + 11θ₀⁴/3072 + ...]
    """
    T0 = 2 * np.pi * np.sqrt(l / g)
    return T0 * (1 + th0**2 / 16 + 11 * th0**4 / 3072)


def pendulum_energy(state, g, l):
    """E = ½ml²ω² + mgl(1-cosθ)，取 m=1。"""
    th, w = state
    return 0.5 * l**2 * w**2 + g * l * (1 - np.cos(th))


def pendulum_energy_threshold(g, l):
    """旋转能量阈值 E_c = 2mgl（m=1）。"""
    return 2 * g * l


# ============================================================
# 受驱阻尼摆（Duffing 型）
# ============================================================

def driven_damped_pendulum(t, state, g=9.81, l=1.0, c=0.1, A=1.0, omega_d=2.0/3.0):
    """受驱阻尼摆：θ̈ + cθ̇ + (g/l)sinθ = A·cos(ω_d t)。"""
    th, w = state
    return np.array([w, -c * w - g / l * np.sin(th) + A * np.cos(omega_d * t)])


def solve_driven_pendulum(g, l, c, A, omega_d, th0=0.2, w0=0.0,
                            t_end=100.0, n=10001):
    """数值积分受驱阻尼摆。"""
    t_eval = np.linspace(0, t_end, n)
    sol = solve_ivp(driven_damped_pendulum, (0, t_end), [th0, w0],
                    args=(g, l, c, A, omega_d),
                    t_eval=t_eval, rtol=1e-10, atol=1e-12,
                    method='RK45')
    return sol


# ============================================================
# 庞加莱截面
# ============================================================

def poincare_section(sol, omega_d, t_start=10.0):
    """计算庞加莱截面（在每个驱动周期采样）。

    在 t = n·T_d (T_d = 2π/ω_d) 处采样。
    跳过前 t_start 秒（瞬态衰减）。
    """
    T_d = 2 * np.pi / omega_d
    n_periods = int((sol.t[-1] - t_start) / T_d)
    sections = []
    for n in range(1, n_periods + 1):
        t_sample = t_start + n * T_d
        if t_sample > sol.t[-1]:
            break
        # 插值
        th = np.interp(t_sample, sol.t, sol.y[0])
        w = np.interp(t_sample, sol.t, sol.y[1])
        sections.append([th, w])
    return np.array(sections)


# ============================================================
# Lyapunov 指数（数值估计）
# ============================================================

def estimate_lyapunov_exponent(g, l, c, A, omega_d,
                                th0=0.2, w0=0.0,
                                t_end=200.0, n=20001,
                                delta0=1e-8):
    """估计最大 Lyapunov 指数（双轨迹法）。

    方法：两条初始距离 δ₀ 的轨迹，测量距离增长率。
    λ ≈ (1/t) · ln(d(t)/δ₀)
    """
    # 两条轨迹
    t_eval = np.linspace(0, t_end, n)
    sol1 = solve_ivp(driven_damped_pendulum, (0, t_end), [th0, w0],
                     args=(g, l, c, A, omega_d),
                     t_eval=t_eval, rtol=1e-12, atol=1e-14)
    sol2 = solve_ivp(driven_damped_pendulum, (0, t_end),
                     [th0 + delta0, w0],
                     args=(g, l, c, A, omega_d),
                     t_eval=t_eval, rtol=1e-12, atol=1e-14)

    # 距离（考虑角度周期性）
    dth = sol2.y[0] - sol1.y[0]
    dth = np.arctan2(np.sin(dth), np.cos(dth))  # 归一到 [-π, π]
    dw = sol2.y[1] - sol1.y[1]
    d = np.sqrt(dth**2 + dw**2)

    # Lyapunov 指数（取后期对数增长率）
    mask = (t_eval > 10) & (d > 1e-20) & (d < 1.0)
    if np.sum(mask) > 10:
        log_d = np.log(d[mask])
        t_mask = t_eval[mask]
        # 线性回归 log_d vs t
        coeffs = np.polyfit(t_mask, log_d, 1)
        return coeffs[0]
    return 0.0


# ============================================================
# 双摆混沌验证
# ============================================================

def double_pendulum_dynamics_inline(t, state, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """双摆动力学方程（内联，不依赖 MEC-060）。"""
    th1, th2, w1, w2 = state
    delta = th1 - th2
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)

    M11 = (m1 + m2) * l1**2
    M12 = m2 * l1 * l2 * cos_d
    M22 = m2 * l2**2
    det = M11 * M22 - M12**2

    b1 = -m2 * l1 * l2 * sin_d * w2**2 - (m1 + m2) * g * l1 * np.sin(th1)
    b2 = m2 * l1 * l2 * sin_d * w1**2 - m2 * g * l2 * np.sin(th2)

    a1 = (b1 * M22 - b2 * M12) / det
    a2 = (M11 * b2 - M12 * b1) / det
    return np.array([w1, w2, a1, a2])


def double_pendulum_divergence(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81,
                                 th1=0.5, th2=0.5, delta=1e-8,
                                 t_end=20.0, n=20001):
    """双摆轨迹发散验证。

    两条初始条件差 δ 的双摆，测量角度差增长率。
    """
    t_eval = np.linspace(0, t_end, n)
    sol1 = solve_ivp(double_pendulum_dynamics_inline, (0, t_end),
                     [th1, th2, 0, 0],
                     args=(m1, m2, l1, l2, g),
                     t_eval=t_eval, rtol=1e-12, atol=1e-14)
    sol2 = solve_ivp(double_pendulum_dynamics_inline, (0, t_end),
                     [th1 + delta, th2, 0, 0],
                     args=(m1, m2, l1, l2, g),
                     t_eval=t_eval, rtol=1e-12, atol=1e-14)

    dth1 = sol2.y[0] - sol1.y[0]
    dth1 = np.arctan2(np.sin(dth1), np.cos(dth1))
    d = np.abs(dth1)

    return t_eval, d
