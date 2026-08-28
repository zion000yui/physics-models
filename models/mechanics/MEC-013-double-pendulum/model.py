"""MEC-013-double-pendulum —— 模型定义（引擎无关）

双摆（double pendulum）：两个单摆串联连接，第一个摆悬挂于固定支点，
第二个摆悬挂于第一个摆的末端。系统具有强非线性耦合，表现出混沌行为。

双摆是经典力学中混沌动力学的标准范例：
- 小角度极限退化为 MEC-014 耦合振子（线性化）
- 大角度行为是非线性的、不可预测的、对初始条件极度敏感
- 与 MEC-015 非线性单摆形成对照（单自由度非线性 vs 多自由度混沌）

状态向量 state = [theta1, theta2, omega1, omega2]
    theta1, theta2 —— 上摆、下摆的角度（rad，相对于竖直方向）
    omega1, omega2 —— 上摆、下摆的角速度（rad/s）

参数：
    m1, m2 —— 上摆、下摆的质量（kg，m > 0）
    L1, L2 —— 上摆、下摆的摆长（m，L > 0）
    g     —— 重力加速度（m/s²，g > 0）

动力学（一阶常微分方程）：

    双摆的拉格朗日方程推导出的运动方程为：

    (m1+m2)·L1·θ̈1 + m2·L2·θ̈2·cos(θ1-θ2)
        + m2·L2·θ̇2²·sin(θ1-θ2) + (m1+m2)·g·sin(θ1) = 0

    m2·L2·θ̈2 + m2·L1·θ̈1·cos(θ1-θ2)
        - m2·L1·θ̇1²·sin(θ1-θ2) + m2·g·sin(θ2) = 0

    这是耦合的二阶 ODE 系统，需要解出 θ̈1 和 θ̈2 的显式表达式。
    通过克莱默法则求解 2×2 线性方程组，得到：

    令 Δ = θ1 - θ2，A = (m1+m2)·L1，B = m2·L2·cos(Δ)，
        C = m2·L1·cos(Δ)，D = m2·L2

    det = A·D - B·C

    右端项：
        rhs1 = -(m2·L2·ω2²·sin(Δ) + (m1+m2)·g·sin(θ1))
        rhs2 = -(m2·L1·ω1²·(-sin(Δ)) + m2·g·sin(θ2))
             = m2·L1·ω1²·sin(Δ) - m2·g·sin(θ2)

    θ̈1 = (D·rhs1 - B·rhs2) / det
    θ̈2 = (A·rhs2 - C·rhs1) / det

守恒量：

    总机械能（无阻尼时守恒）：
        E = ½·(m1+m2)·L1²·ω1² + ½·m2·L2²·ω2²
            + m2·L1·L2·ω1·ω2·cos(θ1-θ2)
            + (m1+m2)·g·L1·(1-cos(θ1)) + m2·g·L2·(1-cos(θ2))

小角度线性化（退化为 MEC-014）：

    当 |θ1|, |θ2| ≪ 1 时，cos(θ1-θ2)≈1, sin(θ1)≈θ1, sin(θ2)≈θ2，
    sin(θ1-θ2)≈θ1-θ2, ω² 项忽略。方程退化为线性耦合振子，
    可通过简正模态分析求解，与 MEC-014 形式一致。

混沌特征：

    - 对初始条件极度敏感（Lyapunov 指数为正）
    - 相空间轨迹不可预测
    - Poincaré 截面显示混沌区域
    - 能量守恒但运动无周期

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
    """验证基本物理参数合法性。

    参数
    ----
    m1, m2 : float
        上摆、下摆的质量（必须 > 0）。
    L1, L2 : float
        上摆、下摆的摆长（必须 > 0）。
    g : float
        重力加速度（必须 > 0）。

    返回
    ----
    None
        条件满足时静默返回。

    抛出
    ----
    AssertionError
        参数不合法时给出明确错误信息。
    """
    assert m1 > 0, f"质量 m1 必须为正，当前 m1={m1}"
    assert m2 > 0, f"质量 m2 必须为正，当前 m2={m2}"
    assert L1 > 0, f"摆长 L1 必须为正，当前 L1={L1}"
    assert L2 > 0, f"摆长 L2 必须为正，当前 L2={L2}"
    assert g > 0, f"重力加速度 g 必须为正，当前 g={g}"


def mechanical_energy(state, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
    """计算总机械能。

    E = ½(m1+m2)L1²ω1² + ½m2L2²ω2² + m2L1L2ω1ω2cos(θ1-θ2)
        + (m1+m2)gL1(1-cos(θ1)) + m2gL2(1-cos(θ2))

    参数
    ----
    state : array_like, shape (4,)
        状态 [theta1, theta2, omega1, omega2]。
    m1, m2, L1, L2, g : float
        物理参数。

    返回
    ----
    float
        总机械能。
    """
    t1, t2, w1, w2 = state
    delta = t1 - t2
    ke = (0.5 * (m1 + m2) * L1 ** 2 * w1 ** 2
          + 0.5 * m2 * L2 ** 2 * w2 ** 2
          + m2 * L1 * L2 * w1 * w2 * np.cos(delta))
    pe = ((m1 + m2) * g * L1 * (1 - np.cos(t1))
          + m2 * g * L2 * (1 - np.cos(t2)))
    return ke + pe


def dynamics(t, state, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
    """返回状态的时间导数 d(state)/dt。

    使用完整非线性双摆运动方程。

    参数
    ----
    t : float
        当前时刻（保守系统不依赖 t，保留以统一接口）。
    state : array_like, shape (4,)
        状态 [theta1, theta2, omega1, omega2]。
    m1, m2, L1, L2, g : float
        物理参数。

    返回
    ----
    np.ndarray, shape (4,)
        [omega1, omega2, alpha1, alpha2]
    """
    t1, t2, w1, w2 = state
    delta = t1 - t2
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)

    # 系数矩阵元素
    A = (m1 + m2) * L1
    B = m2 * L2 * cos_d
    C = m2 * L1 * cos_d
    D = m2 * L2

    det = A * D - B * C  # = m2*L2*((m1+m2)*L1 - m2*L1*cos²(Δ)) > 0

    # 右端项（不含惯性耦合项的加速度）
    rhs1 = -(m2 * L2 * w2 ** 2 * sin_d + (m1 + m2) * g * np.sin(t1))
    rhs2 = m2 * L1 * w1 ** 2 * sin_d - m2 * g * np.sin(t2)

    # 克莱默法则求解 θ̈1, θ̈2
    alpha1 = (D * rhs1 - B * rhs2) / det
    alpha2 = (A * rhs2 - C * rhs1) / det

    return np.array([w1, w2, alpha1, alpha2])
