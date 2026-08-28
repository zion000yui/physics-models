"""MEC-014-coupled-oscillators —— 模型定义（引擎无关）

耦合振子（coupled oscillators）：两个质量块通过弹簧串联并耦合振动。
这是多自由度振动系统的标准入门模型，引入简正模态（normal modes）概念，
是 MEC-013 双摆、MEC-060 分析力学和 MEC-080 多体动力学的重要过渡。

物理模型：三个弹簧 + 两个质量块
    墙壁 —[k1]— m1 —[kc]— m2 —[k2]— 墙壁

    质点 1（质量 m1，位移 x1）受左侧弹簧 k1 和耦合弹簧 kc 作用。
    质点 2（质量 m2，位移 x2）受右侧弹簧 k2 和耦合弹簧 kc 作用。

状态向量 state = [x1, x2, v1, v2]
    x1, x2 —— 质点 1、2 的位移（相对于各自平衡位置）
    v1, v2 —— 质点 1、2 的速度

参数：
    m1, m2 —— 两个质点的质量（kg，m > 0）
    k1     —— 左侧弹簧弹性系数（N/m，k1 > 0）
    k2     —— 右侧弹簧弹性系数（N/m，k2 > 0）
    kc     —— 耦合弹簧弹性系数（N/m，kc ≥ 0）

动力学（一阶常微分方程）：

    质点 1 受力：F1 = -k1·x1 + kc·(x2 - x1) = -(k1+kc)·x1 + kc·x2
    质点 2 受力：F2 = -k2·x2 + kc·(x1 - x2) = kc·x1 - (k2+kc)·x2

    因此：
        dx1/dt = v1
        dx2/dt = v2
        dv1/dt = -(k1+kc)/m1 · x1 + kc/m1 · x2
        dv2/dt = kc/m2 · x1 - (k2+kc)/m2 · x2

    矩阵形式：M·ẍ + K·x = 0
    其中 M = diag(m1, m2)，K = [[k1+kc, -kc], [-kc, k2+kc]]

简正模态（normal modes）：

    通过求解广义特征值问题 K·φ = ω²·M·φ 得到两个简正模态：
    - 简正频率 ω₁, ω₂（对应两个正交的模态形状）
    - 模态向量 φ₁, φ₂

    对称系统（m1=m2=m, k1=k2=k）的简正模态：
    - 同相模态（in-phase）：x1=x2，频率 ω₁ = √(k/m)
    - 反相模态（anti-phase）：x1=-x2，频率 ω₂ = √((k+2kc)/m）

    通解 = 两个简正模态的线性叠加：
        x(t) = A₁·φ₁·cos(ω₁·t+φ₁_phase) + A₂·φ₂·cos(ω₂·t+φ₂_phase)

无耦合极限：
    kc=0 时两个振子完全独立，退化为两个 MEC-010 简谐振子。
    频率分别为 ω₁=√(k1/m1)，ω₂=√(k2/m2)。

守恒量：

    总机械能：
        E = ½·m1·v1² + ½·m2·v2² + ½·k1·x1² + ½·k2·x2² + ½·kc·(x2-x1)²
    无阻尼时机械能守恒。

模态空间表示：

    本模型有 4 维相空间 (x1, x2, v1, v2)。通过简正模态变换
    可投影到两个独立的 2D 子空间 (q₁, q̇₁) 和 (q₂, q̇₂)，
    每个子空间内行为等价于一个独立的简谐振子。

说明：本文件不依赖任何求解引擎，只描述物理本身。
后续换 PyTorch / MuJoCo / Modelica 时，复用这里的方程即可。
"""

import numpy as np


def validate_parameters(m1=1.0, m2=1.0, k1=1.0, k2=1.0, kc=0.5):
    """验证基本物理参数合法性。

    参数
    ----
    m1, m2 : float
        两个质点的质量（必须 > 0）。
    k1, k2 : float
        两侧弹簧弹性系数（必须 > 0）。
    kc : float
        耦合弹簧弹性系数（必须 ≥ 0）。

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
    assert k1 > 0, f"弹簧系数 k1 必须为正，当前 k1={k1}"
    assert k2 > 0, f"弹簧系数 k2 必须为正，当前 k2={k2}"
    assert kc >= 0, f"耦合弹簧系数 kc 必须非负，当前 kc={kc}"


def stiffness_matrix(k1=1.0, k2=1.0, kc=0.5):
    """构造刚度矩阵 K = [[k1+kc, -kc], [-kc, k2+kc]]。

    参数
    ----
    k1, k2, kc : float
        弹簧系数。

    返回
    ----
    np.ndarray, shape (2, 2)
        刚度矩阵。
    """
    return np.array([
        [k1 + kc, -kc],
        [-kc, k2 + kc]
    ])


def mass_matrix(m1=1.0, m2=1.0):
    """构造质量矩阵 M = diag(m1, m2)。

    参数
    ----
    m1, m2 : float
        两个质点的质量。

    返回
    ----
    np.ndarray, shape (2, 2)
        质量矩阵（对角矩阵）。
    """
    return np.diag([m1, m2])


def normal_modes(m1=1.0, m2=1.0, k1=1.0, k2=1.0, kc=0.5):
    """计算简正模态（频率和模态向量）。

    求解广义特征值问题 K·φ = ω²·M·φ。

    参数
    ----
    m1, m2, k1, k2, kc : float
        物理参数。

    返回
    ----
    list of dict
        每个字典包含：
        - 'omega': 简正角频率（rad/s）
        - 'mode': 模态向量（np.ndarray, shape (2,)），归一化到第一个分量为 1
        按频率升序排列。
    """
    K = stiffness_matrix(k1, k2, kc)
    M = mass_matrix(m1, m2)
    # 求解 M^{-1} K 的特征值
    Minv_K = np.linalg.solve(M, K)
    eigenvalues, eigenvectors = np.linalg.eig(Minv_K)
    # 按特征值升序排列
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real
    results = []
    for i in range(2):
        omega = np.sqrt(max(eigenvalues[i], 0.0))
        vec = eigenvectors[:, i]
        # 归一化：第一个分量为 1
        if abs(vec[0]) > 1e-15:
            vec = vec / vec[0]
        else:
            # 第一个分量接近零，归一化到第二个分量
            vec = vec / vec[1]
        results.append({
            'omega': omega,
            'mode': vec,
        })
    return results


def mechanical_energy(state, m1=1.0, m2=1.0, k1=1.0, k2=1.0, kc=0.5):
    """计算总机械能。

    E = ½·m1·v1² + ½·m2·v2² + ½·k1·x1² + ½·k2·x2² + ½·kc·(x2-x1)²

    参数
    ----
    state : array_like, shape (4,)
        状态 [x1, x2, v1, v2]。
    m1, m2, k1, k2, kc : float
        物理参数。

    返回
    ----
    float
        总机械能。
    """
    x1, x2, v1, v2 = state
    ke = 0.5 * m1 * v1 ** 2 + 0.5 * m2 * v2 ** 2
    pe = (0.5 * k1 * x1 ** 2 + 0.5 * k2 * x2 ** 2
           + 0.5 * kc * (x2 - x1) ** 2)
    return ke + pe


def dynamics(t, state, m1=1.0, m2=1.0, k1=1.0, k2=1.0, kc=0.5):
    """返回状态的时间导数 d(state)/dt。

    参数
    ----
    t : float
        当前时刻（保守系统不依赖 t，保留以统一接口）。
    state : array_like, shape (4,)
        状态 [x1, x2, v1, v2]。
    m1, m2, k1, k2, kc : float
        物理参数。

    返回
    ----
    np.ndarray, shape (4,)
        [v1, v2, a1, a2]
    """
    x1, x2, v1, v2 = state
    a1 = -(k1 + kc) / m1 * x1 + kc / m1 * x2
    a2 = kc / m2 * x1 - (k2 + kc) / m2 * x2
    return np.array([v1, v2, a1, a2])


def analytical(t, initial_state, m1=1.0, m2=1.0, k1=1.0, k2=1.0, kc=0.5):
    """耦合振子解析解（通过简正模态分解）。

    将初始条件投影到简正模态空间，在每个模态空间中独立求解
    简谐振动，然后叠加回物理坐标。

    参数
    ----
    t : float 或 array_like
        时间点。
    initial_state : array_like, shape (4,)
        初始状态 [x1_0, x2_0, v1_0, v2_0]。
    m1, m2, k1, k2, kc : float
        物理参数。

    返回
    ----
    (x1, x2, v1, v2) : tuple
        四个质点的位置和速度，形状与 t 一致。
    """
    t = np.asarray(t, dtype=float)
    x1_0, x2_0, v1_0, v2_0 = initial_state

    modes = normal_modes(m1, m2, k1, k2, kc)
    omega1, omega2 = modes[0]['omega'], modes[1]['omega']
    phi1, phi2 = modes[0]['mode'], modes[1]['mode']

    # 构造模态矩阵 P = [phi1, phi2]（列向量）
    P = np.column_stack([phi1, phi2])  # shape (2, 2)

    # 物理初始条件投影到模态空间
    # x = P · q，所以 q = P^{-1} · x
    x0_vec = np.array([x1_0, x2_0])
    v0_vec = np.array([v1_0, v2_0])
    q0 = np.linalg.solve(P, x0_vec)
    qd0 = np.linalg.solve(P, v0_vec)

    # 在每个模态空间中求解简谐振动
    # q_i(t) = q0_i·cos(ω_i·t) + (qd0_i/ω_i)·sin(ω_i·t)
    # 处理 ω=0 的情况（无耦合且某弹簧为零，但此处 k1,k2>0 保证 ω>0）
    q1 = q0[0] * np.cos(omega1 * t) + (qd0[0] / omega1) * np.sin(omega1 * t)
    q2 = q0[1] * np.cos(omega2 * t) + (qd0[1] / omega2) * np.sin(omega2 * t)

    qd1 = -q0[0] * omega1 * np.sin(omega1 * t) + qd0[0] * np.cos(omega1 * t)
    qd2 = -q0[1] * omega2 * np.sin(omega2 * t) + qd0[1] * np.cos(omega2 * t)

    # 叠加回物理坐标
    # x = P · [q1, q2]^T
    x1 = phi1[0] * q1 + phi2[0] * q2
    x2 = phi1[1] * q1 + phi2[1] * q2
    v1 = phi1[0] * qd1 + phi2[0] * qd2
    v2 = phi1[1] * qd1 + phi2[1] * qd2

    return x1, x2, v1, v2
