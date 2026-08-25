自由质点：不受外力、无约束的点质量。这是整个仓库里最简单的模型，

用来验证「模型定义 → 引擎求解 → 解析解对照」的最小闭环。
物理模型

状态：位置 x、速度 v

一阶常微分方程：
dx/dt = v
dv/dt = 0

解析解（正确性金标准）：
x(t) = x0 + v0 * t
v(t) = v0

文件说明
文件	              作用
model.py	          引擎无关：动力学方程 dynamics + 解析解 analytical
scipy_solve.py	    用 SciPy solve_ivp 做数值积分并打印误差
test_consistency.py	数值解 vs 解析解一致性测试

运行
bash
python scipy_solve.py
python test_consistency.py

依赖
numpy
scipy
