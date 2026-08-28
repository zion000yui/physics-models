"""pytest conftest — 隔离各 MEC 模型目录的模块导入。

各模型测试使用 `from model import ...` 裸模块名导入。
当 pytest 从根目录批量收集时，sys.modules['model'] 会被第一个
模型锁定，导致后续模型导入到错误的 model 模块。

解决方案：在 conftest.py 中用 pytest_collect_file 钩子，
为每个模型目录的 test 文件动态添加该目录到 sys.path，
并在收集前清除已缓存的 model 模块。
"""

import sys
import importlib
from pathlib import Path


def pytest_collect_file(file_path, parent):
    """收集测试文件前，清除可能冲突的模块缓存。"""
    # 清除所有 MEC 模型可能冲突的模块名
    for mod_name in list(sys.modules.keys()):
        if mod_name in ('model', 'scipy_solve'):
            del sys.modules[mod_name]

    # 将 test 文件所在目录加入 sys.path 头部
    test_dir = str(file_path.parent)
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)

    # 让 pytest 用默认收集器
    return None  # 返回 None 让 pytest 使用默认行为


def pytest_runtest_setup(item):
    """每个测试运行前，清除 model 模块缓存并确保正确的 sys.path。"""
    # 获取测试文件路径
    test_file = item.fspath if hasattr(item, 'fspath') else Path(str(item.nodeid).split('::')[0])
    if hasattr(test_file, 'dirname'):
        test_dir = test_file.dirname
    elif hasattr(test_file, 'parent'):
        test_dir = str(test_file.parent)
    else:
        test_dir = str(Path(str(test_file)).parent)

    # 清除可能冲突的模块
    for mod_name in list(sys.modules.keys()):
        if mod_name in ('model', 'scipy_solve'):
            del sys.modules[mod_name]

    # 确保测试目录在 path 最前
    # 先移除其他 MEC 目录
    paths_to_remove = []
    for p in sys.path:
        if 'MEC-' in p and p != test_dir:
            paths_to_remove.append(p)
    for p in paths_to_remove:
        sys.path.remove(p)

    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    elif sys.path[0] != test_dir:
        sys.path.remove(test_dir)
        sys.path.insert(0, test_dir)
