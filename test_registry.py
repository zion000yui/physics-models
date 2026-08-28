"""Registry 测试 — 验证 Model Registry 的发现、查询、加载功能。

运行方式（在项目根目录执行）：
    python -m pytest test_registry.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from registry import ModelRegistry, ModelEntry


@pytest.fixture
def registry():
    """创建 Registry 实例。"""
    return ModelRegistry()


# ============================================================
# 1. Catalog 加载
# ============================================================

def test_catalog_loads(registry):
    """catalog.json 能被正确加载。"""
    assert registry._catalog is not None
    assert registry._catalog["version"] == "1.1.0"


# ============================================================
# 2. 模型数量
# ============================================================

def test_model_count_38(registry):
    """模型总数应为 38。"""
    assert len(registry.list_models()) == 38


# ============================================================
# 3. ID 唯一性
# ============================================================

def test_all_ids_unique(registry):
    """所有模型 ID 应唯一。"""
    ids = registry.list_models()
    assert len(ids) == len(set(ids))


# ============================================================
# 4. 目录存在性
# ============================================================

def test_all_directories_exist(registry):
    """catalog 中的目录都实际存在。"""
    mechanics_dir = registry._mechanics_dir
    for mid, info in registry._models.items():
        dir_path = mechanics_dir / info.directory
        assert dir_path.exists(), f"{mid}: 目录不存在 {dir_path}"


# ============================================================
# 5. 四件套完整性
# ============================================================

def test_all_four_files_exist(registry):
    """每个模型的 model.py / scipy_solve.py / test / README 都存在。"""
    mechanics_dir = registry._mechanics_dir
    for mid, info in registry._models.items():
        assert (mechanics_dir / info.model_path).exists(), f"{mid}: model.py 缺失"
        assert (mechanics_dir / info.scipy_solve_path).exists(), f"{mid}: scipy_solve.py 缺失"
        assert info.test_path is not None, f"{mid}: test_path 为 null"
        assert (mechanics_dir / info.test_path).exists(), f"{mid}: test 文件缺失"
        readme = mechanics_dir / info.directory / "README.md"
        assert readme.exists(), f"{mid}: README.md 缺失"


# ============================================================
# 6. 分类完整性
# ============================================================

def test_categories_complete(registry):
    """应有 10 个分类。"""
    cats = registry.list_categories()
    assert len(cats) == 10
    expected = {
        "point-particle-mechanics",
        "oscillatory-systems",
        "rigid-body-dynamics",
        "mechanisms",
        "contact-and-impact",
        "continuum-mechanics",
        "analytical-mechanics",
        "multibody-dynamics",
        "nonlinear-mechanics",
        "elastoplasticity",
    }
    assert set(cats) == expected


# ============================================================
# 7. get_model_info
# ============================================================

def test_get_model_info(registry):
    """get_model_info 返回正确的 ModelEntry。"""
    info = registry.get_model_info("MEC-010")
    assert isinstance(info, ModelEntry)
    assert info.id == "MEC-010"
    assert info.name == "mass-spring"
    assert info.category == "oscillatory-systems"
    assert info.has_dynamics is True
    assert info.dynamics_fn == "dynamics"
    assert info.has_energy is True
    assert info.energy_fn == "mechanical_energy"


# ============================================================
# 8. get_model 动态加载
# ============================================================

def test_get_model_loads(registry):
    """get_model 返回的 module 有预期的函数。"""
    model = registry.get_model("MEC-010")
    assert hasattr(model, "dynamics")
    assert hasattr(model, "analytical")
    assert hasattr(model, "mechanical_energy")
    assert hasattr(model, "validate_parameters")


def test_get_model_no_module_conflict(registry):
    """连续加载多个模型不应冲突。"""
    m1 = registry.get_model("MEC-001")
    m2 = registry.get_model("MEC-010")
    assert hasattr(m1, "analytical")
    assert hasattr(m2, "dynamics")
    # 确保加载的不是同一个模块
    assert m1 is not m2


# ============================================================
# 9. has_interface
# ============================================================

def test_has_interface(registry):
    """has_interface 正确返回 True/False。"""
    assert registry.has_interface("MEC-010", "dynamics") is True
    assert registry.has_interface("MEC-010", "energy") is True
    assert registry.has_interface("MEC-053", "dynamics") is False
    assert registry.has_interface("MEC-100", "dynamics") is False
    assert registry.has_interface("MEC-053", "energy") is True


# ============================================================
# 10. 分类查询
# ============================================================

def test_list_models_by_category(registry):
    """按分类过滤正确。"""
    oscillatory = registry.list_models(category="oscillatory-systems")
    assert len(oscillatory) == 6
    assert "MEC-010" in oscillatory
    assert "MEC-015" in oscillatory

    continuum = registry.list_models(category="continuum-mechanics")
    assert len(continuum) == 4
    assert "MEC-050" in continuum
    assert "MEC-053" in continuum


# ============================================================
# 11. 属性搜索
# ============================================================

def test_search(registry):
    """按属性搜索正确。"""
    with_dynamics = registry.search(has_dynamics=True)
    assert len(with_dynamics) == 36
    assert "MEC-010" in with_dynamics
    assert "MEC-053" not in with_dynamics

    without_dynamics = registry.search(has_dynamics=False)
    assert len(without_dynamics) == 2
    assert "MEC-053" in without_dynamics
    assert "MEC-100" in without_dynamics

    with_energy_and_dynamics = registry.search(has_dynamics=True, has_energy=True)
    assert len(with_energy_and_dynamics) == 27

    by_category = registry.search(category="elastoplasticity")
    assert by_category == ["MEC-100"]


# ============================================================
# 12. dynamics 可调用性
# ============================================================

def test_dynamics_callable(registry):
    """对有 dynamics 的模型，加载后 dynamics 可调用。"""
    model = registry.get_model("MEC-010")
    result = model.dynamics(0.0, [1.0, 0.0], k=1.0, m=1.0)
    assert len(result) == 2


def test_dynamics_callable_2d(registry):
    """4 维 state 的模型也能调用。"""
    model = registry.get_model("MEC-006")
    result = model.dynamics(0.0, [1.0, 0.0, 0.0, 0.0], k=1.0, m=1.0)
    assert len(result) == 4


# ============================================================
# 13. 无 dynamics 模型的正确识别
# ============================================================

def test_no_dynamics_models(registry):
    """MEC-053 和 MEC-100 的 has_dynamics = False。"""
    info_053 = registry.get_model_info("MEC-053")
    assert info_053.has_dynamics is False
    assert info_053.dynamics_fn is None

    info_100 = registry.get_model_info("MEC-100")
    assert info_100.has_dynamics is False
    assert info_100.dynamics_fn is None


# ============================================================
# 14. get_scipy_solve
# ============================================================

def test_get_scipy_solve(registry):
    """get_scipy_solve 能加载 scipy_solve.py。"""
    solver = registry.get_scipy_solve("MEC-001")
    assert hasattr(solver, "simulate")


# ============================================================
# 15. state_dim 一致性
# ============================================================

def test_state_dim_consistency(registry):
    """state_dim 与 state_vars 长度一致（variable 除外）。"""
    for mid, info in registry._models.items():
        if info.state_dim == "variable":
            assert info.state_vars == ["variable"]
        elif info.state_dim is None:
            assert info.state_vars == []
        else:
            assert info.state_dim == len(info.state_vars), \
                f"{mid}: state_dim={info.state_dim} != len(state_vars)={len(info.state_vars)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
