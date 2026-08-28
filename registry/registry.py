"""Model Registry — 发现、查询、动态加载 MEC 模型。

不修改任何现有 model.py。通过 importlib 动态加载，使用独立模块名
避免 sys.modules['model'] 冲突。

用法:
    from registry import ModelRegistry

    reg = ModelRegistry()
    reg.list_models()                     # → ['MEC-001', ..., 'MEC-100']
    reg.list_categories()                # → ['point-particle-mechanics', ...]
    info = reg.get_model_info("MEC-010") # → ModelEntry(...)
    model = reg.get_model("MEC-010")     # → 原始 module 对象
    reg.has_interface("MEC-053", "dynamics")  # → False
    reg.search(has_dynamics=True, has_energy=True)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelEntry:
    """单个模型的注册表条目（与 catalog.json 一一对应）。"""
    id: str
    name: str
    directory: str
    category: str
    description: str
    has_dynamics: bool
    has_validate: bool
    has_analytical: bool
    has_energy: bool
    has_scipy_solve: bool
    dynamics_fn: str | None
    energy_fn: str | None
    validate_fn: str | None
    analytical_fn: str | None
    model_path: str
    scipy_solve_path: str
    test_path: str | None
    state_dim: int | str | None
    state_vars: list[str] = field(default_factory=list)


class ModelRegistry:
    """模型注册表：发现、查询、动态加载 38 个 MEC 模型。"""

    def __init__(self, catalog_path: str | None = None):
        """加载 catalog.json。

        Args:
            catalog_path: catalog.json 的路径。默认为 registry/catalog.json（相对于本文件）。
        """
        if catalog_path is None:
            catalog_path = Path(__file__).parent / "catalog.json"
        self._catalog_path = str(catalog_path)

        with open(self._catalog_path, "r", encoding="utf-8") as f:
            self._catalog = json.load(f)

        self._base_path = self._catalog.get("base_path", "models/mechanics")
        self._models: dict[str, ModelEntry] = {}
        self._raw: dict[str, dict] = {}

        for entry in self._catalog["models"]:
            me = ModelEntry(**entry)
            self._models[me.id] = me
            self._raw[me.id] = entry

        # 项目根目录 = registry/ 的上两级
        self._project_root = Path(__file__).resolve().parent.parent
        self._mechanics_dir = self._project_root / self._base_path

    def list_models(self, category: str | None = None) -> list[str]:
        """列出所有模型 ID。可选按分类过滤。"""
        if category is None:
            return list(self._models.keys())
        return [mid for mid, m in self._models.items() if m.category == category]

    def list_categories(self) -> list[str]:
        """列出所有分类名。"""
        seen = []
        for m in self._models.values():
            if m.category not in seen:
                seen.append(m.category)
        return seen

    def get_model_info(self, model_id: str) -> ModelEntry:
        """查询模型元数据。不加载 model.py。"""
        if model_id not in self._models:
            raise KeyError(f"未知模型 ID: {model_id}")
        return self._models[model_id]

    def get_model(self, model_id: str):
        """动态加载模型的 model.py，返回原始 Python module。

        使用独立模块名（如 physics_models_mec_001_model）避免 sys.modules 冲突。
        """
        info = self.get_model_info(model_id)
        full_path = self._mechanics_dir / info.model_path

        # 构造唯一模块名
        safe_id = model_id.replace("-", "_").lower()
        module_name = f"physics_models_{safe_id}_model"

        spec = importlib.util.spec_from_file_location(module_name, str(full_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {full_path}")

        module = importlib.util.module_from_spec(spec)

        # 临时将模型目录加入 sys.path，使 model.py 内部的
        # `from model import ...` 等裸导入在 scipy_solve 中也可用
        model_dir = str(full_path.parent)
        old_path = sys.path[:]
        sys.path.insert(0, model_dir)

        try:
            spec.loader.exec_module(module)
        finally:
            sys.path = old_path

        # 注册到 sys.modules 以便后续引用
        sys.modules[module_name] = module

        return module

    def get_scipy_solve(self, model_id: str):
        """动态加载模型的 scipy_solve.py，返回原始 Python module。"""
        info = self.get_model_info(model_id)
        full_path = self._mechanics_dir / info.scipy_solve_path

        safe_id = model_id.replace("-", "_").lower()
        module_name = f"physics_models_{safe_id}_scipy_solve"

        spec = importlib.util.spec_from_file_location(module_name, str(full_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {full_path}")

        module = importlib.util.module_from_spec(spec)

        # scipy_solve.py 依赖 model.py，需要先加载 model 到 sys.modules['model']
        model_dir = str(full_path.parent)
        old_path = sys.path[:]
        sys.path.insert(0, model_dir)

        # 先加载对应的 model.py 作为 'model' 模块
        model_path = self._mechanics_dir / info.model_path
        model_spec = importlib.util.spec_from_file_location("model", str(model_path))
        model_module = importlib.util.module_from_spec(model_spec)

        old_model = sys.modules.get("model")
        sys.modules["model"] = model_module

        try:
            model_spec.loader.exec_module(model_module)
            spec.loader.exec_module(module)
        finally:
            sys.path = old_path
            if old_model is not None:
                sys.modules["model"] = old_model
            else:
                sys.modules.pop("model", None)

        sys.modules[module_name] = module
        return module

    def has_interface(self, model_id: str, interface: str) -> bool:
        """检查模型是否具有某个接口。

        Args:
            interface: 'dynamics' | 'energy' | 'validate' | 'analytical'
        """
        info = self.get_model_info(model_id)
        mapping = {
            "dynamics": info.has_dynamics,
            "energy": info.has_energy,
            "validate": info.has_validate,
            "analytical": info.has_analytical,
        }
        if interface not in mapping:
            raise ValueError(f"未知接口名: {interface}。可选: {list(mapping.keys())}")
        return mapping[interface]

    def search(self, **kwargs) -> list[str]:
        """按属性过滤模型，返回匹配的模型 ID 列表。

        示例:
            reg.search(has_dynamics=True, has_energy=True)
            reg.search(category="oscillatory-systems")
        """
        results = []
        for mid, m in self._models.items():
            match = True
            for key, value in kwargs.items():
                if not hasattr(m, key) or getattr(m, key) != value:
                    match = False
                    break
            if match:
                results.append(mid)
        return results
