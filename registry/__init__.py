"""physics-models Model Registry (v1.1)

提供模型发现、查询和动态加载功能。
不修改任何现有 model.py；不创建 ABC、wrapper 或 adapter。
"""

from .registry import ModelRegistry, ModelEntry

__all__ = ["ModelRegistry", "ModelEntry"]
