"""
ShiftLab Algorithm Registry.

Auto-discovers all BaseAlgorithm subclasses from the algorithms/ package tree.
Each algorithm is identified by a stable slug: family.task_type.ClassName (lowercased).
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.core.base_algorithm import BaseAlgorithm


class AlgorithmRegistry:
    """Singleton registry of all discovered algorithms."""

    _registry: dict[str, type["BaseAlgorithm"]] = {}
    _loaded: bool = False

    @classmethod
    def register(cls, algo_cls: type["BaseAlgorithm"]) -> type["BaseAlgorithm"]:
        """Decorator / direct call to register an algorithm class."""
        slug = cls._make_slug(algo_cls)
        cls._registry[slug] = algo_cls
        return algo_cls

    @classmethod
    def _make_slug(cls, algo_cls: type["BaseAlgorithm"]) -> str:
        return f"{algo_cls.family}.{algo_cls.task_type}.{algo_cls.__name__}".lower()

    @classmethod
    def load_all(cls) -> None:
        """Import all modules under backend.algorithms to trigger @register decorators."""
        if cls._loaded:
            return
        import backend.algorithms as algo_pkg
        cls._walk_package(algo_pkg)
        cls._loaded = True

    @classmethod
    def _walk_package(cls, package) -> None:
        prefix = package.__name__ + "."
        for _, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix):
            try:
                importlib.import_module(modname)
            except ImportError as e:
                # Optional dependency missing — skip gracefully
                print(f"[Registry] Skipped {modname}: {e}")

    @classmethod
    def get(cls, slug: str) -> type["BaseAlgorithm"]:
        cls.load_all()
        if slug not in cls._registry:
            raise KeyError(f"Algorithm '{slug}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[slug]

    @classmethod
    def list_all(cls) -> list[dict]:
        cls.load_all()
        return [
            {
                "slug": slug,
                "name": klass.name,
                "family": klass.family,
                "task_type": klass.task_type,
                "tags": klass.tags,
                "sklearn_compatible": klass.sklearn_compatible,
            }
            for slug, klass in sorted(cls._registry.items())
        ]

    @classmethod
    def list_by_family(cls, family: str) -> list[dict]:
        return [a for a in cls.list_all() if a["family"] == family]

    @classmethod
    def list_by_task(cls, task_type: str) -> list[dict]:
        return [a for a in cls.list_all() if a["task_type"] == task_type]

    @classmethod
    def search(cls, query: str) -> list[dict]:
        q = query.lower()
        return [
            a for a in cls.list_all()
            if q in a["name"].lower()
            or q in a["family"].lower()
            or q in a["task_type"].lower()
            or any(q in t.lower() for t in a["tags"])
        ]

    @classmethod
    def instantiate(cls, slug: str, random_state: int = 42, **hyperparams) -> "BaseAlgorithm":
        klass = cls.get(slug)
        return klass(random_state=random_state, **hyperparams)


registry = AlgorithmRegistry()
