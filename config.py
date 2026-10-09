"""YAML configuration with attribute access and command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """A dictionary whose keys are also attributes, recursively.

    Example:
        >>> cfg = Config.from_yaml("configs/top0covernet_skwanda.yaml")
        >>> cfg.merge_overrides(["model.use_gtrm=false", "train.epochs=50"])
        >>> cfg.model.use_gtrm
        False
    """

    def __init__(self, data: dict | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, Config):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Config has no key '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle) or {})

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    def to_dict(self) -> dict:
        def unwrap(value):
            if isinstance(value, Config):
                return {k: unwrap(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unwrap(v) for v in value]
            return value

        return unwrap(self)

    def merge_overrides(self, overrides: Iterable[str] | None) -> "Config":
        """Apply ``dotted.key=value`` overrides; values are parsed as YAML."""
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"Override '{item}' must have the form key=value")
            dotted, raw = item.split("=", 1)
            node = self
            keys = dotted.split(".")
            for key in keys[:-1]:
                if key not in node or not isinstance(node[key], dict):
                    node[key] = Config()
                node = node[key]
            node[keys[-1]] = self._wrap(yaml.safe_load(raw))
        return self

    def clone(self) -> "Config":
        return Config(copy.deepcopy(self.to_dict()))
