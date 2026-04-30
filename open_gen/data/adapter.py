from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from open_gen.data.schema import CANONICAL_FIELDS, CanonicalTrajectory


def get_by_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


@dataclass
class TrajectoryFieldMapping:
    fields: dict[str, str]
    defaults: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.fields) - CANONICAL_FIELDS
        if unknown:
            raise ValueError(f"Unsupported canonical fields: {sorted(unknown)}")
        required = {"left_wrist_image", "right_wrist_image", "action"}
        missing = required - set(self.fields)
        if missing:
            raise ValueError(f"Missing required mappings: {sorted(missing)}")


class CanonicalTrajectoryAdapter:
    def __init__(self, mapping: TrajectoryFieldMapping, *, action_dim: int, proprio_dim: int, force_dim: int) -> None:
        self.mapping = mapping
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.force_dim = force_dim

    def adapt(self, sample: dict[str, Any]) -> CanonicalTrajectory:
        data: dict[str, Any] = {}
        for key, path in self.mapping.fields.items():
            try:
                data[key] = get_by_path(sample, path)
            except KeyError:
                if key in self.mapping.defaults:
                    data[key] = self.mapping.defaults[key]
                else:
                    raise
        for key, value in self.mapping.defaults.items():
            data.setdefault(key, value)
        trajectory = CanonicalTrajectory(**data)
        return trajectory.validate(action_dim=self.action_dim, proprio_dim=self.proprio_dim, force_dim=self.force_dim)
