from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FrameSampling = Literal["max", "uniform", "geometric"]


@dataclass
class CurriculumStage:
    name: str
    max_past_frames: int
    min_past_frames: int = 1
    frame_sampling: FrameSampling = "max"
    token_budget: int = 32768
    until_step: int | None = None
    until_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.max_past_frames < self.min_past_frames:
            raise ValueError(
                f"stage {self.name}: max_past_frames ({self.max_past_frames}) "
                f"< min_past_frames ({self.min_past_frames})"
            )
        if self.frame_sampling not in ("max", "uniform", "geometric"):
            raise ValueError(f"stage {self.name}: unknown frame_sampling {self.frame_sampling}")
        if self.until_step is not None and self.until_epoch is not None:
            raise ValueError(f"stage {self.name}: set only one of until_step/until_epoch")


@dataclass
class CurriculumConfig:
    enabled: bool
    unit: Literal["step", "epoch"]
    stages: list[CurriculumStage]


def parse_curriculum_config(payload: dict[str, Any] | None) -> CurriculumConfig:
    if not payload or not payload.get("enabled", False):
        return CurriculumConfig(enabled=False, unit="step", stages=[])
    unit = payload.get("unit", "step")
    if unit not in ("step", "epoch"):
        raise ValueError(f"curriculum.unit must be step or epoch, got {unit}")
    stages_payload = payload.get("stages") or []
    if not stages_payload:
        raise ValueError("curriculum.enabled=true requires at least one stage")
    stages = [CurriculumStage(**stage) for stage in stages_payload]
    return CurriculumConfig(enabled=True, unit=unit, stages=stages)


class CurriculumScheduler:
    def __init__(self, config: CurriculumConfig, *, fallback: CurriculumStage) -> None:
        self.config = config
        self.fallback = fallback
        self._current_index = 0

    @property
    def stages(self) -> list[CurriculumStage]:
        return self.config.stages if self.config.enabled else [self.fallback]

    def stage_index_for(self, *, step: int, epoch: int) -> int:
        if not self.config.enabled:
            return 0
        unit_value = step if self.config.unit == "step" else epoch
        for idx, stage in enumerate(self.config.stages):
            limit = stage.until_step if self.config.unit == "step" else stage.until_epoch
            if limit is None or unit_value < limit:
                return idx
        return len(self.config.stages) - 1

    def stage_for(self, *, step: int, epoch: int) -> CurriculumStage:
        return self.stages[self.stage_index_for(step=step, epoch=epoch)]

    def update(self, *, step: int, epoch: int) -> tuple[bool, CurriculumStage]:
        new_index = self.stage_index_for(step=step, epoch=epoch)
        changed = new_index != self._current_index
        self._current_index = new_index
        return changed, self.stages[new_index]

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self._current_index]
