from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TerrainSamplingConfig:
    environment_horizons: tuple[int, ...]
    matching_trajectory_horizons: tuple[int, ...]
    prediction_frame_step: int
    strip_half_width: float
    sdf_clamp_distance: float

    @property
    def environment_feature_count(self) -> int:
        return len(self.environment_horizons) * 3

    @property
    def total_environment_feature_count(self) -> int:
        return self.environment_feature_count * 2

    @property
    def prediction_sample_count(self) -> int:
        max_horizon = max(self.environment_horizons + self.matching_trajectory_horizons)
        return 1 + max_horizon // self.prediction_frame_step

    def environment_horizon_to_index(self, horizon: int) -> int:
        return horizon // self.prediction_frame_step


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def load_terrain_config(path: str | Path | None = None) -> TerrainSamplingConfig:
    if path is None:
        path = Path(__file__).resolve().with_name("terrain_sampling_config.txt")
    else:
        path = Path(path)

    values: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip()

    config = TerrainSamplingConfig(
        environment_horizons=_parse_int_tuple(values["environment_horizons"]),
        matching_trajectory_horizons=_parse_int_tuple(values["matching_trajectory_horizons"]),
        prediction_frame_step=int(values["prediction_frame_step"]),
        strip_half_width=float(values["strip_half_width"]),
        sdf_clamp_distance=float(values["sdf_clamp_distance"]),
    )

    if any(h <= 0 for h in config.environment_horizons):
        raise ValueError("environment_horizons must be positive")
    if any(h <= 0 for h in config.matching_trajectory_horizons):
        raise ValueError("matching_trajectory_horizons must be positive")
    if config.prediction_frame_step <= 0:
        raise ValueError("prediction_frame_step must be positive")
    if any(h % config.prediction_frame_step != 0 for h in config.environment_horizons + config.matching_trajectory_horizons):
        raise ValueError("all horizons must be divisible by prediction_frame_step")

    return config
