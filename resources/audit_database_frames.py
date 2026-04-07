from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np

from train_common import load_database, load_environment_features, load_features, save_frame_mask


DEFAULT_EDGE_TRIM = 20
DEFAULT_SPEED_QUANTILE = 0.9995
DEFAULT_SPEED_SCALE = 1.25
DEFAULT_ANGULAR_QUANTILE = 0.9995
DEFAULT_POSITION_EPSILON = 1e-5


def quat_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.sum(a * b, axis=-1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def robust_limit(values: np.ndarray, quantile: float, scale: float = 1.0, minimum: float | None = None) -> float:
    limit = float(np.quantile(values, quantile)) * scale
    if minimum is not None:
        limit = max(limit, minimum)
    return limit


def build_reasons(
    db: dict,
    features: dict,
    terrain_features: np.ndarray,
    edge_trim: int,
    speed_quantile: float,
    speed_scale: float,
    angular_quantile: float,
) -> tuple[np.ndarray, list[list[str]], dict]:
    root_positions = db["bone_positions"][:, 0]
    root_velocities = db["bone_velocities"][:, 0]
    root_rotations = db["bone_rotations"][:, 0]
    root_angular_velocities = db["bone_angular_velocities"][:, 0]
    range_starts = db["range_starts"]
    range_stops = db["range_stops"]

    planar_speed = np.linalg.norm(root_velocities[:, [0, 2]], axis=1)
    angular_speed = np.linalg.norm(root_angular_velocities, axis=1)

    step_delta = np.zeros_like(planar_speed)
    step_delta[1:] = np.linalg.norm(root_positions[1:, [0, 2]] - root_positions[:-1, [0, 2]], axis=1)
    step_speed = step_delta * 60.0

    rotation_step = np.zeros_like(planar_speed)
    rotation_step[1:] = quat_angle_diff(root_rotations[1:], root_rotations[:-1]) * 60.0

    velocity_mismatch = np.abs(step_speed - planar_speed)

    speed_limit = robust_limit(planar_speed, speed_quantile, speed_scale, minimum=4.0)
    angular_limit = robust_limit(angular_speed, angular_quantile, scale=1.25, minimum=2.0)
    step_speed_limit = robust_limit(step_speed[1:], speed_quantile, scale=1.5, minimum=4.0)
    rotation_step_limit = robust_limit(rotation_step[1:], angular_quantile, scale=1.5, minimum=2.0)
    velocity_mismatch_limit = robust_limit(velocity_mismatch[1:], speed_quantile, scale=1.5, minimum=1.0)

    feature_norm = np.linalg.norm(features["features"], axis=1)
    terrain_norm = np.linalg.norm(terrain_features, axis=1)

    reasons: list[list[str]] = [[] for _ in range(len(planar_speed))]

    for start, stop in zip(range_starts, range_stops):
        for frame in range(start, min(stop, start + edge_trim)):
            reasons[frame].append("range_edge")
        for frame in range(max(start, stop - edge_trim), stop):
            reasons[frame].append("range_edge")

    for frame in range(len(planar_speed)):
        if planar_speed[frame] > speed_limit:
            reasons[frame].append("root_speed_outlier")
        if angular_speed[frame] > angular_limit:
            reasons[frame].append("root_angular_speed_outlier")

    for frame in range(1, len(planar_speed)):
        if step_speed[frame] > step_speed_limit:
            reasons[frame].append("root_step_speed_outlier")
        if rotation_step[frame] > rotation_step_limit:
            reasons[frame].append("root_rotation_discontinuity")
        if velocity_mismatch[frame] > velocity_mismatch_limit:
            reasons[frame].append("root_velocity_mismatch")

        duplicated_root = step_delta[frame] <= DEFAULT_POSITION_EPSILON and planar_speed[frame] > speed_limit
        if duplicated_root:
            reasons[frame].append("duplicate_root_extreme_velocity")

    for frame in range(len(planar_speed)):
        if not np.isfinite(feature_norm[frame]) or not np.isfinite(terrain_norm[frame]):
            reasons[frame].append("non_finite_feature")

    frame_valid = np.array([len(x) == 0 for x in reasons], dtype=np.uint8)

    stats = {
        "nframes": int(len(frame_valid)),
        "valid_frames": int(frame_valid.sum()),
        "invalid_frames": int(len(frame_valid) - frame_valid.sum()),
        "speed_limit": speed_limit,
        "angular_limit": angular_limit,
        "step_speed_limit": step_speed_limit,
        "rotation_step_limit": rotation_step_limit,
        "velocity_mismatch_limit": velocity_mismatch_limit,
        "per_frame": {
            "root_planar_speed_mean": float(planar_speed.mean()),
            "root_planar_speed_max": float(planar_speed.max()),
            "root_angular_speed_mean": float(angular_speed.mean()),
            "root_angular_speed_max": float(angular_speed.max()),
        },
        "per_range": [],
    }

    for range_index, (start, stop) in enumerate(zip(range_starts, range_stops)):
        range_mask = frame_valid[start:stop]
        stats["per_range"].append(
            {
                "range_index": int(range_index),
                "start": int(start),
                "stop": int(stop),
                "length": int(stop - start),
                "valid_frames": int(range_mask.sum()),
                "invalid_frames": int((stop - start) - range_mask.sum()),
                "root_planar_speed_mean": float(planar_speed[start:stop].mean()),
                "root_planar_speed_max": float(planar_speed[start:stop].max()),
                "root_angular_speed_mean": float(angular_speed[start:stop].mean()),
                "root_angular_speed_max": float(angular_speed[start:stop].max()),
            }
        )

    return frame_valid, reasons, {
        "stats": stats,
        "planar_speed": planar_speed,
        "angular_speed": angular_speed,
        "step_speed": step_speed,
        "rotation_step": rotation_step,
        "velocity_mismatch": velocity_mismatch,
        "feature_norm": feature_norm,
        "terrain_norm": terrain_norm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frame quality for terrain-aware LMM training.")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--terrain-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-trim", type=int, default=DEFAULT_EDGE_TRIM)
    parser.add_argument("--speed-quantile", type=float, default=DEFAULT_SPEED_QUANTILE)
    parser.add_argument("--speed-scale", type=float, default=DEFAULT_SPEED_SCALE)
    parser.add_argument("--angular-quantile", type=float, default=DEFAULT_ANGULAR_QUANTILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    db = load_database(args.database)
    features = load_features(args.features)
    terrain = load_environment_features(args.terrain_features)["features"]

    if features["features"].shape[0] != db["bone_positions"].shape[0]:
        raise RuntimeError("features.bin frame count does not match database.bin")
    if terrain.shape[0] != db["bone_positions"].shape[0]:
        raise RuntimeError("terrain_features.bin frame count does not match database.bin")

    frame_valid, reasons, metrics = build_reasons(
        db,
        features,
        terrain,
        edge_trim=args.edge_trim,
        speed_quantile=args.speed_quantile,
        speed_scale=args.speed_scale,
        angular_quantile=args.angular_quantile,
    )

    save_frame_mask(args.output_dir / "frame_mask.bin", frame_valid)

    with (args.output_dir / "frame_stats.json").open("w") as f:
        json.dump(metrics["stats"], f, indent=2)

    with (args.output_dir / "frame_audit.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "valid",
                "reasons",
                "root_planar_speed",
                "root_angular_speed",
                "root_step_speed",
                "root_rotation_step",
                "root_velocity_mismatch",
                "feature_norm",
                "terrain_norm",
            ]
        )
        for frame, reason_codes in enumerate(reasons):
            writer.writerow(
                [
                    frame,
                    int(frame_valid[frame]),
                    ";".join(reason_codes),
                    float(metrics["planar_speed"][frame]),
                    float(metrics["angular_speed"][frame]),
                    float(metrics["step_speed"][frame]),
                    float(metrics["rotation_step"][frame]),
                    float(metrics["velocity_mismatch"][frame]),
                    float(metrics["feature_norm"][frame]),
                    float(metrics["terrain_norm"][frame]),
                ]
            )


if __name__ == "__main__":
    main()
