from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TerrainGrid:
    width: int
    height: int
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    data: np.ndarray

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            width, height = struct.unpack("II", f.read(8))
            x_min, x_max, z_min, z_max = struct.unpack("ffff", f.read(16))
            data = np.frombuffer(f.read(width * height * 4), dtype=np.float32, count=width * height).reshape([height, width])
        return cls(width, height, x_min, x_max, z_min, z_max, data)


@dataclass
class EvalScene:
    name: str
    start_x: float
    start_z: float
    start_yaw: float
    frames: int
    script_name: str


def load_boxes(path: Path):
    boxes = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            x, y, z, sx, sy, sz, traversable = stripped.split()
            boxes.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "sx": float(sx),
                    "sy": float(sy),
                    "sz": float(sz),
                    "traversable": int(traversable),
                }
            )
    return boxes


def finite_difference(grid: TerrainGrid):
    dx = (grid.x_max - grid.x_min) / max(grid.width - 1, 1)
    dz = (grid.z_max - grid.z_min) / max(grid.height - 1, 1)
    grad_x = np.gradient(grid.data, axis=1) / max(dx, 1e-5)
    grad_z = np.gradient(grid.data, axis=0) / max(dz, 1e-5)
    slope = np.sqrt(grad_x ** 2 + grad_z ** 2)

    rough = np.zeros_like(grid.data)
    radius = 2
    for z in range(grid.height):
        z0 = max(z - radius, 0)
        z1 = min(z + radius + 1, grid.height)
        for x in range(grid.width):
            x0 = max(x - radius, 0)
            x1 = min(x + radius + 1, grid.width)
            rough[z, x] = grid.data[z0:z1, x0:x1].std()
    return grad_x, grad_z, slope, rough


def grid_index_to_world(grid: TerrainGrid, ix: int, iz: int):
    x = grid.x_min + (grid.x_max - grid.x_min) * ix / max(grid.width - 1, 1)
    z = grid.z_min + (grid.z_max - grid.z_min) * iz / max(grid.height - 1, 1)
    return x, z


def pick_scene_cells(grid: TerrainGrid):
    grad_x, grad_z, slope, rough = finite_difference(grid)
    center_bias = np.zeros_like(slope)
    xs = np.linspace(-1.0, 1.0, grid.width)[None, :]
    zs = np.linspace(-1.0, 1.0, grid.height)[:, None]
    center_bias = np.sqrt(xs ** 2 + zs ** 2)

    flat_score = slope + 0.5 * rough + 0.2 * center_bias
    flat_iz, flat_ix = np.unravel_index(np.argmin(flat_score), flat_score.shape)

    uphill_score = np.where(slope > 0.025, slope - 0.5 * rough, -1e9)
    uphill_iz, uphill_ix = np.unravel_index(np.argmax(uphill_score), uphill_score.shape)

    rough_score = rough + 0.25 * slope
    rough_iz, rough_ix = np.unravel_index(np.argmax(rough_score), rough_score.shape)

    return {
        "flat": (flat_ix, flat_iz, grad_x[flat_iz, flat_ix], grad_z[flat_iz, flat_ix]),
        "uphill": (uphill_ix, uphill_iz, grad_x[uphill_iz, uphill_ix], grad_z[uphill_iz, uphill_ix]),
        "rough": (rough_ix, rough_iz, grad_x[rough_iz, rough_ix], grad_z[rough_iz, rough_ix]),
    }


def write_constant_script(path: Path, frames: int, left_x: float, left_z: float, gait: float):
    with path.open("w") as f:
        for _ in range(frames):
            f.write(f"{left_x:.3f} {left_z:.3f} 0.000 0.000 {gait:.3f} 0\n")


def write_obstacle_script(path: Path, frames: int):
    with path.open("w") as f:
        for i in range(frames):
            if i < frames // 3:
                left_x = 0.35
            elif i < 2 * frames // 3:
                left_x = -0.35
            else:
                left_x = 0.0
            f.write(f"{left_x:.3f} 1.000 0.000 0.000 0.000 0\n")


def create_eval_scenes(grid: TerrainGrid, boxes, output_dir: Path):
    cells = pick_scene_cells(grid)
    scenes = []

    flat_x, flat_z = grid_index_to_world(grid, cells["flat"][0], cells["flat"][1])
    uphill_x, uphill_z = grid_index_to_world(grid, cells["uphill"][0], cells["uphill"][1])
    rough_x, rough_z = grid_index_to_world(grid, cells["rough"][0], cells["rough"][1])

    uphill_yaw = math.atan2(cells["uphill"][2], cells["uphill"][3] + 1e-8)
    downhill_yaw = uphill_yaw + math.pi
    rough_yaw = math.atan2(cells["rough"][2], cells["rough"][3] + 1e-8)

    obstacle = next((b for b in boxes if b["traversable"] == 0), boxes[0])
    obstacle_start_x = obstacle["x"]
    obstacle_start_z = obstacle["z"] - 4.0

    scripts_dir = output_dir / "eval_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    write_constant_script(scripts_dir / "flat.csv", 720, 0.0, 1.0, 0.0)
    write_constant_script(scripts_dir / "uphill.csv", 720, 0.0, 1.0, 0.0)
    write_constant_script(scripts_dir / "downhill.csv", 720, 0.0, 1.0, 0.0)
    write_constant_script(scripts_dir / "rough.csv", 720, 0.0, 1.0, 0.0)
    write_obstacle_script(scripts_dir / "obstacle.csv", 720)

    scenes.append(EvalScene("flat", flat_x, flat_z, 0.0, 720, "flat.csv"))
    scenes.append(EvalScene("uphill", uphill_x, uphill_z, uphill_yaw, 720, "uphill.csv"))
    scenes.append(EvalScene("downhill", uphill_x, uphill_z, downhill_yaw, 720, "downhill.csv"))
    scenes.append(EvalScene("rough", rough_x, rough_z, rough_yaw, 720, "rough.csv"))
    scenes.append(EvalScene("obstacle", obstacle_start_x, obstacle_start_z, 0.0, 720, "obstacle.csv"))

    with (output_dir / "eval_scene_defs.json").open("w") as f:
        json.dump([asdict(scene) for scene in scenes], f, indent=2)

    return scenes


def run_controller(controller_path: Path, workdir: Path, scene: EvalScene, mode: str, script_path: Path, trace_path: Path):
    cmd = [
        "xvfb-run",
        "-a",
        str(controller_path),
        "--eval-mode",
        mode,
        "--eval-script",
        str(script_path),
        "--eval-trace",
        str(trace_path),
        "--eval-start-x",
        str(scene.start_x),
        "--eval-start-z",
        str(scene.start_z),
        "--eval-start-yaw",
        str(scene.start_yaw),
        "--eval-frames",
        str(scene.frames),
    ]
    subprocess.run(cmd, cwd=workdir, check=True)


def load_trace(path: Path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (float(v) if k not in {"mode"} else v) for k, v in row.items()})
    return rows


def summarize_trace(rows):
    arr = lambda key: np.asarray([row[key] for row in rows], dtype=np.float32)
    return {
        "frames": len(rows),
        "contact_height_error_mean": float(arr("contact_height_error").mean()),
        "contact_slip_mean": float(arr("contact_slip").mean()),
        "terrain_penetrations": int(arr("terrain_penetrations").sum()),
        "projector_correction_mean": float(arr("projector_correction").mean()),
        "stepper_drift_mean": float(arr("stepper_drift").mean()),
        "root_xyz": np.stack([arr("root_x"), arr("root_y"), arr("root_z")], axis=1),
    }


def save_plots(output_dir: Path, scene_name: str, ordinary, learned):
    plot_dir = output_dir / "eval_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    baseline = ordinary["root_xyz"]
    candidate = learned["root_xyz"]
    frames = np.arange(min(len(baseline), len(candidate)))

    plt.figure(figsize=(8, 4))
    plt.plot(frames, baseline[: len(frames), 1], label="ordinary root_y")
    plt.plot(frames, candidate[: len(frames), 1], label="learned root_y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{scene_name}_root_height.png")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate learned vs ordinary terrain-aware motion matching.")
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terrain-grid", type=Path, required=True)
    parser.add_argument("--boxes-file", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = args.output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    grid = TerrainGrid.load(args.terrain_grid)
    boxes = load_boxes(args.boxes_file)
    scenes = create_eval_scenes(grid, boxes, args.output_dir)

    results = {}
    summary_lines = ["# Terrain Evaluation", ""]

    for scene in scenes:
        ordinary_trace = traces_dir / f"{scene.name}_ordinary.csv"
        learned_trace = traces_dir / f"{scene.name}_learned.csv"
        script_path = args.output_dir / "eval_scripts" / scene.script_name

        run_controller(args.controller, args.workdir, scene, "ordinary", script_path, ordinary_trace)
        run_controller(args.controller, args.workdir, scene, "learned", script_path, learned_trace)

        ordinary_rows = load_trace(ordinary_trace)
        learned_rows = load_trace(learned_trace)

        ordinary_summary = summarize_trace(ordinary_rows)
        learned_summary = summarize_trace(learned_rows)
        count = min(len(ordinary_summary["root_xyz"]), len(learned_summary["root_xyz"]))
        root_error = np.linalg.norm(
            learned_summary["root_xyz"][:count, [0, 2]] - ordinary_summary["root_xyz"][:count, [0, 2]],
            axis=1,
        )

        save_plots(args.output_dir, scene.name, ordinary_summary, learned_summary)

        results[scene.name] = {
            "ordinary": {k: v for k, v in ordinary_summary.items() if k != "root_xyz"},
            "learned": {k: v for k, v in learned_summary.items() if k != "root_xyz"},
            "root_planar_error_mean": float(root_error.mean()),
            "root_planar_error_max": float(root_error.max()),
        }

        summary_lines.append(f"## {scene.name}")
        summary_lines.append(f"- ordinary contact height error: {results[scene.name]['ordinary']['contact_height_error_mean']:.4f}")
        summary_lines.append(f"- learned contact height error: {results[scene.name]['learned']['contact_height_error_mean']:.4f}")
        summary_lines.append(f"- ordinary contact slip: {results[scene.name]['ordinary']['contact_slip_mean']:.4f}")
        summary_lines.append(f"- learned contact slip: {results[scene.name]['learned']['contact_slip_mean']:.4f}")
        summary_lines.append(f"- learned projector correction: {results[scene.name]['learned']['projector_correction_mean']:.4f}")
        summary_lines.append(f"- learned stepper drift: {results[scene.name]['learned']['stepper_drift_mean']:.4f}")
        summary_lines.append(f"- root planar error vs ordinary: {results[scene.name]['root_planar_error_mean']:.4f}")
        summary_lines.append("")

    with (args.output_dir / "eval_metrics.json").open("w") as f:
        json.dump(results, f, indent=2)

    with (args.output_dir / "eval_summary.md").open("w") as f:
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
