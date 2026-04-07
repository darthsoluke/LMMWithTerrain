from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path


ABLATIONS = [
    ("teacher_selector_teacher_residual", "teacher", "teacher"),
    ("teacher_selector_learned_residual", "teacher", "learned"),
    ("learned_selector_zero_residual", "learned", "zero"),
    ("learned_selector_learned_residual", "learned", "learned"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run 4-way hybrid runtime ablation debug matrix.")
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--scene-defs", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def trace_has_nonfinite(path: Path):
    if not path.exists():
        return True
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k, v in row.items():
            if k in {"mode", "selector_mode", "residual_mode"}:
                continue
            if not math.isfinite(float(v)):
                return True
    return False


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = args.output_dir / "traces"
    dumps_dir = args.output_dir / "failure_dumps"
    traces_dir.mkdir(parents=True, exist_ok=True)
    dumps_dir.mkdir(parents=True, exist_ok=True)

    scenes = json.loads(args.scene_defs.read_text())
    results = {}

    for ablation_name, selector_mode, residual_mode in ABLATIONS:
        results[ablation_name] = {}
        for scene in scenes:
            script_path = args.scripts_dir / scene["script_name"]
            trace_path = traces_dir / f"{ablation_name}_{scene['name']}.csv"
            tag = f"{ablation_name}_{scene['name']}"
            cmd = [
                "xvfb-run",
                "-a",
                str(args.controller),
                "--eval-mode",
                "learned",
                "--eval-script",
                str(script_path),
                "--eval-trace",
                str(trace_path),
                "--eval-tag",
                tag,
                "--hybrid-selector-mode",
                selector_mode,
                "--hybrid-residual-mode",
                residual_mode,
                "--failure-dump-dir",
                str(dumps_dir),
                "--eval-start-x",
                str(scene["start_x"]),
                "--eval-start-z",
                str(scene["start_z"]),
                "--eval-start-yaw",
                str(scene["start_yaw"]),
                "--eval-frames",
                str(scene["frames"]),
            ]
            subprocess.run(cmd, cwd=args.workdir, check=True)

            failure_dump = dumps_dir / f"{tag}_failure_dump.txt"
            results[ablation_name][scene["name"]] = {
                "trace": str(trace_path),
                "failure_dump": str(failure_dump) if failure_dump.exists() else None,
                "has_nonfinite": trace_has_nonfinite(trace_path),
            }

    (args.output_dir / "ablation_results.json").write_text(json.dumps(results, indent=2))

    lines = ["# Hybrid Runtime Ablation Matrix", ""]
    for ablation_name, scenes_result in results.items():
        lines.append(f"## {ablation_name}")
        for scene_name, info in scenes_result.items():
            lines.append(
                f"- {scene_name}: nonfinite={info['has_nonfinite']} dump={info['failure_dump'] or 'none'}"
            )
        lines.append("")
    (args.output_dir / "ablation_results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
