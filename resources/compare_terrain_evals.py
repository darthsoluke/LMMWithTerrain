from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare hybrid eval against pure learned and ordinary baseline.")
    parser.add_argument("--pure", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    pure = json.loads(args.pure.read_text())
    hybrid = json.loads(args.hybrid.read_text())

    report = {}
    lines = ["# Hybrid Vs Pure Terrain Evaluation", ""]

    for scene in sorted(hybrid.keys()):
        report[scene] = {
            "pure": pure[scene],
            "hybrid": hybrid[scene],
        }
        lines.append(f"## {scene}")
        lines.append(f"- ordinary contact height: {hybrid[scene]['ordinary']['contact_height_error_mean']:.4f}")
        lines.append(f"- pure learned contact height: {pure[scene]['learned']['contact_height_error_mean']:.4f}")
        lines.append(f"- hybrid learned contact height: {hybrid[scene]['learned']['contact_height_error_mean']:.4f}")
        lines.append(f"- ordinary slip: {hybrid[scene]['ordinary']['contact_slip_mean']:.4f}")
        lines.append(f"- pure learned slip: {pure[scene]['learned']['contact_slip_mean']:.4f}")
        lines.append(f"- hybrid learned slip: {hybrid[scene]['learned']['contact_slip_mean']:.4f}")
        lines.append(f"- pure root planar error: {pure[scene]['root_planar_error_mean']:.4f}")
        lines.append(f"- hybrid root planar error: {hybrid[scene]['root_planar_error_mean']:.4f}")
        lines.append("")

    args.output_json.write_text(json.dumps(report, indent=2))
    args.output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
