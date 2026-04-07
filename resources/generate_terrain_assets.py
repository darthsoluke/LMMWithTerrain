import argparse
import struct
from pathlib import Path

import numpy as np

from terrain_config import load_terrain_config


LMM_TO_METERS = 5.6444 / 100.0
PFNN_HEIGHTMAP_HSCALE = 3.937007874 * LMM_TO_METERS
PFNN_HEIGHTMAP_VSCALE = 3.0 * LMM_TO_METERS
PFNN_HEIGHTMAP_PATH = Path(__file__).resolve().parents[2] / "PFNN" / "demo" / "heightmaps" / "hmap_007_smooth.txt"
PFNN_HEIGHTMAP_ORIGIN_X = -320.0 * LMM_TO_METERS
PFNN_HEIGHTMAP_ORIGIN_Z = 680.0 * LMM_TO_METERS
GRID_RESOLUTION_X = 240
GRID_RESOLUTION_Z = 240
DEFAULT_ENVIRONMENT_BOXES = [
    (1.75, 0.0, 1.25, 1.2, 1.0, 2.2, 0),
    (-1.35, 0.0, -1.10, 1.8, 1.0, 1.6, 0),
    (0.15, 0.0, 2.05, 1.1, 1.0, 1.0, 0),
]


def load_database(filename):
    with open(filename, "rb") as f:
        nframes, nbones = struct.unpack("II", f.read(8))
        bone_positions = np.frombuffer(
            f.read(nframes * nbones * 3 * 4), dtype=np.float32
        ).reshape([nframes, nbones, 3])

        nframes, nbones = struct.unpack("II", f.read(8))
        _bone_velocities = np.frombuffer(
            f.read(nframes * nbones * 3 * 4), dtype=np.float32
        ).reshape([nframes, nbones, 3])

        nframes, nbones = struct.unpack("II", f.read(8))
        bone_rotations = np.frombuffer(
            f.read(nframes * nbones * 4 * 4), dtype=np.float32
        ).reshape([nframes, nbones, 4])

        nframes, nbones = struct.unpack("II", f.read(8))
        _bone_angular_velocities = np.frombuffer(
            f.read(nframes * nbones * 3 * 4), dtype=np.float32
        ).reshape([nframes, nbones, 3])

        nbones = struct.unpack("I", f.read(4))[0]
        _bone_parents = np.frombuffer(f.read(nbones * 4), dtype=np.int32)

        nranges = struct.unpack("I", f.read(4))[0]
        range_starts = np.frombuffer(f.read(nranges * 4), dtype=np.int32)

        nranges = struct.unpack("I", f.read(4))[0]
        range_stops = np.frombuffer(f.read(nranges * 4), dtype=np.int32)

    return {
        "bone_positions": bone_positions,
        "bone_rotations": bone_rotations,
        "range_starts": range_starts,
        "range_stops": range_stops,
    }


def clamp_index(range_starts, range_stops, frame, offset):
    for start, stop in zip(range_starts, range_stops):
        if start <= frame < stop:
            return np.clip(frame + offset, start, stop - 1)
    raise RuntimeError(f"frame {frame} not found in any range")


def build_future_indices(range_starts, range_stops, offset, nframes):
    indices = np.zeros(nframes, dtype=np.int32)
    for start, stop in zip(range_starts, range_stops):
        frames = np.arange(start, stop, dtype=np.int32)
        indices[start:stop] = np.clip(frames + offset, start, stop - 1)
    return indices


def quat_mul_vec(q, v):
    q_xyz = q[..., 1:]
    qw = q[..., :1]
    t = 2.0 * np.cross(q_xyz, v)
    return v + qw * t + np.cross(q_xyz, t)


class PFNNTerrainFunction:
    def __init__(self, path, hscale, vscale, origin_x, origin_z):
        rows = []
        with open(path) as f:
            for line in f:
                row = [float(x) for x in line.split() if x]
                if row:
                    rows.append(row)

        self.heightmap = np.asarray(rows, dtype=np.float32)
        self.width = self.heightmap.shape[0]
        self.height = self.heightmap.shape[1]
        self.offset = float(self.heightmap.mean())
        self.hscale = hscale
        self.vscale = vscale
        self.origin_x = origin_x
        self.origin_z = origin_z

    def sample(self, x, z):
        px = ((x + self.origin_x) / self.hscale) + 0.5 * self.width
        pz = ((z + self.origin_z) / self.hscale) + 0.5 * self.height

        x0 = np.clip(np.floor(px).astype(np.int32), 0, self.width - 1)
        x1 = np.clip(np.ceil(px).astype(np.int32), 0, self.width - 1)
        z0 = np.clip(np.floor(pz).astype(np.int32), 0, self.height - 1)
        z1 = np.clip(np.ceil(pz).astype(np.int32), 0, self.height - 1)

        ax = px - np.floor(px)
        az = pz - np.floor(pz)

        s0 = self.vscale * (self.heightmap[x0, z0] - self.offset)
        s1 = self.vscale * (self.heightmap[x1, z0] - self.offset)
        s2 = self.vscale * (self.heightmap[x0, z1] - self.offset)
        s3 = self.vscale * (self.heightmap[x1, z1] - self.offset)

        return (s0 * (1.0 - ax) + s1 * ax) * (1.0 - az) + (s2 * (1.0 - ax) + s3 * ax) * az


def load_environment_boxes(path):
    if path is None or not path.exists():
        return list(DEFAULT_ENVIRONMENT_BOXES)

    boxes = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            values = stripped.split()
            if len(values) != 7:
                continue

            px, py, pz, sx, sy, sz = [float(x) for x in values[:6]]
            traversable = int(values[6])
            boxes.append((px, py, pz, sx, sy, sz, traversable))

    return boxes


def write_environment_boxes(boxes, output_path):
    with open(output_path, "w") as f:
        f.write("# x y z sx sy sz traversable\n")
        for box in boxes:
            px, py, pz, sx, sy, sz, traversable = box
            f.write(f"{px:.6f} {py:.6f} {pz:.6f} {sx:.6f} {sy:.6f} {sz:.6f} {traversable:d}\n")


def box_signed_distance_xz(sample_points, box):
    px, _py, pz, sx, _sy, sz, _traversable = box
    center = np.array([px, pz], dtype=np.float32)
    half_extent = 0.5 * np.array([sx, sz], dtype=np.float32)

    delta = np.abs(sample_points - center[None, :]) - half_extent[None, :]
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
    inside = np.minimum(np.max(delta, axis=-1), 0.0)
    return outside + inside


def sample_obstacle_sdf(sample_points, boxes, clamp_distance):
    sdf = np.full(sample_points.shape[0], clamp_distance, dtype=np.float32)
    found_obstacle = False

    for box in boxes:
        if box[6] != 0:
            continue

        sdf = np.minimum(sdf, box_signed_distance_xz(sample_points, box))
        found_obstacle = True

    if not found_obstacle:
        return sdf

    return np.clip(sdf, -clamp_distance, clamp_distance)


def write_feature_file(features, output_path):
    with open(output_path, "wb") as f:
        f.write(struct.pack("II", features.shape[0], features.shape[1]))
        f.write(features.astype(np.float32).tobytes())


def export_environment_features(db, terr_func, boxes, output_path, config, include_obstacle_sdf=True):
    root_positions = db["bone_positions"][:, 0]
    root_rotations = db["bone_rotations"][:, 0]
    range_starts = db["range_starts"]
    range_stops = db["range_stops"]

    nframes = root_positions.shape[0]
    nterrain_features = len(config.environment_horizons) * 3
    nsdf_features = nterrain_features if include_obstacle_sdf else 0
    environment = np.zeros((nframes, nterrain_features + nsdf_features), dtype=np.float32)
    root_heights = terr_func.sample(root_positions[:, 0], root_positions[:, 2])

    terrain_cursor = 0
    sdf_cursor = nterrain_features

    for future_offset in config.environment_horizons:
        future_indices = build_future_indices(range_starts, range_stops, future_offset, nframes)
        center = root_positions[future_indices]
        right = quat_mul_vec(root_rotations[future_indices], np.array([1.0, 0.0, 0.0], dtype=np.float32))

        sample_right = center + config.strip_half_width * right
        sample_center = center
        sample_left = center - config.strip_half_width * right

        flat_x = np.concatenate([sample_right[:, 0], sample_center[:, 0], sample_left[:, 0]])
        flat_z = np.concatenate([sample_right[:, 2], sample_center[:, 2], sample_left[:, 2]])
        flat_heights = terr_func.sample(flat_x, flat_z)
        flat_heights = flat_heights.reshape(3, nframes).transpose(1, 0) - root_heights[:, None]
        environment[:, terrain_cursor:terrain_cursor + 3] = flat_heights
        terrain_cursor += 3

        if include_obstacle_sdf:
            flat_points = np.concatenate([
                np.stack([sample_right[:, 0], sample_right[:, 2]], axis=1),
                np.stack([sample_center[:, 0], sample_center[:, 2]], axis=1),
                np.stack([sample_left[:, 0], sample_left[:, 2]], axis=1),
            ], axis=0)
            flat_sdf = sample_obstacle_sdf(flat_points, boxes, config.sdf_clamp_distance)
            environment[:, sdf_cursor:sdf_cursor + 3] = flat_sdf.reshape(3, nframes).transpose(1, 0)
            sdf_cursor += 3

    write_feature_file(environment, output_path)


def export_render_grid(terr_func, output_path):
    x_min = float(-0.5 * terr_func.width * terr_func.hscale - terr_func.origin_x)
    x_max = float(+0.5 * terr_func.width * terr_func.hscale - terr_func.origin_x)
    z_min = float(-0.5 * terr_func.height * terr_func.hscale - terr_func.origin_z)
    z_max = float(+0.5 * terr_func.height * terr_func.hscale - terr_func.origin_z)

    xs = np.linspace(x_min, x_max, GRID_RESOLUTION_X, dtype=np.float32)
    zs = np.linspace(z_min, z_max, GRID_RESOLUTION_Z, dtype=np.float32)
    grid = np.zeros((GRID_RESOLUTION_Z, GRID_RESOLUTION_X), dtype=np.float32)

    for zi, z in enumerate(zs):
        grid[zi, :] = terr_func.sample(xs, np.full_like(xs, z))

    with open(output_path, "wb") as f:
        f.write(struct.pack("II", GRID_RESOLUTION_X, GRID_RESOLUTION_Z))
        f.write(struct.pack("ffff", x_min, x_max, z_min, z_max))
        f.write(grid.astype(np.float32).tobytes())


def parse_args():
    resources_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=resources_dir / "database.bin")
    parser.add_argument("--output-dir", type=Path, default=resources_dir)
    parser.add_argument("--boxes-file", type=Path, default=resources_dir / "environment_boxes.txt")
    parser.add_argument("--heightmap-path", type=Path, default=PFNN_HEIGHTMAP_PATH)
    parser.add_argument("--terrain-config", type=Path, default=resources_dir / "terrain_sampling_config.txt")
    parser.add_argument(
        "--export-rocky-features",
        action="store_true",
        help="also write terrain_features_rocky.bin with terrain-only strips for offline comparisons",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_terrain_config(args.terrain_config)

    db = load_database(args.database)
    terr_func = PFNNTerrainFunction(
        args.heightmap_path,
        PFNN_HEIGHTMAP_HSCALE,
        PFNN_HEIGHTMAP_VSCALE,
        PFNN_HEIGHTMAP_ORIGIN_X,
        PFNN_HEIGHTMAP_ORIGIN_Z,
    )
    boxes = load_environment_boxes(args.boxes_file)

    export_environment_features(db, terr_func, boxes, args.output_dir / "terrain_features.bin", config, include_obstacle_sdf=True)
    export_render_grid(terr_func, args.output_dir / "pfnn_terrain_rocky_grid.bin")
    write_environment_boxes(boxes, args.output_dir / "environment_boxes.txt")

    if args.export_rocky_features:
        export_environment_features(
            db,
            terr_func,
            boxes,
            args.output_dir / "terrain_features_rocky.bin",
            config,
            include_obstacle_sdf=False,
        )


if __name__ == "__main__":
    main()
