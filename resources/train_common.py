from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


MOTION_MATCH_BASE_FEATURE_COUNT = 31


def load_database(filename):
    with open(filename, "rb") as f:
        nframes, nbones = struct.unpack("II", f.read(8))
        bone_positions = np.frombuffer(
            f.read(nframes * nbones * 3 * 4), dtype=np.float32, count=nframes * nbones * 3
        ).reshape([nframes, nbones, 3])

        nframes, nbones = struct.unpack("II", f.read(8))
        bone_velocities = np.frombuffer(
            f.read(nframes * nbones * 3 * 4), dtype=np.float32, count=nframes * nbones * 3
        ).reshape([nframes, nbones, 3])

        nframes, nbones = struct.unpack("II", f.read(8))
        bone_rotations = np.frombuffer(
            f.read(nframes * nbones * 4 * 4), dtype=np.float32, count=nframes * nbones * 4
        ).reshape([nframes, nbones, 4])

        nframes, nbones = struct.unpack("II", f.read(8))
        bone_angular_velocities = np.frombuffer(
            f.read(nframes * nbones * 3 * 4), dtype=np.float32, count=nframes * nbones * 3
        ).reshape([nframes, nbones, 3])

        nbones = struct.unpack("I", f.read(4))[0]
        bone_parents = np.frombuffer(f.read(nbones * 4), dtype=np.int32, count=nbones).reshape([nbones])

        nranges = struct.unpack("I", f.read(4))[0]
        range_starts = np.frombuffer(f.read(nranges * 4), dtype=np.int32, count=nranges).reshape([nranges])

        nranges = struct.unpack("I", f.read(4))[0]
        range_stops = np.frombuffer(f.read(nranges * 4), dtype=np.int32, count=nranges).reshape([nranges])

        nframes, ncontacts = struct.unpack("II", f.read(8))
        contact_states = np.frombuffer(f.read(nframes * ncontacts), dtype=np.int8, count=nframes * ncontacts).reshape([nframes, ncontacts])

    return {
        "bone_positions": bone_positions,
        "bone_rotations": bone_rotations,
        "bone_velocities": bone_velocities,
        "bone_angular_velocities": bone_angular_velocities,
        "bone_parents": bone_parents,
        "range_starts": range_starts,
        "range_stops": range_stops,
        "contact_states": contact_states,
    }


def load_features(filename):
    with open(filename, "rb") as f:
        nframes, nfeatures = struct.unpack("II", f.read(8))
        features = np.frombuffer(f.read(nframes * nfeatures * 4), dtype=np.float32, count=nframes * nfeatures).reshape([nframes, nfeatures])

        nfeatures = struct.unpack("I", f.read(4))[0]
        features_offset = np.frombuffer(f.read(nfeatures * 4), dtype=np.float32, count=nfeatures).reshape([nfeatures])

        nfeatures = struct.unpack("I", f.read(4))[0]
        features_scale = np.frombuffer(f.read(nfeatures * 4), dtype=np.float32, count=nfeatures).reshape([nfeatures])

    return {
        "features": features,
        "features_offset": features_offset,
        "features_scale": features_scale,
    }


def load_environment_features(filename):
    with open(filename, "rb") as f:
        nframes, nfeatures = struct.unpack("II", f.read(8))
        features = np.frombuffer(
            f.read(nframes * nfeatures * 4),
            dtype=np.float32,
            count=nframes * nfeatures,
        ).reshape([nframes, nfeatures])
    return {
        "features": features,
    }


def load_latent(filename):
    with open(filename, "rb") as f:
        nframes, nfeatures = struct.unpack("II", f.read(8))
        latent = np.frombuffer(f.read(nframes * nfeatures * 4), dtype=np.float32, count=nframes * nfeatures).reshape([nframes, nfeatures])

    return {
        "latent": latent,
    }


def save_network(filename, layers, mean_in, std_in, mean_out, std_out):
    with torch.no_grad():
        with open(filename, "wb") as f:
            f.write(struct.pack("I", mean_in.shape[0]) + mean_in.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack("I", std_in.shape[0]) + std_in.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack("I", mean_out.shape[0]) + mean_out.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack("I", std_out.shape[0]) + std_out.cpu().numpy().astype(np.float32).ravel().tobytes())
            f.write(struct.pack("I", len(layers)))
            for layer in layers:
                f.write(struct.pack("II", *layer.weight.T.shape) + layer.weight.T.cpu().numpy().astype(np.float32).ravel().tobytes())
                f.write(struct.pack("I", *layer.bias.shape) + layer.bias.cpu().numpy().astype(np.float32).ravel().tobytes())


def load_frame_mask(filename):
    with open(filename, "rb") as f:
        nframes = struct.unpack("I", f.read(4))[0]
        mask = np.frombuffer(f.read(nframes), dtype=np.uint8, count=nframes).reshape([nframes]).astype(bool)
    return mask


def save_frame_mask(filename, mask):
    mask = np.asarray(mask, dtype=np.uint8).reshape([-1])
    with open(filename, "wb") as f:
        f.write(struct.pack("I", mask.shape[0]))
        f.write(mask.tobytes())


def default_frame_mask_path(database_path):
    return str(Path(database_path).with_name("frame_mask.bin"))


def load_or_default_frame_mask(database_path, nframes):
    mask_path = Path(default_frame_mask_path(database_path))
    if mask_path.exists():
        mask = load_frame_mask(mask_path)
        if mask.shape[0] != nframes:
            raise RuntimeError(f"frame mask size mismatch: {mask.shape[0]} vs {nframes}")
        return mask
    return np.ones(nframes, dtype=bool)


def save_valid_spans(path, spans):
    data = [
        {"start": int(start), "stop": int(stop), "length": int(stop - start)}
        for start, stop in spans
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_valid_spans(range_starts, range_stops, frame_valid):
    spans = []
    for start, stop in zip(range_starts, range_stops):
        i = int(start)
        stop = int(stop)
        while i < stop:
            while i < stop and not frame_valid[i]:
                i += 1
            span_start = i
            while i < stop and frame_valid[i]:
                i += 1
            if span_start < i:
                spans.append((span_start, i))
    return spans


def valid_window_starts(spans, window):
    starts = []
    for start, stop in spans:
        max_start = stop - window
        for frame in range(start, max_start + 1):
            starts.append(frame)
    return np.asarray(starts, dtype=np.int64)


def sample_window_batch(valid_starts, window, batchsize):
    chosen = np.random.randint(0, len(valid_starts), size=[batchsize])
    starts = valid_starts[chosen]
    offsets = np.arange(window, dtype=np.int64)[None, :]
    return starts[:, None] + offsets


def valid_frame_indices(frame_valid):
    return np.flatnonzero(frame_valid).astype(np.int64)


def control_feature_slice(nfeatures):
    if nfeatures < MOTION_MATCH_BASE_FEATURE_COUNT:
        raise ValueError(f"expected at least {MOTION_MATCH_BASE_FEATURE_COUNT} features, got {nfeatures}")
    return slice(15, nfeatures)


def environment_feature_slice(nfeatures):
    if nfeatures < MOTION_MATCH_BASE_FEATURE_COUNT:
        raise ValueError(f"expected at least {MOTION_MATCH_BASE_FEATURE_COUNT} features, got {nfeatures}")
    return slice(MOTION_MATCH_BASE_FEATURE_COUNT, nfeatures)


def feature_group_weights(nfeatures, environment_weight=2.5, trajectory_direction_weight=1.5):
    weights = np.ones(nfeatures, dtype=np.float32)
    weights[21:27] = trajectory_direction_weight
    weights[27:31] = 1.25
    weights[MOTION_MATCH_BASE_FEATURE_COUNT:] = environment_weight
    return weights


def hard_mining_scores(features):
    env = features[:, environment_feature_slice(features.shape[1])]
    scores = np.zeros(features.shape[0], dtype=np.float32)
    if env.shape[1] == 0:
        return scores
    delta_prev = np.zeros_like(env)
    delta_next = np.zeros_like(env)
    delta_prev[1:] = np.abs(env[1:] - env[:-1])
    delta_next[:-1] = np.abs(env[:-1] - env[1:])
    scores = delta_prev.mean(axis=1) + delta_next.mean(axis=1)
    return scores


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

    def sample(self, x, z):
        x = np.asarray(x, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        px = (x - self.x_min) / max(self.x_max - self.x_min, 1e-5) * (self.width - 1)
        pz = (z - self.z_min) / max(self.z_max - self.z_min, 1e-5) * (self.height - 1)

        x0 = np.clip(np.floor(px).astype(np.int32), 0, self.width - 1)
        x1 = np.clip(np.ceil(px).astype(np.int32), 0, self.width - 1)
        z0 = np.clip(np.floor(pz).astype(np.int32), 0, self.height - 1)
        z1 = np.clip(np.ceil(pz).astype(np.int32), 0, self.height - 1)

        ax = px - np.floor(px)
        az = pz - np.floor(pz)

        s0 = self.data[z0, x0]
        s1 = self.data[z0, x1]
        s2 = self.data[z1, x0]
        s3 = self.data[z1, x1]
        return (s0 * (1.0 - ax) + s1 * ax) * (1.0 - az) + (s2 * (1.0 - ax) + s3 * ax) * az


@dataclass
class DistributedContext:
    enabled: bool
    device: torch.device
    rank: int
    world_size: int
    local_rank: int


def init_distributed(args_device=None, args_gpu=0) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args_gpu)))
    enabled = world_size > 1

    if args_device is not None:
        device = torch.device(args_device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank if enabled else args_gpu}")
    else:
        device = torch.device("cpu")

    torch.set_num_threads(1)
    if enabled:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return DistributedContext(
        enabled=enabled,
        device=device,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )


def is_main_process(ctx: DistributedContext) -> bool:
    return ctx.rank == 0


def distributed_mean(value: float, ctx: DistributedContext) -> float:
    if not ctx.enabled:
        return float(value)
    tensor = torch.tensor([value], dtype=torch.float32, device=ctx.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= ctx.world_size
    return float(tensor.item())


def barrier(ctx: DistributedContext) -> None:
    if ctx.enabled:
        dist.barrier()


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
