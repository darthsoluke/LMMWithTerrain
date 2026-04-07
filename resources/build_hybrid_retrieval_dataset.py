from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sklearn.neighbors import BallTree

from train_common import (
    build_valid_spans,
    default_frame_mask_path,
    init_distributed,
    is_main_process,
    load_database,
    load_features,
    load_frame_mask,
    save_valid_spans,
    valid_frame_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build top-K ordinary-MM retrieval targets for hybrid learned controller.")
    parser.add_argument("--database", default="./database.bin")
    parser.add_argument("--features", default="./features.bin")
    parser.add_argument("--frame-mask", default=None)
    parser.add_argument("--output", default="hybrid_retrieval_targets.npz")
    parser.add_argument("--stats-out", default="hybrid_retrieval_stats.json")
    parser.add_argument("--valid-spans-out", default="hybrid_valid_spans.json")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ignore-surrounding", type=int, default=20)
    parser.add_argument("--query-batch", type=int, default=4096)
    parser.add_argument("--db-chunk", type=int, default=65536)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def shard_slice(length: int, rank: int, world_size: int) -> tuple[int, int]:
    start = (length * rank) // world_size
    stop = (length * (rank + 1)) // world_size
    return start, stop


def merge_topk(best_costs, best_indices, cand_costs, cand_indices, k):
    merged_costs = torch.cat([best_costs, cand_costs], dim=1)
    merged_indices = torch.cat([best_indices, cand_indices], dim=1)
    top_costs, top_order = torch.topk(merged_costs, k=k, dim=1, largest=False)
    top_indices = torch.gather(merged_indices, 1, top_order)
    return top_costs, top_indices


def gpu_topk_retrieval(features, valid_idx, query_frames, top_k, ignore_surrounding, query_batch, db_chunk, device, rank):
    feats = torch.as_tensor(features[valid_idx], dtype=torch.float32, device=device)
    feats_sq = torch.sum(feats * feats, dim=1)

    all_indices = []
    all_costs = []

    for start in range(0, len(query_frames), query_batch):
        stop = min(start + query_batch, len(query_frames))
        frames = query_frames[start:stop]
        q = torch.as_tensor(features[frames], dtype=torch.float32, device=device)
        q_sq = torch.sum(q * q, dim=1)

        best_costs = torch.full((q.shape[0], top_k), float("inf"), device=device)
        best_indices = torch.full((q.shape[0], top_k), -1, dtype=torch.long, device=device)

        frame_tensor = torch.as_tensor(frames, dtype=torch.long, device=device)
        for db_start in range(0, len(valid_idx), db_chunk):
            db_stop = min(db_start + db_chunk, len(valid_idx))
            chunk = feats[db_start:db_stop]
            chunk_sq = feats_sq[db_start:db_stop]
            dist = q_sq[:, None] + chunk_sq[None, :] - 2.0 * torch.matmul(q, chunk.T)

            chunk_idx = torch.as_tensor(valid_idx[db_start:db_stop], dtype=torch.long, device=device)
            invalid = torch.abs(chunk_idx[None, :] - frame_tensor[:, None]) < ignore_surrounding
            dist[invalid] = float("inf")

            cand_costs, cand_local = torch.topk(dist, k=top_k, dim=1, largest=False)
            cand_indices = chunk_idx[cand_local]
            best_costs, best_indices = merge_topk(best_costs, best_indices, cand_costs, cand_indices, top_k)

        if rank == 0 and start % max(query_batch * 16, 1) == 0:
            print(f"retrieval progress {start}/{len(query_frames)}")

        all_indices.append(best_indices.cpu().numpy().astype(np.int32))
        all_costs.append(best_costs.cpu().numpy().astype(np.float32))

    return np.concatenate(all_indices, axis=0), np.concatenate(all_costs, axis=0)


def cpu_topk_retrieval(features, valid_idx, query_frames, top_k, ignore_surrounding, query_batch):
    tree = BallTree(features[valid_idx])
    query_k = min(max(top_k + 32, top_k), len(valid_idx))

    topk_indices = np.full((len(query_frames), top_k), -1, dtype=np.int32)
    topk_costs = np.full((len(query_frames), top_k), np.inf, dtype=np.float32)

    for start in range(0, len(query_frames), query_batch):
        stop = min(start + query_batch, len(query_frames))
        frames = query_frames[start:stop]
        distances, neighbors = tree.query(features[frames], k=query_k)

        for row, frame in enumerate(frames):
            insert = 0
            for dist, neigh_pos in zip(distances[row], neighbors[row]):
                candidate = valid_idx[neigh_pos]
                if abs(int(candidate) - int(frame)) < ignore_surrounding:
                    continue
                topk_indices[start + row, insert] = int(candidate)
                topk_costs[start + row, insert] = float(dist)
                insert += 1
                if insert == top_k:
                    break

            if insert == 0:
                topk_indices[start + row, 0] = int(frame)
                topk_costs[start + row, 0] = 0.0
                insert = 1

            while insert < top_k:
                topk_indices[start + row, insert] = topk_indices[start + row, insert - 1]
                topk_costs[start + row, insert] = topk_costs[start + row, insert - 1]
                insert += 1

    return topk_indices, topk_costs


def main():
    args = parse_args()
    ctx = init_distributed(args.device, args.gpu)
    device = ctx.device

    db = load_database(args.database)
    features = load_features(args.features)["features"].astype(np.float32)
    frame_mask_path = args.frame_mask or default_frame_mask_path(args.database)
    frame_valid = load_frame_mask(frame_mask_path)

    nframes = features.shape[0]
    if frame_valid.shape[0] != nframes:
        raise RuntimeError(f"frame mask size mismatch: {frame_valid.shape[0]} vs {nframes}")

    valid_idx = valid_frame_indices(frame_valid)
    spans = build_valid_spans(db["range_starts"], db["range_stops"], frame_valid)
    if is_main_process(ctx):
        save_valid_spans(args.valid_spans_out, spans)

    shard_start, shard_stop = shard_slice(len(valid_idx), ctx.rank, ctx.world_size)
    shard_frames = valid_idx[shard_start:shard_stop]

    if device.type == "cuda":
        shard_topk_indices, shard_topk_costs = gpu_topk_retrieval(
            features,
            valid_idx,
            shard_frames,
            args.top_k,
            args.ignore_surrounding,
            args.query_batch,
            args.db_chunk,
            device,
            ctx.rank,
        )
    else:
        shard_topk_indices, shard_topk_costs = cpu_topk_retrieval(
            features,
            valid_idx,
            shard_frames,
            args.top_k,
            args.ignore_surrounding,
            args.query_batch,
        )

    shard_path = Path(f"{args.output}.rank{ctx.rank}.npz")
    np.savez_compressed(
        shard_path,
        valid_frames=shard_frames,
        topk_indices=shard_topk_indices,
        topk_costs=shard_topk_costs,
    )

    if ctx.enabled:
        dist.barrier()

    if is_main_process(ctx):
        topk_indices = np.full((len(valid_idx), args.top_k), -1, dtype=np.int32)
        topk_costs = np.full((len(valid_idx), args.top_k), np.inf, dtype=np.float32)
        teacher_index = np.full(nframes, -1, dtype=np.int32)

        for rank in range(ctx.world_size):
            part = np.load(f"{args.output}.rank{rank}.npz")
            frames = part["valid_frames"].astype(np.int64)
            start = (len(valid_idx) * rank) // ctx.world_size
            stop = start + len(frames)
            topk_indices[start:stop] = part["topk_indices"]
            topk_costs[start:stop] = part["topk_costs"]
            teacher_index[frames] = part["topk_indices"][:, 0]
            Path(f"{args.output}.rank{rank}.npz").unlink(missing_ok=True)

        np.savez_compressed(
            args.output,
            valid_frames=valid_idx,
            topk_indices=topk_indices,
            topk_costs=topk_costs,
            teacher_index=teacher_index,
        )

        with open(args.stats_out, "w") as f:
            json.dump(
                {
                    "nframes": int(nframes),
                    "valid_frames": int(len(valid_idx)),
                    "top_k": int(args.top_k),
                    "ignore_surrounding": int(args.ignore_surrounding),
                    "query_batch": int(args.query_batch),
                    "db_chunk": int(args.db_chunk),
                    "world_size": int(ctx.world_size),
                    "device_type": device.type,
                },
                f,
                indent=2,
            )

    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
