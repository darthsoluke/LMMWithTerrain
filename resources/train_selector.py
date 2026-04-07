from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[override]
        def add_scalar(self, *args, **kwargs):
            pass
        def add_scalars(self, *args, **kwargs):
            pass
        def flush(self):
            pass
        def close(self):
            pass

from train_common import (
    cleanup_distributed,
    distributed_mean,
    init_distributed,
    is_main_process,
    load_features,
    save_network,
    unwrap_model,
)


class Selector(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super().__init__()
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        return self.linear2(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--niter", type=int, default=100000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--targets", default="hybrid_retrieval_targets.npz")
    parser.add_argument("--stats-out", default="selector_train_stats.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    X = load_features("./features.bin")["features"].astype(np.float32)
    targets = np.load(args.targets)
    valid_frames = targets["valid_frames"].astype(np.int64)
    topk_indices = targets["topk_indices"].astype(np.int64)
    topk_costs = targets["topk_costs"].astype(np.float32)

    nfeatures = X.shape[1]
    top_k = topk_indices.shape[1]
    input_size = nfeatures * 3 + 1

    ctx = init_distributed(args.device, args.gpu)
    device = ctx.device
    np.random.seed(1234 + ctx.rank)
    torch.manual_seed(1234 + ctx.rank)
    if is_main_process(ctx):
        print(f"Using device: {device}, world_size={ctx.world_size}")

    feature_scale = np.maximum(X.std(axis=0), 1e-8)
    cost_scale = max(float(topk_costs[np.isfinite(topk_costs)].std()), 1e-6)

    selector_mean_in = torch.as_tensor(
        np.concatenate([X.mean(axis=0), X.mean(axis=0), X.mean(axis=0), np.array([topk_costs[np.isfinite(topk_costs)].mean()], dtype=np.float32)]).astype(np.float32),
        device=device,
    )
    selector_std_in = torch.as_tensor(
        np.concatenate([feature_scale, feature_scale, feature_scale, np.array([cost_scale], dtype=np.float32)]).astype(np.float32),
        device=device,
    )
    selector_mean_out = torch.zeros([1], dtype=torch.float32, device=device)
    selector_std_out = torch.ones([1], dtype=torch.float32, device=device)

    preload_device = device if device.type == "cuda" else None
    X_t = torch.as_tensor(X, device=preload_device)
    network_selector = Selector(input_size).to(device)
    if ctx.enabled:
        network_selector = nn.parallel.DistributedDataParallel(
            network_selector,
            device_ids=[ctx.local_rank] if device.type == "cuda" else None,
            output_device=ctx.local_rank if device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(network_selector.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    writer = SummaryWriter()
    rolling_loss = None

    if is_main_process(ctx):
        with open(args.stats_out, "w") as f:
            json.dump(
                {
                    "valid_frames": int(len(valid_frames)),
                    "top_k": int(top_k),
                    "input_size": int(input_size),
                    "world_size": int(ctx.world_size),
                },
                f,
                indent=2,
            )

    for i in range(args.niter):
        rows = np.random.randint(0, len(valid_frames), size=[args.batchsize])
        frames = valid_frames[rows]
        cand_idx = topk_indices[rows]
        cand_cost = topk_costs[rows]

        query = X_t[frames].to(device)
        curr = query
        cand = X_t[cand_idx].to(device)
        cost = torch.as_tensor(cand_cost, device=device)

        query_rep = query[:, None, :].expand(-1, top_k, -1)
        curr_rep = curr[:, None, :].expand(-1, top_k, -1)
        selector_in = torch.cat([query_rep, curr_rep, cand, cost[..., None]], dim=-1)

        optimizer.zero_grad()
        scores = network_selector((selector_in - selector_mean_in) / selector_std_in).squeeze(-1)
        target = torch.zeros([args.batchsize], dtype=torch.long, device=device)
        loss_ce = F.cross_entropy(scores, target)
        margin = 0.05
        loss_rank = torch.mean(F.relu(margin + scores[:, 1:] - scores[:, :1]))
        loss = loss_ce + 0.5 * loss_rank
        loss.backward()
        optimizer.step()

        mean_loss = distributed_mean(loss.item(), ctx)

        if is_main_process(ctx):
            writer.add_scalar("selector/loss", mean_loss, i)
            writer.add_scalars("selector/loss_terms", {"ce": loss_ce.item(), "rank": loss_rank.item()}, i)
            rolling_loss = mean_loss if rolling_loss is None else rolling_loss * 0.99 + mean_loss * 0.01

        if i % 10 == 0 and is_main_process(ctx):
            print(f"\rIter: {i:7d} Loss: {rolling_loss:5.3f}", end="")

        if i % args.save_every == 0 and is_main_process(ctx):
            model = unwrap_model(network_selector)
            save_network(
                "selector.bin",
                [model.linear0, model.linear1, model.linear2],
                selector_mean_in,
                selector_std_in,
                selector_mean_out,
                selector_std_out,
            )
        if i % args.save_every == 0:
            scheduler.step()
        if i % args.save_every == 0 and ctx.enabled:
            dist.barrier()

    if is_main_process(ctx):
        print()
    cleanup_distributed(ctx)
