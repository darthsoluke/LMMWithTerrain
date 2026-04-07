from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    build_valid_spans,
    cleanup_distributed,
    control_feature_slice,
    default_frame_mask_path,
    distributed_mean,
    init_distributed,
    is_main_process,
    load_database,
    load_features,
    load_frame_mask,
    load_latent,
    sample_window_batch,
    save_network,
    save_valid_spans,
    unwrap_model,
    valid_window_starts,
)


class ResidualStepper(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        return self.linear2(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--niter", type=int, default=100000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--frame-mask", default=None)
    parser.add_argument("--targets", default="hybrid_retrieval_targets.npz")
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--valid-spans-out", default="hybrid_stepper_valid_spans.json")
    parser.add_argument("--stats-out", default="hybrid_stepper_stats.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    database = load_database("./database.bin")
    X = load_features("./features.bin")["features"].astype(np.float32)
    Z = load_latent("./latent.bin")["latent"].astype(np.float32)
    control_slice = control_feature_slice(X.shape[1])
    C = X[:, control_slice].astype(np.float32)
    targets = np.load(args.targets)
    teacher_anchor = targets["teacher_index"].astype(np.int64)

    nframes = X.shape[0]
    nfeatures = X.shape[1]
    nlatent = Z.shape[1]
    ncontrol = C.shape[1]

    frame_mask_path = args.frame_mask or default_frame_mask_path("./database.bin")
    frame_valid = load_frame_mask(frame_mask_path)
    if frame_valid.shape[0] != nframes:
        raise RuntimeError(f"frame mask size mismatch: {frame_valid.shape[0]} vs {nframes}")

    anchor_valid = teacher_anchor >= 0
    frame_valid = np.logical_and(frame_valid, anchor_valid)
    spans = build_valid_spans(database["range_starts"], database["range_stops"], frame_valid)
    valid_starts = valid_window_starts(spans, args.window)
    save_valid_spans(args.valid_spans_out, spans)
    if len(valid_starts) == 0:
        raise RuntimeError("no valid residual-stepper spans found")

    ctx = init_distributed(args.device, args.gpu)
    device = ctx.device
    np.random.seed(1234 + ctx.rank)
    torch.manual_seed(1234 + ctx.rank)
    if is_main_process(ctx):
        print(f"Using device: {device}, world_size={ctx.world_size}")

    anchor_X = X[teacher_anchor.clip(min=0)]
    anchor_Z = Z[teacher_anchor.clip(min=0)]
    residual_X = X - anchor_X
    residual_Z = Z - anchor_Z

    RX_scale = np.maximum(residual_X.std(axis=0), 1e-8)
    RZ_scale = np.maximum(residual_Z.std(axis=0), 1e-8)
    C_scale = np.maximum(C.std(axis=0), 1e-8)
    AX_scale = np.maximum(anchor_X.std(axis=0), 1e-8)
    AZ_scale = np.maximum(anchor_Z.std(axis=0), 1e-8)

    mean_in = torch.as_tensor(
        np.concatenate([
            residual_X.mean(axis=0),
            residual_Z.mean(axis=0),
            C.mean(axis=0),
            anchor_X.mean(axis=0),
            anchor_Z.mean(axis=0),
        ]).astype(np.float32),
        device=device,
    )
    std_in = torch.as_tensor(
        np.concatenate([RX_scale, RZ_scale, C_scale, AX_scale, AZ_scale]).astype(np.float32),
        device=device,
    )

    residual_X_delta = residual_X[1:] - residual_X[:-1]
    residual_Z_delta = residual_Z[1:] - residual_Z[:-1]
    mean_out = torch.as_tensor(
        np.concatenate([
            (residual_X_delta / (1.0 / 60.0)).mean(axis=0),
            (residual_Z_delta / (1.0 / 60.0)).mean(axis=0),
        ]).astype(np.float32),
        device=device,
    )
    std_out = torch.as_tensor(
        np.concatenate([
            np.maximum((residual_X_delta / (1.0 / 60.0)).std(axis=0), 1e-8),
            np.maximum((residual_Z_delta / (1.0 / 60.0)).std(axis=0), 1e-8),
        ]).astype(np.float32),
        device=device,
    )

    preload_device = device if device.type == "cuda" else None
    X_t = torch.as_tensor(X, device=preload_device)
    Z_t = torch.as_tensor(Z, device=preload_device)
    C_t = torch.as_tensor(C, device=preload_device)
    anchor_X_t = torch.as_tensor(anchor_X, device=preload_device)
    anchor_Z_t = torch.as_tensor(anchor_Z, device=preload_device)
    residual_X_t = torch.as_tensor(residual_X, device=preload_device)
    residual_Z_t = torch.as_tensor(residual_Z, device=preload_device)

    network = ResidualStepper(
        nfeatures + nlatent + ncontrol + nfeatures + nlatent,
        nfeatures + nlatent,
    ).to(device)
    if ctx.enabled:
        network = nn.parallel.DistributedDataParallel(
            network,
            device_ids=[ctx.local_rank] if device.type == "cuda" else None,
            output_device=ctx.local_rank if device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    writer = SummaryWriter()
    rolling_loss = None
    dt = 1.0 / 60.0

    if is_main_process(ctx):
        with open(args.stats_out, "w") as f:
            json.dump(
                {
                    "valid_spans": len(spans),
                    "window": args.window,
                    "control_dim": ncontrol,
                    "input_size": nfeatures + nlatent + ncontrol + nfeatures + nlatent,
                    "world_size": int(ctx.world_size),
                },
                f,
                indent=2,
            )

    for i in range(args.niter):
        batch = torch.as_tensor(sample_window_batch(valid_starts, args.window, args.batchsize), dtype=torch.long, device=preload_device)

        Xgnd = X_t[batch]
        Zgnd = Z_t[batch]
        Cgnd = C_t[batch]
        AX = anchor_X_t[batch]
        AZ = anchor_Z_t[batch]
        RX = residual_X_t[batch]
        RZ = residual_Z_t[batch]

        optimizer.zero_grad()

        RXtil = [RX[:, 0]]
        RZtil = [RZ[:, 0]]
        Xabs = [Xgnd[:, 0]]
        Zabs = [Zgnd[:, 0]]

        for step in range(1, args.window):
            inp = torch.cat([
                RXtil[-1],
                RZtil[-1],
                Cgnd[:, step - 1],
                AX[:, step - 1],
                AZ[:, step - 1],
            ], dim=-1)
            output = network((inp - mean_in) / std_in) * std_out + mean_out
            RXnext = RXtil[-1] + dt * output[:, :nfeatures]
            RZnext = RZtil[-1] + dt * output[:, nfeatures:]
            RXtil.append(RXnext)
            RZtil.append(RZnext)
            Xabs.append(AX[:, step - 1] + RXnext)
            Zabs.append(AZ[:, step - 1] + RZnext)

        Xabs = torch.stack(Xabs, dim=1)
        Zabs = torch.stack(Zabs, dim=1)
        RXtil = torch.stack(RXtil, dim=1)
        RZtil = torch.stack(RZtil, dim=1)

        loss_xabs = torch.mean(2.0 * torch.abs(Xgnd - Xabs))
        loss_zabs = torch.mean(7.5 * torch.abs(Zgnd - Zabs))
        loss_rx = torch.mean(0.5 * torch.abs(RX - RXtil))
        loss_rz = torch.mean(1.0 * torch.abs(RZ - RZtil))
        loss = loss_xabs + loss_zabs + loss_rx + loss_rz
        loss.backward()
        optimizer.step()

        mean_loss = distributed_mean(loss.item(), ctx)

        if is_main_process(ctx):
            writer.add_scalar("hybrid_stepper/loss", mean_loss, i)
            writer.add_scalars(
                "hybrid_stepper/loss_terms",
                {
                    "xabs": loss_xabs.item(),
                    "zabs": loss_zabs.item(),
                    "rx": loss_rx.item(),
                    "rz": loss_rz.item(),
                },
                i,
            )

            rolling_loss = mean_loss if rolling_loss is None else rolling_loss * 0.99 + mean_loss * 0.01
        if i % 10 == 0 and is_main_process(ctx):
            print(f"\rIter: {i:7d} Loss: {rolling_loss:5.3f}", end="")

        if i % args.save_every == 0 and is_main_process(ctx):
            model = unwrap_model(network)
            save_network(
                "residual_stepper.bin",
                [model.linear0, model.linear1, model.linear2],
                mean_in,
                std_in,
                mean_out,
                std_out,
            )
        if i % args.save_every == 0:
            scheduler.step()
        if i % args.save_every == 0 and ctx.enabled:
            torch.distributed.barrier()

    if is_main_process(ctx):
        print()
    cleanup_distributed(ctx)
