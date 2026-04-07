import sys
import struct
import argparse

import numpy as np
import tquat
import txform
import quat
import bvh

import matplotlib.pyplot as plt

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
    TerrainGrid,
    build_valid_spans,
    cleanup_distributed,
    default_frame_mask_path,
    distributed_mean,
    init_distributed,
    is_main_process,
    load_database,
    load_features,
    load_frame_mask,
    sample_window_batch,
    save_network,
    save_valid_spans,
    unwrap_model,
    valid_window_starts,
)


def safe_scalar_std(x, eps=1e-8):
    return max(float(x), eps)


def sample_terrain_height_torch(grid_values, x_min, x_max, z_min, z_max, xs, zs):
    width = grid_values.shape[1]
    height = grid_values.shape[0]

    px = (xs - x_min) / max(float(x_max - x_min), 1e-5) * (width - 1)
    pz = (zs - z_min) / max(float(z_max - z_min), 1e-5) * (height - 1)

    x0 = torch.clamp(torch.floor(px).long(), 0, width - 1)
    x1 = torch.clamp(torch.ceil(px).long(), 0, width - 1)
    z0 = torch.clamp(torch.floor(pz).long(), 0, height - 1)
    z1 = torch.clamp(torch.ceil(pz).long(), 0, height - 1)

    ax = px - torch.floor(px)
    az = pz - torch.floor(pz)

    s0 = grid_values[z0, x0]
    s1 = grid_values[z0, x1]
    s2 = grid_values[z1, x0]
    s3 = grid_values[z1, x1]
    return (s0 * (1.0 - ax) + s1 * ax) * (1.0 - az) + (s2 * (1.0 - ax) + s3 * ax) * az

# Networks

class Compressor(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=512):
        super(Compressor, self).__init__()
        
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        nbatch, nwindow = x.shape[:2]
        x = x.reshape([nbatch * nwindow, -1])
        x = F.elu(self.linear0(x))
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = self.linear3(x)
        return x.reshape([nbatch, nwindow, -1])
        
        
class Decompressor(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=512):
        super(Decompressor, self).__init__()
        
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        nbatch, nwindow = x.shape[:2]
        x = x.reshape([nbatch * nwindow, -1])
        x = F.relu(self.linear0(x))
        x = self.linear1(x)
        return x.reshape([nbatch, nwindow, -1])


# Training procedure

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--niter', type=int, default=100000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--frame-mask', default=None)
    parser.add_argument('--terrain-grid', default='pfnn_terrain_rocky_grid.bin')
    parser.add_argument('--valid-spans-out', default='decompressor_valid_spans.json')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Load data
    
    database = load_database('./database.bin')
    
    parents = database['bone_parents']
    contacts = database['contact_states']
    range_starts = database['range_starts']
    range_stops = database['range_stops']
    
    X = load_features('./features.bin')['features'].astype(np.float32)
    Ypos = database['bone_positions'].astype(np.float32)
    Yrot = database['bone_rotations'].astype(np.float32)
    Yvel = database['bone_velocities'].astype(np.float32)
    Yang = database['bone_angular_velocities'].astype(np.float32)
    
    nframes = Ypos.shape[0]
    nbones = Ypos.shape[1]
    nextra = contacts.shape[1]
    nfeatures = X.shape[1]
    nlatent = 32
    support_bones = np.array([4, 5, 8, 9], dtype=np.int64)
    support_contact_index = np.array([0, 0, 1, 1], dtype=np.int64)
    
    # Parameters
    
    seed = 1234
    batchsize = args.batchsize
    lr = args.lr
    niter = args.niter
    window = 2
    dt = 1.0 / 60.0

    ctx = init_distributed(args.device, args.gpu)
    device = ctx.device
    np.random.seed(seed + ctx.rank)
    torch.manual_seed(seed + ctx.rank)
    if is_main_process(ctx):
        print(f'Using device: {device}, world_size={ctx.world_size}')

    frame_mask_path = args.frame_mask or default_frame_mask_path('./database.bin')
    frame_valid = load_frame_mask(frame_mask_path)
    if frame_valid.shape[0] != nframes:
        raise RuntimeError(f'frame mask size mismatch: {frame_valid.shape[0]} vs {nframes}')
    spans = build_valid_spans(range_starts, range_stops, frame_valid)
    valid_starts = valid_window_starts(spans, window)
    if len(valid_starts) == 0:
        raise RuntimeError('no valid decompressor spans found')
    save_valid_spans(args.valid_spans_out, spans)

    terrain_grid = TerrainGrid.load(args.terrain_grid)
    
    # Compute world space
    
    Grot, Gpos, Gvel, Gang = quat.fk_vel(Yrot, Ypos, Yvel, Yang, parents)
    
    # Compute character space
    
    Qrot = quat.inv_mul(Grot[:,0:1], Grot)
    Qpos = quat.inv_mul_vec(Grot[:,0:1], Gpos - Gpos[:,0:1])
    Qvel = quat.inv_mul_vec(Grot[:,0:1], Gvel)
    Qang = quat.inv_mul_vec(Grot[:,0:1], Gang)
    
    # Compute transformation matrix
    
    Yxfm = quat.to_xform(Yrot)
    Qxfm = quat.to_xform(Qrot)
    
    # Compute two-column transformation matrix
    
    Ytxy = quat.to_xform_xy(Yrot).astype(np.float32)
    Qtxy = quat.to_xform_xy(Qrot).astype(np.float32)
    
    # Compute local root velocity
    
    Yrvel = quat.inv_mul_vec(Yrot[:,0], Yvel[:,0])
    Yrang = quat.inv_mul_vec(Yrot[:,0], Yang[:,0])
    
    # Compute extra outputs (contacts)
    
    Yextra = contacts.astype(np.float32)
    
    # Compute means/stds
    
    Ypos_scale = safe_scalar_std(Ypos[:,1:].std())
    Ytxy_scale = safe_scalar_std(Ytxy[:,1:].std())
    Yvel_scale = safe_scalar_std(Yvel[:,1:].std())
    Yang_scale = safe_scalar_std(Yang[:,1:].std())
    
    Qpos_scale = safe_scalar_std(Qpos[:,1:].std())
    Qtxy_scale = safe_scalar_std(Qtxy[:,1:].std())
    Qvel_scale = safe_scalar_std(Qvel[:,1:].std())
    Qang_scale = safe_scalar_std(Qang[:,1:].std())
    
    Yrvel_scale = safe_scalar_std(Yrvel.std())
    Yrang_scale = safe_scalar_std(Yrang.std())
    Yextra_scale = safe_scalar_std(Yextra.std())
    
    decompressor_mean_out = torch.as_tensor(np.hstack([
        Ypos[:,1:].mean(axis=0).ravel(),
        Ytxy[:,1:].mean(axis=0).ravel(),
        Yvel[:,1:].mean(axis=0).ravel(),
        Yang[:,1:].mean(axis=0).ravel(),
        Yrvel.mean(axis=0).ravel(),
        Yrang.mean(axis=0).ravel(),
        Yextra.mean(axis=0).ravel(),
    ]).astype(np.float32), device=device)
    
    decompressor_std_out = torch.as_tensor(np.hstack([
        Ypos[:,1:].std(axis=0).ravel(),
        Ytxy[:,1:].std(axis=0).ravel(),
        Yvel[:,1:].std(axis=0).ravel(),
        Yang[:,1:].std(axis=0).ravel(),
        Yrvel.std(axis=0).ravel(),
        Yrang.std(axis=0).ravel(),
        Yextra.std(axis=0).ravel(),
    ]).astype(np.float32), device=device)
    
    decompressor_mean_in = torch.zeros([nfeatures + nlatent], dtype=torch.float32, device=device)
    decompressor_std_in = torch.ones([nfeatures + nlatent], dtype=torch.float32, device=device)
    
    compressor_mean_in = torch.as_tensor(np.hstack([
        Ypos[:,1:].mean(axis=0).ravel(),
        Ytxy[:,1:].mean(axis=0).ravel(),
        Yvel[:,1:].mean(axis=0).ravel(),
        Yang[:,1:].mean(axis=0).ravel(),
        Qpos[:,1:].mean(axis=0).ravel(),
        Qtxy[:,1:].mean(axis=0).ravel(),
        Qvel[:,1:].mean(axis=0).ravel(),
        Qang[:,1:].mean(axis=0).ravel(),
        Yrvel.mean(axis=0).ravel(),
        Yrang.mean(axis=0).ravel(),
        Yextra.mean(axis=0).ravel(),
    ]).astype(np.float32), device=device)
    
    compressor_std_in = torch.as_tensor(np.hstack([
        np.repeat(Ypos_scale, (nbones-1)*3),
        np.repeat(Ytxy_scale, (nbones-1)*6),
        np.repeat(Yvel_scale, (nbones-1)*3),
        np.repeat(Yang_scale, (nbones-1)*3),
        np.repeat(Qpos_scale, (nbones-1)*3),
        np.repeat(Qtxy_scale, (nbones-1)*6),
        np.repeat(Qvel_scale, (nbones-1)*3),
        np.repeat(Qang_scale, (nbones-1)*3),
        np.repeat(Yrvel_scale, 3),
        np.repeat(Yrang_scale, 3),
        np.repeat(Yextra_scale, nextra),
    ]).astype(np.float32), device=device)
    
    # Make PyTorch tensors
    
    preload_device = device if device.type == 'cuda' else None
    Ypos = torch.as_tensor(Ypos, device=preload_device)
    Yrot = torch.as_tensor(Yrot, device=preload_device)
    Ytxy = torch.as_tensor(Ytxy, device=preload_device)
    Yvel = torch.as_tensor(Yvel, device=preload_device)
    Yang = torch.as_tensor(Yang, device=preload_device)
    
    Qpos = torch.as_tensor(Qpos, device=preload_device)
    Qrot = torch.as_tensor(Qrot, device=preload_device)
    Qxfm = torch.as_tensor(Qxfm, device=preload_device)
    Qtxy = torch.as_tensor(Qtxy, device=preload_device)
    Qvel = torch.as_tensor(Qvel, device=preload_device)
    Qang = torch.as_tensor(Qang, device=preload_device)
    
    Yrvel = torch.as_tensor(Yrvel, device=preload_device)
    Yrang = torch.as_tensor(Yrang, device=preload_device)
    Yextra = torch.as_tensor(Yextra, device=preload_device)
    
    X = torch.as_tensor(X, device=preload_device)
    support_bones_t = torch.as_tensor(support_bones, dtype=torch.long, device=device)
    support_contact_index_t = torch.as_tensor(support_contact_index, dtype=torch.long, device=device)
    terrain_grid_t = torch.as_tensor(terrain_grid.data, dtype=torch.float32, device=device)
    
    # Make networks
    
    network_compressor = Compressor(len(compressor_mean_in), nlatent).to(device)
    network_decompressor = Decompressor(nfeatures + nlatent, len(decompressor_mean_out)).to(device)
    if ctx.enabled:
        ddp_kwargs = {
            "device_ids": [ctx.local_rank] if device.type == "cuda" else None,
            "output_device": ctx.local_rank if device.type == "cuda" else None,
        }
        network_compressor = nn.parallel.DistributedDataParallel(network_compressor, **ddp_kwargs)
        network_decompressor = nn.parallel.DistributedDataParallel(network_decompressor, **ddp_kwargs)
    
    # Function to save compressed database
    
    def save_compressed_database():
    
        with torch.no_grad():
            
            chunk_size = 4096
            latent_chunks = []
            compressor_model = unwrap_model(network_compressor)
            for start in range(0, nframes, chunk_size):
                stop = min(start + chunk_size, nframes)
                chunk = torch.cat([
                    Ypos[start:stop,1:].reshape([1, stop-start, -1]).to(device),
                    Ytxy[start:stop,1:].reshape([1, stop-start, -1]).to(device),
                    Yvel[start:stop,1:].reshape([1, stop-start, -1]).to(device),
                    Yang[start:stop,1:].reshape([1, stop-start, -1]).to(device),
                    Qpos[start:stop,1:].reshape([1, stop-start, -1]).to(device),
                    Qtxy[start:stop,1:].reshape([1, stop-start, -1]).to(device),
                    Qvel[start:stop,1:].reshape([1, stop-start, -1]).to(device),
                    Qang[start:stop,1:].reshape([1, stop-start, -1]).to(device),
                    Yrvel[start:stop].reshape([1, stop-start, -1]).to(device),
                    Yrang[start:stop].reshape([1, stop-start, -1]).to(device),
                    Yextra[start:stop].reshape([1, stop-start, -1]).to(device),
                ], dim=-1)
                latent_chunks.append(
                    compressor_model((chunk - compressor_mean_in) / compressor_std_in)[0].cpu()
                )
            Z = torch.cat(latent_chunks, dim=0)
            
            # Write latent variables
            
            with open('latent.bin', 'wb') as f:
                f.write(struct.pack('II', nframes, nlatent) + Z.cpu().numpy().astype(np.float32).ravel().tobytes())

    # Function to generate test animation for comparison
    
    def generate_animation():
        
        with torch.no_grad():
            
            # Get slice of database for first clip
            
            preview_index = min(2, len(spans) - 1)
            start, stop = spans[preview_index]
            stop = min(start + 1000, stop)
            
            Ygnd_pos = Ypos[start:stop][np.newaxis].to(device)
            Ygnd_rot = Yrot[start:stop][np.newaxis].to(device)
            Ygnd_txy = Ytxy[start:stop][np.newaxis].to(device)
            Ygnd_vel = Yvel[start:stop][np.newaxis].to(device)
            Ygnd_ang = Yang[start:stop][np.newaxis].to(device)
            
            Qgnd_pos = Qpos[start:stop][np.newaxis].to(device)
            Qgnd_txy = Qtxy[start:stop][np.newaxis].to(device)
            Qgnd_vel = Qvel[start:stop][np.newaxis].to(device)
            Qgnd_ang = Qang[start:stop][np.newaxis].to(device)
            
            Ygnd_rvel = Yrvel[start:stop][np.newaxis].to(device)
            Ygnd_rang = Yrang[start:stop][np.newaxis].to(device)
            Ygnd_extra = Yextra[start:stop][np.newaxis].to(device)
            
            Xgnd = X[start:stop][np.newaxis].to(device)
            
            # Pass through compressor
            
            compressor_model = unwrap_model(network_compressor)
            decompressor_model = unwrap_model(network_decompressor)

            Zgnd = compressor_model((torch.cat([
                Ygnd_pos[:,:,1:].reshape([1, stop-start, -1]),
                Ygnd_txy[:,:,1:].reshape([1, stop-start, -1]),
                Ygnd_vel[:,:,1:].reshape([1, stop-start, -1]),
                Ygnd_ang[:,:,1:].reshape([1, stop-start, -1]),
                Qgnd_pos[:,:,1:].reshape([1, stop-start, -1]),
                Qgnd_txy[:,:,1:].reshape([1, stop-start, -1]),
                Qgnd_vel[:,:,1:].reshape([1, stop-start, -1]),
                Qgnd_ang[:,:,1:].reshape([1, stop-start, -1]),
                Ygnd_rvel.reshape([1, stop-start, -1]),
                Ygnd_rang.reshape([1, stop-start, -1]),
                Ygnd_extra.reshape([1, stop-start, -1]),
            ], dim=-1) - compressor_mean_in) / compressor_std_in)
            
            # Pass through decompressor
            
            Ytil = (decompressor_model(torch.cat([Xgnd, Zgnd], dim=-1)) * 
                decompressor_std_out + decompressor_mean_out)
            
            # Extract required components
            
            Ytil_pos = Ytil[:,:, 0*(nbones-1): 3*(nbones-1)].reshape([1, stop-start, nbones-1, 3])
            Ytil_txy = Ytil[:,:, 3*(nbones-1): 9*(nbones-1)].reshape([1, stop-start, nbones-1, 3, 2])
            Ytil_rvel = Ytil[:,:,15*(nbones-1)+0:15*(nbones-1)+3].reshape([1, stop-start, 3])
            Ytil_rang = Ytil[:,:,15*(nbones-1)+3:15*(nbones-1)+6].reshape([1, stop-start, 3])
            
            # Convert to quat and remove batch
            
            Ytil_rot = quat.from_xform_xy(Ytil_txy[0].cpu().numpy())
            Ytil_pos = Ytil_pos[0].cpu().numpy()
            Ytil_rvel = Ytil_rvel[0].cpu().numpy()
            Ytil_rang = Ytil_rang[0].cpu().numpy()
            
            # Integrate root displacement
            
            Ytil_rootrot = [Ygnd_rot[0,0,0].cpu().numpy()]
            Ytil_rootpos = [Ygnd_pos[0,0,0].cpu().numpy()]
            for i in range(1, Ygnd_pos.shape[1]):
                Ytil_rootpos.append(Ytil_rootpos[-1] + quat.mul_vec(Ytil_rootrot[-1], Ytil_rvel[i-1]) * dt)
                Ytil_rootrot.append(quat.mul(Ytil_rootrot[-1], quat.from_scaled_angle_axis(quat.mul_vec(Ytil_rootrot[-1], Ytil_rang[i-1]) * dt)))
            
            Ytil_rootrot = np.concatenate([r[np.newaxis] for r in Ytil_rootrot])
            Ytil_rootpos = np.concatenate([p[np.newaxis] for p in Ytil_rootpos])
            
            Ytil_rot = np.concatenate([Ytil_rootrot[:,np.newaxis], Ytil_rot], axis=1)
            Ytil_pos = np.concatenate([Ytil_rootpos[:,np.newaxis], Ytil_pos], axis=1)
            
            # Write BVH
            
            try:
                bvh.save('decompressor_Ygnd.bvh', {
                    'rotations': np.degrees(quat.to_euler(Ygnd_rot[0].cpu().numpy(), order='zyx')),
                    'positions': 100.0 * Ygnd_pos[0].cpu().numpy(),
                    'offsets': 100.0 * Ygnd_pos[0,0].cpu().numpy(),
                    'parents': parents,
                    'names': ['joint_%i' % i for i in range(nbones)],
                    'order': 'zyx'
                })
                
                bvh.save('decompressor_Ytil.bvh', {
                    'rotations': np.degrees(quat.to_euler(Ytil_rot, order='zyx')),
                    'positions': 100.0 * Ytil_pos,
                    'offsets': 100.0 * Ytil_pos[0],
                    'parents': parents,
                    'names': ['joint_%i' % i for i in range(nbones)],
                    'order': 'zyx'
                })
            except IOError as e:
                print(e)
                
            # Write features
            
            fmin, fmax = Xgnd.cpu().numpy().min(), Xgnd.cpu().numpy().max()
            
            fig, axs = plt.subplots(nfeatures, sharex=True, figsize=(12, 2*nfeatures))
            for i in range(nfeatures):
                axs[i].plot(Xgnd[0,:500,i].cpu().numpy())
                axs[i].set_ylim(fmin, fmax)
            plt.tight_layout()
            
            try:
                plt.savefig('decompressor_X.png')
            except IOError as e:
                print(e)

            plt.close()
            
            # Write latent
            
            lmin, lmax = Zgnd.cpu().numpy().min(), Zgnd.cpu().numpy().max()
            
            fig, axs = plt.subplots(nlatent, sharex=True, figsize=(12, 2*nlatent))
            for i in range(nlatent):
                axs[i].plot(Zgnd[0,:500,i].cpu().numpy())
                axs[i].set_ylim(lmin, lmax)
            plt.tight_layout()            
            
            try:
                plt.savefig('decompressor_Z.png')
            except IOError as e:
                print(e)

            plt.close()
    
    # Train
    
    writer = SummaryWriter() if is_main_process(ctx) else SummaryWriter()

    optimizer = torch.optim.AdamW(
        list(network_compressor.parameters()) + 
        list(network_decompressor.parameters()), 
        lr=lr,
        amsgrad=True,
        weight_decay=0.001)
        
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    rolling_loss = None
    
    sys.stdout.write('\n')
    
    for i in range(niter):
    
        optimizer.zero_grad()
        
        # Extract batch
        
        batch = torch.as_tensor(sample_window_batch(valid_starts, window, batchsize), dtype=torch.long, device=preload_device)
        
        Xgnd = X[batch]
        
        Ygnd_pos = Ypos[batch]
        Ygnd_txy = Ytxy[batch]
        Ygnd_vel = Yvel[batch]
        Ygnd_ang = Yang[batch]
        
        Qgnd_pos = Qpos[batch]
        Qgnd_xfm = Qxfm[batch]
        Qgnd_txy = Qtxy[batch]
        Qgnd_vel = Qvel[batch]
        Qgnd_ang = Qang[batch]
        
        Ygnd_rvel = Yrvel[batch]
        Ygnd_rang = Yrang[batch]
        
        Ygnd_extra = Yextra[batch]
        
        # Encode
        
        Zgnd = network_compressor((torch.cat([
            Ygnd_pos[:,:,1:].reshape([batchsize, window, -1]),
            Ygnd_txy[:,:,1:].reshape([batchsize, window, -1]),
            Ygnd_vel[:,:,1:].reshape([batchsize, window, -1]),
            Ygnd_ang[:,:,1:].reshape([batchsize, window, -1]),
            Qgnd_pos[:,:,1:].reshape([batchsize, window, -1]),
            Qgnd_txy[:,:,1:].reshape([batchsize, window, -1]),
            Qgnd_vel[:,:,1:].reshape([batchsize, window, -1]),
            Qgnd_ang[:,:,1:].reshape([batchsize, window, -1]),
            Ygnd_rvel.reshape([batchsize, window, -1]),
            Ygnd_rang.reshape([batchsize, window, -1]),
            Ygnd_extra.reshape([batchsize, window, -1]),
        ], dim=-1) - compressor_mean_in) / compressor_std_in)
            
        # Decode
        
        Ytil = (network_decompressor(torch.cat([Xgnd, Zgnd], dim=-1)) * 
            decompressor_std_out + decompressor_mean_out)
        
        Ytil_pos = Ytil[:,:, 0*(nbones-1): 3*(nbones-1)].reshape([batchsize, window, nbones-1, 3])
        Ytil_txy = Ytil[:,:, 3*(nbones-1): 9*(nbones-1)].reshape([batchsize, window, nbones-1, 3, 2])
        Ytil_vel = Ytil[:,:, 9*(nbones-1):12*(nbones-1)].reshape([batchsize, window, nbones-1, 3])
        Ytil_ang = Ytil[:,:,12*(nbones-1):15*(nbones-1)].reshape([batchsize, window, nbones-1, 3])
        Ytil_rvel = Ytil[:,:,15*(nbones-1)+0:15*(nbones-1)+3].reshape([batchsize, window, 3])
        Ytil_rang = Ytil[:,:,15*(nbones-1)+3:15*(nbones-1)+6].reshape([batchsize, window, 3])
        Ytil_extra = Ytil[:,:,15*(nbones-1)+6:15*(nbones-1)+8].reshape([batchsize, window, nextra])
        
        # Add root bone from ground
        
        Ytil_pos = torch.cat([Ygnd_pos[:,:,0:1], Ytil_pos], dim=2)
        Ytil_txy = torch.cat([Ygnd_txy[:,:,0:1], Ytil_txy], dim=2)
        Ytil_vel = torch.cat([Ygnd_vel[:,:,0:1], Ytil_vel], dim=2)
        Ytil_ang = torch.cat([Ygnd_ang[:,:,0:1], Ytil_ang], dim=2)
        
        # Do FK
        
        Ytil_xfm = txform.from_xy(Ytil_txy)

        Gtil_xfm, Gtil_pos, Gtil_vel, Gtil_ang = txform.fk_vel(
            Ytil_xfm, Ytil_pos, Ytil_vel, Ytil_ang, parents)
        
        # Compute Character Space
        
        Qtil_xfm = txform.inv_mul(Gtil_xfm[:,:,0:1], Gtil_xfm)
        Qtil_pos = txform.inv_mul_vec(Gtil_xfm[:,:,0:1], Gtil_pos - Gtil_pos[:,:,0:1])
        Qtil_vel = txform.inv_mul_vec(Gtil_xfm[:,:,0:1], Gtil_vel)
        Qtil_ang = txform.inv_mul_vec(Gtil_xfm[:,:,0:1], Gtil_ang)
        
        # Compute deltas
        
        Ygnd_dpos = (Ygnd_pos[:,1:] - Ygnd_pos[:,:-1]) / dt
        Ygnd_drot = (Ygnd_txy[:,1:] - Ygnd_txy[:,:-1]) / dt
        Qgnd_dpos = (Qgnd_pos[:,1:] - Qgnd_pos[:,:-1]) / dt
        Qgnd_drot = (Qgnd_xfm[:,1:] - Qgnd_xfm[:,:-1]) / dt
        
        Ytil_dpos = (Ytil_pos[:,1:] - Ytil_pos[:,:-1]) / dt
        Ytil_drot = (Ytil_txy[:,1:] - Ytil_txy[:,:-1]) / dt
        Qtil_dpos = (Qtil_pos[:,1:] - Qtil_pos[:,:-1]) / dt
        Qtil_drot = (Qtil_xfm[:,1:] - Qtil_xfm[:,:-1]) / dt
        
        Zdgnd = (Zgnd[:,1:] - Zgnd[:,:-1]) / dt

        # Terrain-contact losses in world space

        support_pos = Gtil_pos.index_select(2, support_bones_t)
        support_vel = Gtil_vel.index_select(2, support_bones_t)
        contact_mask = Ygnd_extra.index_select(2, support_contact_index_t).to(torch.float32)
        support_ground = sample_terrain_height_torch(
            terrain_grid_t,
            terrain_grid.x_min,
            terrain_grid.x_max,
            terrain_grid.z_min,
            terrain_grid.z_max,
            support_pos[..., 0],
            support_pos[..., 2],
        )
        support_clearance = support_pos[..., 1] - support_ground
        active_contacts = torch.clamp(contact_mask.sum(), min=1.0)
        support_contact_abs = torch.sum(torch.abs(support_clearance) * contact_mask) / active_contacts
        support_contact_vy = torch.sum(torch.abs(support_vel[..., 1]) * contact_mask) / active_contacts
        support_penetration = torch.sum(torch.relu(-support_clearance) * contact_mask) / active_contacts
        support_floating = torch.sum(torch.relu(support_clearance - 0.06) * contact_mask) / active_contacts
        
        # Compute losses
        
        loss_loc_pos = torch.mean(75.0 * torch.abs(Ygnd_pos - Ytil_pos))
        loss_loc_txy = torch.mean(10.0 * torch.abs(Ygnd_txy - Ytil_txy))
        loss_loc_vel = torch.mean(10.0 * torch.abs(Ygnd_vel - Ytil_vel))
        loss_loc_ang = torch.mean(1.25 * torch.abs(Ygnd_ang - Ytil_ang))
        loss_loc_rvel = torch.mean(2.0 * torch.abs(Ygnd_rvel - Ytil_rvel))
        loss_loc_rang = torch.mean(2.0 * torch.abs(Ygnd_rang - Ytil_rang))
        loss_loc_extra = torch.mean(2.0 * torch.abs(Ygnd_extra - Ytil_extra))
        
        loss_chr_pos = torch.mean(15.0 * torch.abs(Qgnd_pos - Qtil_pos))
        loss_chr_xfm = torch.mean( 5.0 * torch.abs(Qgnd_xfm - Qtil_xfm))
        loss_chr_vel = torch.mean( 2.0 * torch.abs(Qgnd_vel - Qtil_vel))
        loss_chr_ang = torch.mean(0.75 * torch.abs(Qgnd_ang - Qtil_ang))
        
        loss_lvel_pos = torch.mean(10.0 * torch.abs(Ygnd_dpos - Ytil_dpos))
        loss_lvel_rot = torch.mean(1.75 * torch.abs(Ygnd_drot - Ytil_drot))
        loss_cvel_pos = torch.mean(2.0  * torch.abs(Qgnd_dpos - Qtil_dpos))
        loss_cvel_rot = torch.mean(0.75 * torch.abs(Qgnd_drot - Qtil_drot))        
        
        loss_sreg = torch.mean(0.1  * torch.abs(Zgnd))
        loss_lreg = torch.mean(0.1  * torch.square(Zgnd))
        loss_vreg = torch.mean(0.01 * torch.abs(Zdgnd))
        loss_terrain_contact = 15.0 * support_contact_abs
        loss_terrain_vy = 5.0 * support_contact_vy
        loss_terrain_penetration = 25.0 * support_penetration
        loss_terrain_floating = 10.0 * support_floating
        
        loss = (
            loss_loc_pos + 
            loss_loc_txy + 
            loss_loc_vel + 
            loss_loc_ang + 
            loss_loc_rvel + 
            loss_loc_rang + 
            loss_loc_extra + 
            loss_chr_pos + 
            loss_chr_xfm + 
            loss_chr_vel + 
            loss_chr_ang + 
            loss_lvel_pos + 
            loss_lvel_rot + 
            loss_cvel_pos + 
            loss_cvel_rot + 
            loss_sreg + 
            loss_lreg +
            loss_vreg +
            loss_terrain_contact +
            loss_terrain_vy +
            loss_terrain_penetration +
            loss_terrain_floating)
                
        # Backprop
        
        loss.backward()

        optimizer.step()
    
        # Logging
        
        mean_loss = distributed_mean(loss.item(), ctx)

        if is_main_process(ctx):
            writer.add_scalar('decompressor/loss', mean_loss, i)
            writer.add_scalars('decompressor/loss_terms', {
                'loc_pos': loss_loc_pos.item(),
                'loc_txy': loss_loc_txy.item(),
                'loc_vel': loss_loc_vel.item(),
                'loc_ang': loss_loc_ang.item(),
                'loc_rvel': loss_loc_rvel.item(),
                'loc_rang': loss_loc_rang.item(),
                'loc_extra': loss_loc_extra.item(),
                'chr_pos': loss_chr_pos.item(),
                'chr_xfm': loss_chr_xfm.item(),
                'chr_vel': loss_chr_vel.item(),
                'chr_ang': loss_chr_ang.item(),
                'lvel_pos': loss_lvel_pos.item(),
                'lvel_rot': loss_lvel_rot.item(),
                'cvel_pos': loss_cvel_pos.item(),
                'cvel_rot': loss_cvel_rot.item(),
                'sreg': loss_sreg.item(),
                'lreg': loss_lreg.item(),
                'vreg': loss_vreg.item(),
                'terrain_contact': loss_terrain_contact.item(),
                'terrain_vy': loss_terrain_vy.item(),
                'terrain_penetration': loss_terrain_penetration.item(),
                'terrain_floating': loss_terrain_floating.item(),
            }, i)
            writer.add_scalars('decompressor/latent', {
                'mean': Zgnd.mean().item(),
                'std': Zgnd.std().item(),
            }, i)
        
            if rolling_loss is None:
                rolling_loss = mean_loss
            else:
                rolling_loss = rolling_loss * 0.99 + mean_loss * 0.01
        
        if i % 10 == 0 and is_main_process(ctx):
            sys.stdout.write('\rIter: %7i Loss: %5.3f' % (i, rolling_loss))
        
        if i % args.save_every == 0 and is_main_process(ctx):
            decompressor_model = unwrap_model(network_decompressor)
            generate_animation()
            save_compressed_database()
            save_network('decompressor.bin', [
                decompressor_model.linear0, 
                decompressor_model.linear1],
                decompressor_mean_in,
                decompressor_std_in,
                decompressor_mean_out,
                decompressor_std_out)
        if i % args.save_every == 0:
            scheduler.step()
        if i % args.save_every == 0 and ctx.enabled:
            torch.distributed.barrier()

    cleanup_distributed(ctx)
            
