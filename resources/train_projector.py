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

from sklearn.neighbors import BallTree

from train_common import load_database, load_features, load_latent, save_network

# Networks

class Projector(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=512):
        super(Projector, self).__init__()
        
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)

# Training procedure

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--niter', type=int, default=100000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--nn-chunk', type=int, default=65536)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Load data
    
    database = load_database('./database.bin')
    range_starts = database['range_starts']
    range_stops = database['range_stops']
    del database
    
    X = load_features('./features.bin')['features'].copy().astype(np.float32)
    Z = load_latent('./latent.bin')['latent'].copy().astype(np.float32)
    
    nframes = X.shape[0]
    nfeatures = X.shape[1]
    nlatent = Z.shape[1]
    
    # Parameters
    
    seed = 1234
    batchsize = args.batchsize
    lr = args.lr
    niter = args.niter

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f'Using device: {device}')
    
    # Fit acceleration structure for nearest neighbors search

    if device.type == 'cuda':
        X_db_device = torch.as_tensor(X, device=device)
        X_db_sqnorm = torch.sum(torch.square(X_db_device), dim=1)
        tree = None
    else:
        tree = BallTree(X)
        X_db_device = None
        X_db_sqnorm = None
    
    # Compute means/stds
    
    X_scale = X.std()
    X_noise_std = X.std(axis=0) + 1.0
    
    projector_mean_out = torch.as_tensor(np.hstack([
        X.mean(axis=0).ravel(),
        Z.mean(axis=0).ravel(),
    ]).astype(np.float32), device=device)
    
    projector_std_out = torch.as_tensor(np.hstack([
        X.std(axis=0).ravel(),
        Z.std(axis=0).ravel(),
    ]).astype(np.float32), device=device)
    
    projector_mean_in = torch.as_tensor(np.hstack([
        X.mean(axis=0).ravel(),
    ]).astype(np.float32), device=device)
    
    projector_std_in = torch.as_tensor(np.hstack([
        np.repeat(X_scale, nfeatures),
    ]).astype(np.float32), device=device)
    
    # Make networks
    
    network_projector = Projector(nfeatures, nfeatures + nlatent).to(device)

    def nearest_neighbor_search(xhat_np):
        if device.type != 'cuda':
            return tree.query(xhat_np, k=1, return_distance=False)[:,0]

        xhat = torch.as_tensor(xhat_np, device=device)
        xhat_sqnorm = torch.sum(torch.square(xhat), dim=1)
        best_dist = torch.full((xhat.shape[0],), float('inf'), device=device)
        best_idx = torch.zeros((xhat.shape[0],), dtype=torch.long, device=device)

        for start_idx in range(0, nframes, args.nn_chunk):
            stop_idx = min(start_idx + args.nn_chunk, nframes)
            chunk = X_db_device[start_idx:stop_idx]
            dist = (
                xhat_sqnorm[:, None] +
                X_db_sqnorm[start_idx:stop_idx][None, :] -
                2.0 * torch.matmul(xhat, chunk.T)
            )
            chunk_best_dist, chunk_best_idx = torch.min(dist, dim=1)
            improved = chunk_best_dist < best_dist
            best_dist[improved] = chunk_best_dist[improved]
            best_idx[improved] = start_idx + chunk_best_idx[improved]

        return best_idx.cpu().numpy()
    
    # Function to generate test predictions

    def generate_predictions():
        
        with torch.no_grad():
            
            # Get slice of database for first clip
            
            start = range_starts[2]
            stop = min(start + 1000, range_stops[2])
            
            nsigma = np.random.uniform(size=[stop-start, 1]).astype(np.float32)
            noise = np.random.normal(size=[stop-start, nfeatures]).astype(np.float32)
            Xhat = X[start:stop] + X_noise_std * nsigma * noise
            
            # Find nearest
            
            nearest = nearest_neighbor_search(Xhat)
            
            Xgnd = torch.as_tensor(X[nearest], device=device)
            Zgnd = torch.as_tensor(Z[nearest], device=device)
            Xhat = torch.as_tensor(Xhat, device=device)
            
            # Project
            
            output = (network_projector((Xhat - projector_mean_in) / projector_std_in) *
                projector_std_out + projector_mean_out)
            
            Xtil = output[:,:nfeatures]
            Ztil = output[:,nfeatures:]
            
            # Write features
            
            fmin, fmax = Xhat.cpu().numpy().min(), Xhat.cpu().numpy().max()
            
            fig, axs = plt.subplots(nfeatures, sharex=True, figsize=(12, 2*nfeatures))
            for i in range(nfeatures):
                axs[i].plot(Xgnd[:500:4,i].cpu().numpy(), marker='.', linestyle='None')
                axs[i].plot(Xtil[:500:4,i].cpu().numpy(), marker='.', linestyle='None')
                axs[i].plot(Xhat[:500:4,i].cpu().numpy(), marker='.', linestyle='None')
                axs[i].set_ylim(fmin, fmax)
            plt.tight_layout()
            
            try:
                plt.savefig('projector_X.png')
            except IOError as e:
                print(e)

            plt.close()
            
            # Write latent
            
            lmin, lmax = Zgnd.cpu().numpy().min(), Zgnd.cpu().numpy().max()
            
            fig, axs = plt.subplots(nlatent, sharex=True, figsize=(12, 2*nlatent))
            for i in range(nlatent):
                axs[i].plot(Zgnd[:500:4,i].cpu().numpy(), marker='.', linestyle='None')
                axs[i].plot(Ztil[:500:4,i].cpu().numpy(), marker='.', linestyle='None')
                axs[i].set_ylim(lmin, lmax)
            plt.tight_layout()
            
            try:
                plt.savefig('projector_Z.png')
            except IOError as e:
                print(e)

            plt.close()

    # Train
    
    writer = SummaryWriter()

    optimizer = torch.optim.AdamW(
        network_projector.parameters(), 
        lr=lr,
        amsgrad=True,
        weight_decay=0.001)
        
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    rolling_loss = None
    
    sys.stdout.write('\n')
    
    for i in range(niter):
    
        optimizer.zero_grad()
        
        # Extract batch
        
        samples = np.random.randint(0, nframes, size=[batchsize])
        
        nsigma = np.random.uniform(size=[batchsize, 1]).astype(np.float32)
        noise = np.random.normal(size=[batchsize, nfeatures]).astype(np.float32)
        Xhat = X[samples] + X_noise_std * nsigma * noise
        
        # Find nearest
        
        nearest = nearest_neighbor_search(Xhat)
        
        Xgnd = torch.as_tensor(X[nearest], device=device)
        Zgnd = torch.as_tensor(Z[nearest], device=device)
        Xhat = torch.as_tensor(Xhat, device=device)
        Dgnd = torch.sqrt(torch.sum(torch.square(Xhat - Xgnd), dim=-1))
        
        # Projector
        
        output = (network_projector((Xhat - projector_mean_in) / projector_std_in) *
            projector_std_out + projector_mean_out)
        
        Xtil = output[:,:nfeatures]
        Ztil = output[:,nfeatures:]
        Dtil = torch.sqrt(torch.sum(torch.square(Xhat - Xtil), dim=-1))
        
        # Compute Losses
        
        loss_xval = torch.mean(1.0 * torch.abs(Xgnd - Xtil))
        loss_zval = torch.mean(5.0 * torch.abs(Zgnd - Ztil))
        loss_dist = torch.mean(0.3 * torch.abs(Dgnd - Dtil))
        loss = loss_xval + loss_zval + loss_dist
        
        # Backprop
        
        loss.backward()

        optimizer.step()
    
        # Logging
        
        writer.add_scalar('projector/loss', loss.item(), i)
        
        writer.add_scalars('projector/loss_terms', {
            'xval': loss_xval.item(),
            'zval': loss_zval.item(),
            'dist': loss_dist.item(),
        }, i)
        
        if rolling_loss is None:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01
        
        if i % 10 == 0:
            sys.stdout.write('\rIter: %7i Loss: %5.3f' % (i, rolling_loss))
        
        if i % args.save_every == 0:
            generate_predictions()
            save_network('projector.bin', [
                network_projector.linear0, 
                network_projector.linear1, 
                network_projector.linear2, 
                network_projector.linear3,
                network_projector.linear4],
                projector_mean_in,
                projector_std_in,
                projector_mean_out,
                projector_std_out)
            
        if i % args.save_every == 0:
            scheduler.step()
            
