from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .fusion import metrics
from .splits import pair_keys, sample_negatives
from .utils import atomic_json, atomic_torch, seed_all


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, variational=False):
        super().__init__()
        self.variational = variational
        self.hidden = GCNConv(input_dim, hidden_dim, cached=True)
        self.mean = GCNConv(hidden_dim, latent_dim, cached=True)
        self.logstd = GCNConv(hidden_dim, latent_dim, cached=True) if variational else None

    def forward(self, x, training_edge_index):
        h = F.relu(self.hidden(x, training_edge_index))
        mu = self.mean(h, training_edge_index)
        if not self.variational:
            return mu, mu, None
        logstd = self.logstd(h, training_edge_index).clamp(max=10)
        z = mu + torch.randn_like(mu) * logstd.exp() if self.training else mu
        return z, mu, logstd


def mean_kl(mu, logstd):
    return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu.square() - (2 * logstd).exp(), dim=1))


def score_embeddings(embedding, pairs):
    return np.sum(embedding[pairs[:, 0]] * embedding[pairs[:, 1]], axis=1).astype(np.float64)


def edge_logits(z, pairs):
    return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=1)


def train_encoder(x_cpu, train_edges, original_edges, reserved_negatives, selection_pairs,
                  selection_labels, config, directory: Path, method, seed, negative_seed, device, resume):
    start = time.perf_counter()
    seed_all(seed)
    n = x_cpu.shape[0]
    model = Encoder(x_cpu.shape[1], config['hidden_dim'], config['latent_dim'], method == 'VGAE').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    x = x_cpu.to(device)
    directed = np.concatenate([train_edges, train_edges[:, ::-1]]).T.copy()
    edge_index = torch.as_tensor(directed, device=device, dtype=torch.long)
    pos = torch.as_tensor(train_edges, device=device, dtype=torch.long)
    selected = torch.as_tensor(selection_pairs, device=device, dtype=torch.long)
    excluded = np.unique(np.concatenate([pair_keys(original_edges, n), pair_keys(reserved_negatives, n)]))
    rng = np.random.default_rng(negative_seed)
    checkpoint = directory / 'checkpoints' / (method.lower() + '_latest.pt')
    best_ap, best_epoch, epoch_start, elapsed_prior = -1.0, 0, 0, 0.0
    best_state = None
    history = []
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state'])
        optimizer.load_state_dict(saved['optimizer'])
        best_ap, best_epoch, best_state = saved['best_ap'], saved['best_epoch'], saved['best_state']
        epoch_start, history, elapsed_prior = saved['epoch'], saved['history'], saved['elapsed_seconds']
        rng.bit_generator.state = saved['negative_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    for epoch in range(epoch_start + 1, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z, mu, logstd = model(x, edge_index)
        negative = sample_negatives(n, len(train_edges), excluded, rng)
        neg = torch.as_tensor(negative, device=device, dtype=torch.long)
        positive_loss = F.softplus(-edge_logits(z, pos)).mean()
        negative_loss = F.softplus(edge_logits(z, neg)).mean()
        kl = mean_kl(mu, logstd) if logstd is not None else torch.zeros((), device=device)
        loss = positive_loss + negative_loss + kl / n
        if not torch.isfinite(loss):
            raise RuntimeError(f'{method}: nonfinite loss at epoch {epoch}')
        loss.backward()
        optimizer.step()
        if epoch % config['selection_every'] == 0 or epoch == config['epochs']:
            model.eval()
            with torch.no_grad():
                mu = model(x, edge_index)[1]
                score = edge_logits(mu, selected).detach().cpu().numpy()
            evaluation = metrics(selection_labels, score)
            if evaluation['ap'] > best_ap:
                best_ap, best_epoch = evaluation['ap'], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = {'epoch': epoch, 'loss': float(loss.detach()), 'positive_bce': float(positive_loss.detach()), 'negative_bce': float(negative_loss.detach()), 'mean_kl': float(kl.detach()), 'selection_ap': evaluation['ap'], 'selection_auc': evaluation['auc']}
            history.append(row)
            elapsed = elapsed_prior + time.perf_counter() - start
            atomic_torch(checkpoint, {'state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_state': best_state, 'best_ap': best_ap, 'best_epoch': best_epoch, 'epoch': epoch, 'history': history, 'negative_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state() if device.type == 'cuda' else None, 'elapsed_seconds': elapsed})
            print(f'  {method} epoch {epoch:3d} selection AP={evaluation["ap"]:.4f} loss={row["loss"]:.4f}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embedding = model(x, edge_index)[1].detach().cpu().numpy()
    info = {'method': method, 'best_epoch': best_epoch, 'selection_ap': best_ap, 'epochs_completed': config['epochs'], 'seconds': elapsed_prior + time.perf_counter() - start, 'history': history, 'device': str(device), 'kl_formula': 'mean node KL divided by N' if method == 'VGAE' else None}
    atomic_torch(directory / 'checkpoints' / (method.lower() + '_best.pt'), best_state)
    atomic_json(directory / (method.lower() + '_training.json'), info)
    return embedding, info
