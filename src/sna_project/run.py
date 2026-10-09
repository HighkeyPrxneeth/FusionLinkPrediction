from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from .data import load_dataset
from .deepwalk import train_deepwalk
from .fusion import fit_predict, metrics
from .heuristics import NAMES, adjacency_sets, compute, transform_features
from .models import score_embeddings, train_encoder
from .splits import labeled_pairs, make_split, save_split, validate_split
from .utils import atomic_json, atomic_npz, digest_json, environment, source_hash


def run_one(root, name, run_id, config, x, edges, stats, device, resume, artifact_folder='artifacts'):
    directory = root / artifact_folder / name / f'run_{run_id}'
    directory.mkdir(parents=True, exist_ok=True)
    identity = {'dataset': name, 'run_id': run_id, 'config': config, 'source_hash': source_hash(),
                'edge_hash': stats['edge_hash'], 'feature_hash': stats['feature_hash'], 'device': str(device)}
    fingerprint = digest_json(identity)
    meta = directory / 'identity.json'
    if meta.exists():
        previous = json.loads(meta.read_text())
        if previous['fingerprint'] != fingerprint:
            raise RuntimeError(f'Protocol/source changed for {directory}; use a new artifact directory to preserve old results.')
    atomic_json(meta, dict(identity, fingerprint=fingerprint))
    complete = directory / 'metrics.json'
    if resume and complete.exists():
        print(f'{name} run {run_id}: already complete', flush=True)
        return
    started = time.perf_counter()
    n = x.shape[0]
    # Each randomness source has a stable and independent seed.
    seed_root = config['base_seed'] + run_id * 1000
    seeds = {'split': seed_root + 1, 'gae_init': seed_root + 2, 'vgae_init': seed_root + 3,
             'gae_negatives': seed_root + 4, 'vgae_negatives': seed_root + 5, 'deepwalk': seed_root + 6}
    if resume and (directory / 'split.npz').exists():
        with np.load(directory / 'split.npz') as saved:
            split = {k: saved[k] for k in saved.files}
        validate_split(edges, split, n)
    else:
        split = make_split(edges, n, config, seeds['split'])
    manifest = save_split(directory, edges, split, n, seeds['split'])
    atomic_json(directory / 'seeds.json', seeds)
    print(f'{name} run {run_id}: train={len(split["train_pos"])} test={len(split["test_pos"])} device={device}', flush=True)
    candidate = {k: labeled_pairs(split, k) for k in ['selection', 'fusion', 'test']}
    reserved = np.concatenate([split[k + '_neg'] for k in ['selection', 'fusion', 'test']])
    timing = {}
    embeddings = {}
    for method in ['GAE', 'VGAE', 'DeepWalk']:
        file = directory / (method.lower() + '_embedding.npz')
        if resume and file.exists():
            with np.load(file) as saved:
                embeddings[method] = saved['embedding']
            info = json.loads((directory / (method.lower() + '_training.json')).read_text())
        elif method == 'DeepWalk':
            embeddings[method], info = train_deepwalk(n, split['train_pos'], config, directory, seeds['deepwalk'])
        else:
            embeddings[method], info = train_encoder(x, split['train_pos'], edges, reserved, candidate['selection'][0],
                                                    candidate['selection'][1], config, directory, method,
                                                    seeds[method.lower() + '_init'], seeds[method.lower() + '_negatives'], device, resume)
        atomic_npz(file, embedding=embeddings[method])
        timing[method] = info['seconds']
    before = time.perf_counter()
    neighbors = adjacency_sets(n, split['train_pos'])
    h = {k: compute(neighbors, candidate[k][0]) for k in ['fusion', 'test']}
    timing['heuristics'] = time.perf_counter() - before
    score = {method: score_embeddings(embedding, candidate['test'][0]) for method, embedding in embeddings.items()}
    for index, method in enumerate(NAMES):
        score[method] = h['test'][:, index]
    vgae_fusion = score_embeddings(embeddings['VGAE'], candidate['fusion'][0])
    features = {k: transform_features(h[k], vgae_fusion if k == 'fusion' else score['VGAE']) for k in ['fusion', 'test']}
    definitions = {'Heuristics LR': list(range(1, 6)), 'VGAE score LR': [0], 'Fusion': list(range(6))}
    for i, method in enumerate(NAMES, start=1):
        definitions['Fusion minus ' + method] = [j for j in range(6) if j != i]
    coefficients = {}
    before = time.perf_counter()
    for method, columns in definitions.items():
        score[method], classifier = fit_predict(features['fusion'][:, columns], candidate['fusion'][1], features['test'][:, columns], config['logistic_c'])
        coefficients[method] = {'feature_names': [['VGAE'] + NAMES][0] if method == 'Fusion' else [(['VGAE'] + NAMES)[j] for j in columns],
                                'coef': classifier[-1].coef_[0].tolist(), 'intercept': float(classifier[-1].intercept_[0]),
                                'scaler_mean': classifier[0].mean_.tolist(), 'scaler_scale': classifier[0].scale_.tolist(), 'fit_partition': 'fusion'}
        joblib.dump(classifier, directory / 'checkpoints' / (method.replace(' ', '_') + '.joblib'))
    timing['fusion_fits'] = time.perf_counter() - before
    test_pairs, y = candidate['test']
    groups = {'all': np.ones(len(y), dtype=bool), 'CN=0': h['test'][:, 0] == 0, 'CN>0': h['test'][:, 0] > 0}
    results = {method: {group: metrics(y[mask], values[mask]) for group, mask in groups.items()} for method, values in score.items()}
    atomic_npz(directory / 'predictions.npz', pairs=test_pairs, labels=y, heuristics=h['test'], **{k: v for k, v in score.items()})
    atomic_npz(directory / 'pair_features.npz', fusion_pairs=candidate['fusion'][0], fusion_labels=candidate['fusion'][1], fusion_features=features['fusion'], test_pairs=test_pairs, test_labels=y, test_features=features['test'])
    atomic_json(directory / 'coefficients.json', coefficients)
    atomic_json(directory / 'timings.json', dict(timing, current_invocation_seconds=time.perf_counter() - started))
    atomic_json(complete, {'dataset': name, 'run_id': run_id, 'fingerprint': fingerprint, 'training_graph_hash': manifest['training_graph_hash'],
                           'method_graph_hashes': {method: manifest['training_graph_hash'] for method in score}, 'results': results})
    print(f'{name} run {run_id}: Fusion AUC={results["Fusion"]["all"]["auc"]:.4f} AP={results["Fusion"]["all"]["ap"]:.4f}; VGAE AP={results["VGAE"]["all"]["ap"]:.4f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Run reproducible link prediction experiments')
    parser.add_argument('--config', default='configs/main.yaml')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--runs', nargs='+', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--artifact-folder', default='artifacts')
    parser.add_argument('--epochs', type=int, help='Pilot override; use a separate artifact folder')
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = yaml.safe_load((root / args.config).read_text())
    if args.epochs:
        config['epochs'] = args.epochs
    requested = args.device or config['device']
    selected = ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested
    if selected == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(config['torch_threads'])
    device = torch.device(selected)
    atomic_json(root / args.artifact_folder / 'environment.json', environment())
    atomic_json(root / args.artifact_folder / 'config.json', config)
    for name in args.datasets or config['datasets']:
        x, edges, stats = load_dataset(root, name)
        for run_id in args.runs if args.runs is not None else config['run_ids']:
            run_one(root, name, run_id, config, x, edges, stats, device, args.resume, args.artifact_folder)


if __name__ == '__main__':
    main()
