from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import atomic_json, atomic_npz, digest_array


def pair_keys(pairs: np.ndarray, n: int):
    p = np.sort(np.asarray(pairs, dtype=np.int64), axis=1)
    return p[:, 0] * n + p[:, 1]


def sample_negatives(n: int, count: int, excluded_keys, rng: np.random.Generator):
    excluded = np.unique(np.asarray(excluded_keys, dtype=np.int64))
    available = n * (n - 1) // 2 - len(excluded)
    if count > available:
        raise ValueError('Not enough distinct nonedges')
    chosen = np.empty(0, dtype=np.int64)
    while len(chosen) < count:
        size = max(1024, int((count - len(chosen)) * 1.8))
        u = rng.integers(0, n, size=size)
        v = rng.integers(0, n, size=size)
        mask = u != v
        keys = np.minimum(u[mask], v[mask]) * n + np.maximum(u[mask], v[mask])
        keys = np.unique(keys)
        keys = keys[~np.isin(keys, excluded, assume_unique=True)]
        chosen = np.union1d(chosen, keys)
    rng.shuffle(chosen)
    chosen = chosen[:count]
    return np.stack([chosen // n, chosen % n], axis=1)


def validate_split(edges, split, n):
    original = set(pair_keys(edges, n).tolist())
    seen_pos, seen_neg = set(), set()
    for name in ['train', 'selection', 'fusion', 'test']:
        p = split[name + '_pos']
        assert np.all(p[:, 0] < p[:, 1]), 'Noncanonical positive pair'
        keys = set(pair_keys(p, n).tolist())
        assert len(keys) == len(p) and not (keys & seen_pos), 'Overlapping positive partitions'
        seen_pos |= keys
        if name == 'train':
            continue
        q = split[name + '_neg']
        assert len(q) == len(p), 'Unbalanced evaluation labels'
        assert np.all(q[:, 0] < q[:, 1]), 'Noncanonical negative pair'
        neg = set(pair_keys(q, n).tolist())
        assert len(neg) == len(q) and not (neg & original) and not (neg & seen_neg), 'Invalid negative pairs'
        seen_neg |= neg
    assert original == seen_pos, 'Positive edges lost or invented'
    return {'passed': True, 'positive_disjoint': True, 'negative_disjoint': True, 'no_false_negative': True, 'no_self_pairs': True}


def make_split(edges, n, config, seed):
    rng = np.random.default_rng(seed)
    shuffled = edges[rng.permutation(len(edges))]
    counts = [int(len(edges) * config[k + '_fraction']) for k in ['selection', 'fusion', 'test']]
    if min(counts) < 1:
        raise ValueError('Graph too small for requested partition')
    a = len(edges) - sum(counts)
    b, c = a + counts[0], a + counts[0] + counts[1]
    result = dict(zip(['train_pos', 'selection_pos', 'fusion_pos', 'test_pos'], [shuffled[:a], shuffled[a:b], shuffled[b:c], shuffled[c:]]))
    negatives = sample_negatives(n, sum(counts), pair_keys(edges, n), rng)
    offset = 0
    for name, count in zip(['selection', 'fusion', 'test'], counts):
        result[name + '_neg'] = negatives[offset:offset + count]
        offset += count
    validate_split(edges, result, n)
    return result


def save_split(directory: Path, edges, split, n, seed):
    atomic_npz(directory / 'split.npz', **split)
    graph = np.asarray(sorted(split['train_pos'].tolist()), dtype=np.int64)
    manifest = {'seed': seed, 'nodes': n, 'original_hash': digest_array(edges), 'training_graph_hash': digest_array(graph), 'partitions': {k: {'count': len(v), 'sha256': digest_array(v)} for k, v in split.items()}, 'invariants': validate_split(edges, split, n)}
    atomic_json(directory / 'split_manifest.json', manifest)
    return manifest


def labeled_pairs(split, name):
    p, q = split[name + '_pos'], split[name + '_neg']
    return np.concatenate([p, q]), np.concatenate([np.ones(len(p), dtype=np.int64), np.zeros(len(q), dtype=np.int64)])
