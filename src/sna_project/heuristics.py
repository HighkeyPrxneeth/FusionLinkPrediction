import numpy as np

NAMES = ['CN', 'JC', 'AA', 'RA', 'PA']


def adjacency_sets(n, edges):
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(int(v))
        neighbors[v].add(int(u))
    return neighbors


def compute(neighbors, pairs):
    degrees = np.asarray([len(s) for s in neighbors], dtype=np.float64)
    result = np.zeros((len(pairs), 5), dtype=np.float64)
    for i, (u, v) in enumerate(pairs):
        common = neighbors[u] & neighbors[v]
        cn = len(common)
        union = degrees[u] + degrees[v] - cn
        aa = sum(1 / np.log(degrees[w]) for w in common)
        ra = sum(1 / degrees[w] for w in common)
        result[i] = [cn, cn / union if union else 0, aa, ra, degrees[u] * degrees[v]]
    return result


def transform_features(raw, vgae_score=None):
    h = np.asarray(raw, dtype=np.float64).copy()
    h[:, [0, 2, 3, 4]] = np.log1p(h[:, [0, 2, 3, 4]])
    return h if vgae_score is None else np.column_stack([vgae_score, h])
