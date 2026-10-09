from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
from torch_geometric.datasets import AttributedGraphDataset, Planetoid
from torch_geometric.transforms import NormalizeFeatures

from .utils import atomic_json, digest_array, digest_bytes


def canonical_edges(edge_index, num_nodes: int) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.shape[0] == 2:
        edges = edges.T
    edges = np.sort(edges, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if np.any(edges < 0) or np.any(edges >= num_nodes):
        raise ValueError('Edge endpoint outside node range')
    return np.unique(edges, axis=0)


def load_dataset(root: Path, name: str):
    directory = root / 'data' / 'raw' / name
    if name == 'Facebook':
        ds = AttributedGraphDataset(str(directory), name=name, transform=NormalizeFeatures())
        source = 'https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.AttributedGraphDataset.html'
    else:
        ds = Planetoid(str(directory), name=name, transform=NormalizeFeatures())
        source = 'https://github.com/kimiyoung/planetoid/tree/master/data'
    data = ds[0]
    edges = canonical_edges(data.edge_index.numpy(), data.num_nodes)
    x = data.x.float()
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(edges.tolist())
    hashes = {str(p.relative_to(directory)): digest_bytes(p.read_bytes()) for p in sorted(directory.rglob('*')) if p.is_file() and 'raw' in p.relative_to(directory).parts}
    provenance_path = root / 'data' / 'processed' / name / 'statistics.json'
    retrieved = datetime.now(timezone.utc).isoformat()
    if provenance_path.exists():
        import json
        retrieved = json.loads(provenance_path.read_text())['retrieved_utc']
    stats = {
        'dataset': name, 'source': source, 'retrieved_utc': retrieved,
        'nodes': data.num_nodes, 'stored_edges': int(data.edge_index.shape[1]),
        'unique_edges': len(edges), 'features': int(x.shape[1]),
        'density': nx.density(graph), 'mean_degree': 2 * len(edges) / data.num_nodes,
        'isolates': nx.number_of_isolates(graph), 'components': nx.number_connected_components(graph),
        'average_clustering': nx.average_clustering(graph),
        'edge_hash': digest_array(edges), 'feature_hash': digest_array(x.numpy()),
        'raw_file_sha256': hashes, 'normalization': 'PyG NormalizeFeatures row normalization',
    }
    atomic_json(provenance_path, stats)
    return x, edges, stats
