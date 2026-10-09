import numpy as np
import pytest
import torch
import networkx as nx
from pathlib import Path

from sna_project.data import canonical_edges
from sna_project.splits import make_split, pair_keys, sample_negatives, validate_split
from sna_project.heuristics import adjacency_sets, compute, transform_features
from sna_project.fusion import fit_predict, metrics
from sna_project.models import Encoder, mean_kl, train_encoder


def config():
    return {'selection_fraction': .05, 'fusion_fraction': .05, 'test_fraction': .1,
            'hidden_dim': 4, 'latent_dim': 2, 'learning_rate': .01, 'epochs': 2, 'selection_every': 1}


def graph_edges():
    return np.asarray([(u, v) for u in range(12) for v in range(u + 1, 12) if (u + v) % 3 == 0])


def test_canonicalization_removes_reverse_duplicates_and_loops():
    raw = np.array([[0, 1, 0, 2, 2], [1, 0, 0, 3, 3]])
    assert np.array_equal(canonical_edges(raw, 4), [[0, 1], [2, 3]])


def test_split_disjoint_balanced_reproducible():
    edges = graph_edges()
    a, b = make_split(edges, 12, config(), 42), make_split(edges, 12, config(), 42)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert validate_split(edges, a, 12)['passed']
    train = set(pair_keys(a['train_pos'], 12))
    assert not train.intersection(pair_keys(a['test_pos'][:, ::-1], 12))


def test_split_validator_catches_leakage():
    edges = graph_edges()
    a = make_split(edges, 12, config(), 42)
    a['test_pos'][0] = a['train_pos'][0]
    with pytest.raises(AssertionError):
        validate_split(edges, a, 12)


def test_negative_sampler_exclusions_and_capacity():
    edges = graph_edges()
    excluded = pair_keys(edges, 12)
    sampled = sample_negatives(12, 10, excluded, np.random.default_rng(4))
    assert len(set(pair_keys(sampled, 12))) == 10
    assert not set(pair_keys(sampled, 12)).intersection(excluded)
    with pytest.raises(ValueError):
        sample_negatives(12, 100, excluded, np.random.default_rng(4))


def test_heuristics_against_networkx_and_hand_calculation():
    edges = np.array([[0, 2], [1, 2], [0, 3], [1, 3], [3, 4]])
    neighbors = adjacency_sets(7, edges)
    pairs = np.array([[0, 1], [0, 4], [5, 6], [1, 0]])
    result = compute(neighbors, pairs)
    assert result[0, 0] == 2 and result[0, 1] == 1 and result[0, 4] == 4
    assert result[0, 2] == pytest.approx(1 / np.log(2) + 1 / np.log(3))
    assert result[0, 3] == pytest.approx(.5 + 1 / 3)
    assert np.array_equal(result[0], result[3])
    assert np.all(result[2] == 0)
    graph = nx.Graph()
    graph.add_nodes_from(range(7))
    graph.add_edges_from(edges)
    for column, reference in [(1, nx.jaccard_coefficient), (2, nx.adamic_adar_index), (3, nx.resource_allocation_index), (4, nx.preferential_attachment)]:
        expected = np.array([s for _, _, s in reference(graph, [tuple(p) for p in pairs])])
        np.testing.assert_allclose(result[:, column], expected)


def test_scaler_fits_only_fusion_pairs():
    fusion = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])
    y = np.array([0, 0, 1, 1])
    extreme_test = np.array([[1e6, -1e6]])
    _, fitted = fit_predict(fusion, y, extreme_test)
    np.testing.assert_allclose(fitted[0].mean_, fusion.mean(axis=0))


def test_metrics_ties_and_undefined_groups():
    assert metrics([0, 1], [0., 0.])['auc'] == .5
    assert metrics([0, 1], [0., 1.])['ap'] == 1
    assert metrics([1, 1], [0., 1.])['auc'] is None
    assert metrics([], [])['ap'] is None


def test_transform_preserves_raw_inputs():
    h = np.array([[2., .2, 1., .5, 4.]])
    transformed = transform_features(h, np.array([-3.]))
    assert transformed.shape == (1, 6)
    assert transformed[0, 0] == -3 and transformed[0, 2] == .2
    assert h[0, 0] == 2


def test_kl_log_standard_deviation_convention():
    mu, logstd = torch.zeros(3, 2), torch.zeros(3, 2)
    assert mean_kl(mu, logstd).item() == 0
    assert mean_kl(torch.ones(3, 2), logstd).item() == pytest.approx(1)


def test_encoder_uses_only_explicit_training_graph():
    torch.manual_seed(1)
    model = Encoder(3, 4, 2, True).eval()
    x = torch.rand(5, 3)
    training = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    heldout = {'edge_index': torch.tensor([[3, 4], [4, 3]])}
    with torch.no_grad():
        first = model(x, training)[0]
        heldout['edge_index'] = torch.tensor([[0, 4], [4, 0]])
        second = model(x, training)[0]
    assert torch.equal(first, second)


def test_completed_checkpoint_resume_matches_embedding(tmp_path):
    x = torch.rand(12, 3)
    edges = graph_edges()
    split = make_split(edges, 12, config(), 8)
    selection = np.concatenate([split['selection_pos'], split['selection_neg']])
    y = np.array([1, 0])
    reserved = np.concatenate([split[k + '_neg'] for k in ['selection', 'fusion', 'test']])
    args = (x, split['train_pos'], edges, reserved, selection, y, config(), tmp_path, 'VGAE', 1, 2, torch.device('cpu'))
    first, _ = train_encoder(*args, resume=False)
    second, info = train_encoder(*args, resume=True)
    np.testing.assert_array_equal(first, second)
    assert info['epochs_completed'] == 2


def test_interrupted_checkpoint_restores_optimizer_and_rng(tmp_path, monkeypatch):
    import sna_project.models as module
    from sna_project.utils import atomic_torch
    cfg = config() | {'epochs': 4}
    torch.manual_seed(99)
    x = torch.rand(12, 3)
    edges = graph_edges()
    split = make_split(edges, 12, cfg, 8)
    selection = np.concatenate([split['selection_pos'], split['selection_neg']])
    reserved = np.concatenate([split[k + '_neg'] for k in ['selection', 'fusion', 'test']])
    def execute(folder, resume):
        return train_encoder(x, split['train_pos'], edges, reserved, selection, np.array([1, 0]), cfg,
                             folder, 'VGAE', 11, 12, torch.device('cpu'), resume)
    reference, _ = execute(tmp_path / 'reference', False)
    def interrupt_after_committed_epoch(path, value):
        atomic_torch(path, value)
        if path.name.endswith('_latest.pt') and value['epoch'] == 2:
            raise InterruptedError('Simulated interruption after checkpoint commit')
    monkeypatch.setattr(module, 'atomic_torch', interrupt_after_committed_epoch)
    with pytest.raises(InterruptedError):
        execute(tmp_path / 'interrupted', False)
    monkeypatch.setattr(module, 'atomic_torch', atomic_torch)
    restored, _ = execute(tmp_path / 'interrupted', True)
    np.testing.assert_array_equal(reference, restored)
