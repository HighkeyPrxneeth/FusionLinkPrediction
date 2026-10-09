from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .fusion import metrics
from .utils import atomic_json, digest_bytes, digest_array
from .splits import validate_split, labeled_pairs
from .heuristics import adjacency_sets, compute

MAIN = ['CN', 'JC', 'AA', 'RA', 'PA', 'DeepWalk', 'GAE', 'VGAE', 'Heuristics LR', 'VGAE score LR', 'Fusion']


def markdown_table(frame):
    head = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    rule = '| ' + ' | '.join('---' for _ in frame.columns) + ' |'
    return '\n'.join([head, rule] + ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def export(root: Path, artifact_folder='artifacts', allow_partial=False):
    base = root / artifact_folder
    config = json.loads((base / 'config.json').read_text())
    tabledir, figuredir = root / 'results' / 'tables', root / 'results' / 'figures'
    tabledir.mkdir(parents=True, exist_ok=True)
    figuredir.mkdir(parents=True, exist_ok=True)
    expected = {(d, r) for d in config['datasets'] for r in config['run_ids']}
    found, rows, timing_rows, coefficients = set(), [], [], []
    files = sorted(base.glob('*/run_*/metrics.json'))
    source_hashes = set()
    for file in files:
        record = json.loads(file.read_text())
        d, run_id = record['dataset'], record['run_id']
        if (d, run_id) not in expected:
            continue
        found.add((d, run_id))
        identity = json.loads((file.parent / 'identity.json').read_text())
        source_hashes.add(identity['source_hash'])
        assert len(set(record['method_graph_hashes'].values())) == 1
        manifest = json.loads((file.parent / 'split_manifest.json').read_text())
        with np.load(file.parent / 'split.npz') as saved_split:
            split = {k: saved_split[k] for k in saved_split.files}
        original = np.unique(np.concatenate([split[k + '_pos'] for k in ['train', 'selection', 'fusion', 'test']]), axis=0)
        assert all(np.all((pairs >= 0) & (pairs < manifest['nodes'])) for pairs in split.values())
        validate_split(original, split, manifest['nodes'])
        assert digest_array(original) == manifest['original_hash'] == identity['edge_hash']
        actual_graph_hash = digest_array(np.asarray(sorted(split['train_pos'].tolist()), dtype=np.int64))
        assert actual_graph_hash == record['training_graph_hash'] == manifest['training_graph_hash']
        assert all(value == actual_graph_hash for value in record['method_graph_hashes'].values())
        for key, pairs in split.items():
            assert digest_array(pairs) == manifest['partitions'][key]['sha256']
        with np.load(file.parent / 'predictions.npz') as saved:
            y = saved['labels']
            expected_pairs, expected_labels = labeled_pairs(split, 'test')
            np.testing.assert_array_equal(saved['pairs'], expected_pairs)
            np.testing.assert_array_equal(y, expected_labels)
            audited_heuristics = compute(adjacency_sets(manifest['nodes'], split['train_pos']), expected_pairs)
            np.testing.assert_allclose(saved['heuristics'], audited_heuristics, rtol=1e-12, atol=1e-12)
            group_masks = {'all': np.ones(len(y), dtype=bool), 'CN=0': audited_heuristics[:, 0] == 0,
                           'CN>0': audited_heuristics[:, 0] > 0}
            with np.load(file.parent / 'pair_features.npz') as features:
                fusion_pairs, fusion_labels = labeled_pairs(split, 'fusion')
                np.testing.assert_array_equal(features['fusion_pairs'], fusion_pairs)
                np.testing.assert_array_equal(features['fusion_labels'], fusion_labels)
                for model in json.loads((file.parent / 'coefficients.json').read_text()).values():
                    columns = [(['VGAE'] + ['CN', 'JC', 'AA', 'RA', 'PA']).index(name) for name in model['feature_names']]
                    np.testing.assert_allclose(model['scaler_mean'], features['fusion_features'][:, columns].mean(axis=0), rtol=1e-10, atol=1e-10)
            for method, groups in record['results'].items():
                audited = metrics(y, saved[method])
                assert abs(audited['auc'] - groups['all']['auc']) < 1e-12
                assert abs(audited['ap'] - groups['all']['ap']) < 1e-12
                for group, values in groups.items():
                    audited_group = metrics(y[group_masks[group]], saved[method][group_masks[group]])
                    for key, value in audited_group.items():
                        if value is None:
                            assert values[key] is None
                        else:
                            assert abs(value - values[key]) < 1e-12
                    rows.append({'dataset': d, 'run_id': run_id, 'method': method, 'group': group, **values})
            if run_id == 0:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for method in ['VGAE', 'GAE', 'DeepWalk', 'Heuristics LR', 'Fusion']:
                    fpr, tpr, _ = roc_curve(y, saved[method])
                    axes[0].plot(fpr, tpr, label=method)
                    precision, recall, _ = precision_recall_curve(y, saved[method])
                    axes[1].plot(recall, precision, label=method)
                axes[0].plot([0, 1], [0, 1], '--', color='gray', lw=1)
                axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title=f'{d} ROC (preselected run 0)')
                axes[1].set(xlabel='Recall', ylabel='Precision', title=f'{d} precision recall (balanced pairs)')
                axes[1].legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(figuredir / f'{d.lower()}_curves.png', dpi=180)
                plt.close(fig)
        times = json.loads((file.parent / 'timings.json').read_text())
        timing_rows.extend({'dataset': d, 'run_id': run_id, 'stage': k, 'seconds': v} for k, v in times.items())
        coef = json.loads((file.parent / 'coefficients.json').read_text())['Fusion']
        coefficients.extend({'dataset': d, 'run_id': run_id, 'feature': feature, 'coefficient': value} for feature, value in zip(coef['feature_names'], coef['coef']))
    if found != expected and not allow_partial:
        raise RuntimeError(f'Missing experiments: {sorted(expected - found)}. Use --allow-partial only for pilot reporting.')
    if not rows:
        raise RuntimeError('No completed experiments')
    if len(source_hashes) != 1:
        raise RuntimeError('Mixed source revisions')
    raw = pd.DataFrame(rows)
    raw.to_csv(tabledir / 'per_run_metrics.csv', index=False)
    all_rows = raw[raw.group == 'all']
    summary = all_rows.groupby(['dataset', 'method']).agg(runs=('ap', 'size'), auc_mean=('auc', 'mean'), auc_std=('auc', 'std'), ap_mean=('ap', 'mean'), ap_std=('ap', 'std')).reset_index()
    summary.to_csv(tabledir / 'summary.csv', index=False)
    summary[summary.method.str.startswith('Fusion')].to_csv(tabledir / 'ablations.csv', index=False)
    subgroups = raw.groupby(['dataset', 'method', 'group']).agg(auc_mean=('auc', 'mean'), ap_mean=('ap', 'mean'), count_mean=('count', 'mean'), positives_mean=('positives', 'mean'), prevalence_mean=('prevalence', 'mean')).reset_index()
    subgroups.to_csv(tabledir / 'subgroups.csv', index=False)
    paired = []
    for dataset, part in all_rows.groupby('dataset'):
        matrix = part.pivot(index='run_id', columns='method', values=['auc', 'ap'])
        for baseline in ['VGAE', 'GAE', 'DeepWalk', 'Heuristics LR', 'VGAE score LR']:
            for metric in ['auc', 'ap']:
                differences = matrix[metric]['Fusion'] - matrix[metric][baseline]
                paired.append({'dataset': dataset, 'baseline': baseline, 'metric': metric, 'mean_difference': differences.mean(), 'std_difference': differences.std(ddof=1), 'wins': int((differences > 0).sum()), 'runs': len(differences), 'per_run_differences': json.dumps(differences.tolist())})
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(tabledir / 'paired_differences.csv', index=False)
    pd.DataFrame(timing_rows).to_csv(tabledir / 'timings.csv', index=False)
    coef_frame = pd.DataFrame(coefficients)
    coef_frame.to_csv(tabledir / 'coefficients.csv', index=False)
    formatted = summary[summary.method.isin(MAIN)].copy()
    formatted['AUC mean ± SD'] = formatted.apply(lambda row: f'{100*row.auc_mean:.2f} ± {100*row.auc_std:.2f}', axis=1)
    formatted['AP mean ± SD'] = formatted.apply(lambda row: f'{100*row.ap_mean:.2f} ± {100*row.ap_std:.2f}', axis=1)
    (tabledir / 'MAIN_RESULTS.md').write_text(markdown_table(formatted[['dataset', 'method', 'runs', 'AUC mean ± SD', 'AP mean ± SD']]), encoding='utf-8')
    for dataset, part in summary.groupby('dataset'):
        ordered = part.set_index('method').reindex([m for m in MAIN if m in set(part.method)])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        xx = np.arange(len(ordered))
        ax.bar(xx - .18, ordered.auc_mean * 100, .36, yerr=ordered.auc_std.fillna(0) * 100, label='AUC')
        ax.bar(xx + .18, ordered.ap_mean * 100, .36, yerr=ordered.ap_std.fillna(0) * 100, label='AP')
        ax.set_xticks(xx, ordered.index, rotation=35, ha='right')
        ax.set(ylabel='Score (%)', ylim=(0, 100), title=f'{dataset}: mean and sample standard deviation')
        ax.legend()
        fig.tight_layout()
        fig.savefig(figuredir / f'{dataset.lower()}_comparison.png', dpi=180)
        plt.close(fig)
        cp = coef_frame[coef_frame.dataset == dataset].groupby('feature').coefficient.agg(['mean', 'std']).reindex(['VGAE', 'CN', 'JC', 'AA', 'RA', 'PA'])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(cp.index, cp['mean'], yerr=cp['std'].fillna(0))
        ax.axhline(0, color='gray', lw=1)
        ax.set(ylabel='Standardized logistic coefficient', title=f'{dataset}: coefficients are not causal importance')
        fig.tight_layout()
        fig.savefig(figuredir / f'{dataset.lower()}_coefficients.png', dpi=180)
        plt.close(fig)
        ablation = part[part.method.str.startswith('Fusion')].set_index('method').ap_mean
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.barh(ablation.index, ablation * 100)
        ax.set(xlabel='Average precision (%)', xlim=(0, 100), title=f'{dataset} fusion ablations')
        fig.tight_layout()
        fig.savefig(figuredir / f'{dataset.lower()}_ablation.png', dpi=180)
        plt.close(fig)
    times = pd.DataFrame(timing_rows)
    means = times[~times.stage.isin(['current_invocation_seconds'])].groupby(['dataset', 'stage']).seconds.mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    means.plot.bar(stacked=True, ax=ax)
    ax.set(ylabel='Mean measured seconds per run', title='Runtime by stage')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figuredir / 'runtime.png', dpi=180)
    plt.close(fig)
    exported = {str(p.relative_to(root)): digest_bytes(p.read_bytes()) for p in sorted((root / 'results').rglob('*')) if p.is_file() and p.name != 'export_manifest.json'}
    atomic_json(root / 'results' / 'export_manifest.json', {'complete': found == expected, 'expected': len(expected), 'completed': len(found), 'missing': sorted(expected - found), 'source_hash': next(iter(source_hashes)), 'prediction_metrics_recomputed': True, 'split_hashes_verified': True, 'fusion_scaler_fit_verified': True, 'files': exported})
    print(f'Exported {len(found)}/{len(expected)} experiments and {len(exported)} result files.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--artifact-folder', default='artifacts')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    export(Path(args.root).resolve(), args.artifact_folder, args.allow_partial)


if __name__ == '__main__':
    main()
