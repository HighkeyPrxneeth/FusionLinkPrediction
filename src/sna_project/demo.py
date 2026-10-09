"""Local, dependency-free demo of measured benchmark results and held-out pairs."""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from .heuristics import NAMES, adjacency_sets


class Demo:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def summary(self):
        manifest = json.loads((self.root / 'results/export_manifest.json').read_text())
        if not manifest['complete']:
            raise ValueError('The demo requires the complete audited benchmark')
        frame = pd.read_csv(self.root / 'results/tables/summary.csv')
        records = frame.astype(object).where(pd.notnull(frame), None).to_dict('records')
        stats = {p.parent.name: json.loads(p.read_text()) for p in (self.root / 'data/processed').glob('*/statistics.json')}
        env = json.loads((self.root / 'artifacts/environment.json').read_text())
        return {'results': records, 'statistics': stats, 'manifest': manifest, 'environment': env}

    @lru_cache(maxsize=6)
    def load(self, dataset, run_id):
        if dataset not in ['Cora', 'CiteSeer', 'PubMed'] or run_id not in range(5):
            raise ValueError('Unknown benchmark/run')
        directory = self.root / 'artifacts' / dataset / f'run_{run_id}'
        with np.load(directory / 'predictions.npz') as file:
            predictions = {k: file[k] for k in file.files}
        with np.load(directory / 'pair_features.npz') as file:
            features = file['test_features']
        with np.load(directory / 'split.npz') as file:
            train = file['train_pos']
        stats = json.loads((self.root / 'data/processed' / dataset / 'statistics.json').read_text())
        neighbors = adjacency_sets(stats['nodes'], train)
        coefficients = json.loads((directory / 'coefficients.json').read_text())['Fusion']
        record = json.loads((directory / 'metrics.json').read_text())
        return predictions, features, neighbors, coefficients, record

    def run(self, dataset, run_id):
        pred, _, _, _, record = self.load(dataset, run_id)
        labels, scores = pred['labels'], pred['Fusion']
        positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        choices = [('Highest-scoring hidden link', positive[np.argmax(scores[positive])]),
                   ('Lowest-scoring hidden link', positive[np.argmin(scores[positive])]),
                   ('Highest-scoring non-edge', negative[np.argmax(scores[negative])]),
                   ('Lowest-scoring non-edge', negative[np.argmin(scores[negative])])]
        choices += [('No-common-neighbor example', i) for i in np.flatnonzero(pred['heuristics'][:, 0] == 0)[:2]]
        return {'metrics': record['results'], 'count': len(labels),
                'examples': [{'label': title, 'index': int(index)} for title, index in choices]}

    def pair(self, dataset, run_id, index):
        pred, features, neighbors, coefficient, _ = self.load(dataset, run_id)
        if not 0 <= index < len(pred['labels']):
            raise ValueError('Pair index outside this test set')
        u, v = map(int, pred['pairs'][index])
        common = neighbors[u] & neighbors[v]
        standardized = (features[index] - np.array(coefficient['scaler_mean'])) / np.array(coefficient['scaler_scale'])
        contributions = standardized * np.array(coefficient['coef'])
        details = [{'feature': name, 'transformed': float(features[index, j]), 'standardized': float(standardized[j]),
                    'coefficient': coefficient['coef'][j], 'contribution': float(contributions[j])}
                   for j, name in enumerate(['VGAE'] + NAMES)]
        others = sorted(common) + sorted((neighbors[u] | neighbors[v]) - common - {u, v})
        shown = [u, v] + others[:28]
        visible = set(shown)
        graph_edges = sorted({tuple(sorted((a, b))) for a in shown for b in neighbors[a] if b in visible})
        return {'index': index, 'u': u, 'v': v, 'label': int(pred['labels'][index]),
                'scores': {k: float(pred[k][index]) for k in ['Fusion', 'VGAE', 'GAE', 'DeepWalk']},
                'heuristics': dict(zip(NAMES, map(float, pred['heuristics'][index]))),
                'degree_u': len(neighbors[u]), 'degree_v': len(neighbors[v]),
                'common_neighbors': sorted(common), 'contributions': details, 'intercept': coefficient['intercept'],
                'logit': float(coefficient['intercept'] + contributions.sum()),
                'graph': {'nodes': [{'id': a, 'role': 'u' if a == u else 'v' if a == v else 'common' if a in common else 'neighbor'} for a in shown],
                          'edges': graph_edges, 'truncated': len(others) > 28},
                'edge_in_training_graph': v in neighbors[u]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    demo = Demo(args.root)
    demo.summary()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path)
            query = parse_qs(route.query)
            try:
                if route.path.startswith('/api/'):
                    dataset = query.get('dataset', ['Cora'])[0]
                    run_id = int(query.get('run', ['0'])[0])
                    if route.path == '/api/summary':
                        payload = demo.summary()
                    elif route.path == '/api/run':
                        payload = demo.run(dataset, run_id)
                    elif route.path == '/api/pair':
                        payload = demo.pair(dataset, run_id, int(query.get('index', ['0'])[0]))
                    else:
                        raise ValueError('Unknown endpoint')
                    data, content_type = json.dumps(payload, allow_nan=False).encode(), 'application/json'
                else:
                    allowed = {'/': 'demo/index.html', '/report': 'output/pdf/SNA_Project_Report.pdf'}
                    if route.path not in allowed:
                        self.send_error(404)
                        return
                    data = (demo.root / allowed[route.path]).read_bytes()
                    content_type = 'application/pdf' if route.path == '/report' else 'text/html; charset=utf-8'
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, FileNotFoundError) as error:
                data = json.dumps({'error': str(error)}).encode()
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Demo ready: http://127.0.0.1:{args.port} (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
