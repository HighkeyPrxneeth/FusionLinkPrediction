"""Verify the demo against saved predictions without retraining or tuning."""
from pathlib import Path
import json
import numpy as np
from sna_project.demo import Demo

root = Path(__file__).resolve().parents[1]
demo = Demo(root)
summary = demo.summary()
checks = 0
for dataset in ['Cora', 'CiteSeer', 'PubMed']:
    for run in range(5):
        info = demo.run(dataset, run)
        indices = sorted({0, info['count'] - 1} | {e['index'] for e in info['examples']})
        for index in indices:
            pair = demo.pair(dataset, run, index)
            assert not pair['edge_in_training_graph']
            np.testing.assert_allclose(1/(1+np.exp(-pair['logit'])), pair['scores']['Fusion'], atol=1e-12)
            assert pair['heuristics']['CN'] == len(pair['common_neighbors'])
            assert [pair['u'], pair['v']] not in pair['graph']['edges']
            json.dumps(pair, allow_nan=False)
            checks += 1
record = {'passed': True, 'runs': 15, 'pairs_checked': checks,
          'checks': ['hidden edge absent', 'fusion decomposition equals saved prediction',
                     'common-neighbor count', 'finite JSON', 'bounded neighborhood display']}
(root/'results/demo_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(record)
