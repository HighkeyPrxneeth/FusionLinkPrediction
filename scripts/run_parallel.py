"""Run independent existing-protocol trials concurrently; no shared run directory."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import argparse
import json
import torch

from sna_project.data import load_dataset
from sna_project.run import run_one
from sna_project.utils import atomic_json, source_hash


def worker(root, name, run_id, config, x, edges, stats):
    torch.set_num_threads(config['torch_threads'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_one(root, name, run_id, config, x, edges, stats, device, True)
    return run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['Cora','CiteSeer','PubMed'], default='PubMed')
    parser.add_argument('--runs', type=int, nargs='+', default=[1,2,3,4])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'artifacts/config.json').read_text())
    if len(set(args.runs)) != len(args.runs) or any(r not in config['run_ids'] for r in args.runs):
        raise ValueError('Use distinct registered run IDs')
    if not 1 <= args.workers <= 4:
        raise ValueError('Choose one to four workers')
    x, edges, stats = load_dataset(root, args.dataset)
    atomic_json(root/'artifacts/parallel_execution.json', {'dataset':args.dataset,'runs':args.runs,
        'workers':args.workers,'source_hash':source_hash(), 'protocol_changed':False,
        'note':'Independent trials; each DeepWalk keeps one training worker. Runtime reflects concurrent workload.'})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(worker, root, args.dataset, run_id, config, x, edges, stats) for run_id in args.runs]
        for job in as_completed(jobs):
            print('Completed parallel run',job.result(),flush=True)


if __name__ == '__main__':
    main()
