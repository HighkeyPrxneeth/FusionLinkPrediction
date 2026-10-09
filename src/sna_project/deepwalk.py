from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from numba import njit

from .utils import atomic_json


@njit(cache=True)
def walk_matrix(indptr, indices, n, rounds, length, seed):
    np.random.seed(seed)
    walks = np.full((n * rounds, length), -1, dtype=np.int32)
    row = 0
    for _ in range(rounds):
        for u in np.random.permutation(n):
            walks[row, 0] = u
            current = u
            for t in range(1, length):
                start, end = indptr[current], indptr[current + 1]
                if start == end:
                    break
                current = indices[np.random.randint(start, end)]
                walks[row, t] = current
            row += 1
    return walks


def stable_hash(value):
    import hashlib
    return int.from_bytes(hashlib.md5(value.encode()).digest()[:4], 'little')


def train_deepwalk(n, train_edges, config, directory: Path, seed):
    from scipy.sparse import csr_matrix
    started = time.perf_counter()
    both = np.concatenate([train_edges, train_edges[:, ::-1]])
    matrix = csr_matrix((np.ones(len(both)), (both[:, 0], both[:, 1])), shape=(n, n))
    corpus = directory / 'deepwalk_corpus.txt'
    # One integer matrix per walk round keeps memory bounded; the text corpus is reusable.
    with corpus.open('w', encoding='ascii') as output:
        for round_id in range(config['deepwalk_walks']):
            walks = walk_matrix(matrix.indptr, matrix.indices, n, 1, config['deepwalk_length'], seed + round_id)
            for walk in walks:
                output.write(' '.join(str(int(v)) for v in walk if v >= 0) + '\n')
    walk_seconds = time.perf_counter() - started
    model = Word2Vec(vector_size=config['deepwalk_dim'], window=config['deepwalk_window'], min_count=1,
                     sg=1, hs=1, negative=0, workers=1, seed=seed, hashfxn=stable_hash,
                     sample=1e-3, epochs=config['deepwalk_epochs'], sorted_vocab=1)
    model.build_vocab(corpus_file=str(corpus))
    model.train(corpus_file=str(corpus), total_words=model.corpus_total_words, epochs=config['deepwalk_epochs'])
    embedding = np.stack([model.wv[str(u)] for u in range(n)])
    model.save(str(directory / 'checkpoints' / 'deepwalk.model'))
    info = {'method': 'DeepWalk', 'seconds': time.perf_counter() - started, 'walk_seconds': walk_seconds,
            'corpus_words': model.corpus_total_words, 'objective': 'Skip-gram hierarchical softmax',
            'negative_sampling': 0, 'subsampling': 1e-3, 'workers': 1, 'seed': seed,
            'scoring': 'input embedding dot product', 'isolates': 'one-token walks retain every node'}
    atomic_json(directory / 'deepwalk_training.json', info)
    print(f'  DeepWalk completed in {info["seconds"]:.1f}s', flush=True)
    return embedding, info
