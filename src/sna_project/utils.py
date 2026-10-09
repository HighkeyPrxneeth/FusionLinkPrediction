from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_array(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    return digest_bytes(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes())


def digest_json(value) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True).encode())


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_npz(path: Path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **values)
    os.replace(temp, path)


def atomic_torch(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def source_hash() -> str:
    names = ['utils.py', 'data.py', 'splits.py', 'heuristics.py', 'models.py', 'deepwalk.py', 'fusion.py', 'run.py']
    root = Path(__file__).parent
    return digest_bytes(b''.join((root / name).read_bytes() for name in names))


def environment():
    import importlib.metadata as metadata
    packages = ['torch', 'torch-geometric', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'gensim', 'numba']
    return {
        'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(), 'cpu_count': os.cpu_count(),
        'executable': os.sys.executable,
        'packages': {p: metadata.version(p) for p in packages},
        'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'determinism': 'seeded; deterministic algorithms warn_only; GPU reductions may vary',
    }
