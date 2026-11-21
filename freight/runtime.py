from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import resource
import subprocess
import time
from pathlib import Path

from freight.data import atomic_json, file_hash


def fingerprint(root, paths, config):
    sources = [Path(root) / 'run.py', Path(root) / 'score.py'] + sorted((Path(root) / 'freight').glob('*.py'))
    value = {'config': config, 'data': {k: file_hash(p) for k, p in paths.items()},
             'source': {p.name: file_hash(p) for p in sources},
             'versions': {name: importlib.metadata.version(name) for name in
                          ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'matplotlib']}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), value


def atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


class RunState:
    def __init__(self, output, config_hash, total=40):
        self.output, self.config_hash = Path(output), config_hash
        self.ckpt = self.output / 'checkpoints' / config_hash[:16]
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.start, self.done, self.total = time.monotonic(), 0, total

    def progress(self, stage, completed=False):
        if completed:
            self.done += 1
        elapsed = time.monotonic() - self.start
        eta = elapsed / self.done * max(0, self.total - self.done) if self.done else None
        gpu = 'unavailable'
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                 text=True, capture_output=True, timeout=2, check=True).stdout.strip() + ' MiB'
        except (OSError, subprocess.SubprocessError):
            pass
        value = {'stage': stage, 'completed_tasks': self.done, 'planned_tasks': self.total,
                 'percent': min(100., 100 * self.done / self.total), 'elapsed_seconds': elapsed,
                 'eta_seconds': eta, 'peak_ram_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                 'gpu_memory': gpu, 'config_hash': self.config_hash}
        atomic_json(self.output / 'progress.json', value)
        duration = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        print(f'[stage {self.done}/{self.total} {stage}] elapsed {duration} | task ETA unknown | total ETA '
              f'{int(eta) if eta is not None else "unknown"}s | peak RAM {value["peak_ram_mib"]:.0f} MiB | GPU {gpu}', flush=True)

    def task(self, name, function):
        path = self.ckpt / (name + '.pkl')
        marker = self.ckpt / (name + '.stage_done.json')
        if path.exists() and marker.exists():
            metadata = json.loads(marker.read_text())
            if metadata['config_hash'] == self.config_hash and metadata['sha256'] == file_hash(path):
                # Only load trusted, locally generated checkpoints.
                with path.open('rb') as handle:
                    value = pickle.load(handle)
                self.progress(name + ' resumed', completed=True)
                return value
        self.progress(name)
        value = function()
        atomic_pickle(path, value)
        atomic_json(marker, {'config_hash': self.config_hash, 'sha256': file_hash(path), 'stage_done': True})
        self.progress(name, completed=True)
        return value
