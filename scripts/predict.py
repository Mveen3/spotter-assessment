"""Regenerate submission files from a trusted local model bundle."""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle

from freight.data import read_inputs
from run import write_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='artifacts/model_bundle.pkl')
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--output', default='.')
    args = parser.parse_args()
    with open(args.model, 'rb') as handle:
        bundle = pickle.load(handle)
    frames, _ = read_inputs(args.data_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    submission, december = write_outputs(bundle, frames, output)
    print(f'Created {len(submission):,} load predictions and {len(december)} December predictions in {output}')
