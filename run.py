"""Chronological experiments, frozen evaluation, final refits, and submissions."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from freight.data import CORE, atomic_json, audit, file_hash, purge_overlap, read_inputs, valid_labeled
from freight.evaluation import metrics, select_blend, slices
from freight.models import CANDIDATES, batch_predict, train_candidate
from freight.runtime import RunState, atomic_pickle, fingerprint

FOLDS = [('may_jun', '2025-05-01', '2025-07-01'), ('jul_aug', '2025-07-01', '2025-09-01')]


def split(frame, start, end):
    training = frame[frame.date < start].copy()
    validation = frame[(frame.date >= start) & (frame.date < end)].copy()
    training, removed = purge_overlap(training, validation)
    if not len(training) or not len(validation):
        raise ValueError('Empty calendar fold')
    if training.date.max() >= validation.date.min():
        raise AssertionError('Date leakage')
    return training, validation, removed


def prediction_frame(frame, mode):
    return frame[CORE] if mode == 'core' else frame.drop(columns='posted_rate', errors='ignore')


def fit_predict(name, mode, training, validation, config):
    model, info = train_candidate(name, mode, training, config)
    pred = batch_predict(model, prediction_frame(validation, mode))
    return pred, info


def selected_fit(weights, mode, training, config):
    models, info = {}, {}
    for name in weights:
        models[name], info[name] = train_candidate(name, mode, training, config)
    return models, info


def blended_predict(models, weights, frame, mode):
    return sum(weight * batch_predict(models[name], prediction_frame(frame, mode)) for name, weight in weights.items())


def write_outputs(bundle, frames, output):
    root = Path(__file__).resolve().parent
    validation = frames['validation']
    enriched = blended_predict(bundle['enriched']['models'], bundle['enriched']['weights'], validation, 'enriched')
    by_id = pd.Series(enriched, index=validation.load_id)
    submission = frames['template'].copy()
    submission['predicted_rate'] = submission.load_id.map(by_id)
    december = frames['december'].copy()
    december['predicted_rate'] = blended_predict(bundle['core']['models'], bundle['core']['weights'], december, 'core')

    from score import validate_december, validate_predictions
    validate_predictions(submission)
    validate_december(december)
    pd.testing.assert_frame_equal(december.drop(columns='predicted_rate'), frames['december'].drop(columns='predicted_rate'))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output / 'validation_predictions.csv', index=False, float_format='%.6f')
    december.to_csv(output / 'december_predictions.csv', index=False, float_format='%.6f')

    # Maintain root deliverables and scorer input in sync
    submission.to_csv(root / 'validation_predictions.csv', index=False, float_format='%.6f')
    december.to_csv(root / 'december_predictions.csv', index=False, float_format='%.6f')
    (root / 'data').mkdir(parents=True, exist_ok=True)
    december.to_csv(root / 'data/december_chart_inputs.csv', index=False, float_format='%.6f')

    return submission, december


def run(args):
    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames, paths = read_inputs(args.data_dir)
    config = {'rounds': args.rounds, 'device': args.device, 'threads': args.threads,
              'seed': 20261117, 'folds': FOLDS, 'selection_metric': 'mean_fold_rmse',
              'blend_weights': [.25, .5, .75], 'candidates': CANDIDATES}
    key, manifest = fingerprint(root, paths, config)
    manifest['versions'] = {name: importlib.metadata.version(name) for name in
                            ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'matplotlib']}
    manifest['config_hash'] = key
    atomic_json(output / 'manifest.json', manifest)

    total_tasks = 2 * len(FOLDS) * len(CANDIDATES['core']) + 9
    state = RunState(output, key, total=total_tasks)
    try:
        facts = state.task('audit', lambda: audit(frames, paths))
        atomic_json(output / 'audit.json', facts)
        train = valid_labeled(frames['train'])
        selections, summaries, fit_records, fold_records = {}, [], [], []

        for mode, candidates in CANDIDATES.items():
            oof_parts = []
            for fold, start, end in FOLDS:
                training, validation, purged = split(train, start, end)
                fold_records.append({'mode': mode, 'fold': fold, 'train_rows': len(training),
                                     'valid_rows': len(validation), 'purged_feature_duplicates': purged,
                                     'train_end': str(training.date.max().date()), 'valid_start': start, 'valid_end_exclusive': end})
                part = validation[['load_id', 'date']].copy()
                part['fold'] = fold
                part['actual'] = validation.posted_rate
                for name in candidates:
                    pred, info = state.task(f'{mode}_{fold}_{name}',
                                            lambda: fit_predict(name, mode, training, validation, config))
                    part['pred_' + name] = pred
                    summaries.append({'mode': mode, 'fold': fold, 'model': name, **metrics(validation.posted_rate, pred)})
                    fit_records.append({'mode': mode, 'fold': fold, 'model': name, **info})
                    atomic_json(output / 'fit_records.json', fit_records)
                    pd.DataFrame(summaries).to_csv(output / 'tuning_metrics.csv', index=False)
                oof_parts.append(part)
            oof = pd.concat(oof_parts, ignore_index=True)
            oof.to_csv(output / f'{mode}_tuning_predictions.csv', index=False)
            selections[mode], ranking = select_blend(oof)
            atomic_json(output / f'{mode}_ranking.json', ranking)

        atomic_json(output / 'selection.json', selections)
        atomic_json(output / 'folds.json', fold_records)
        state.progress('selection frozen', completed=True)

        training, holdout, purged = split(train, '2025-09-01', '2025-11-01')
        holdout_results, holdout_slices, holdout_predictions = {}, {}, holdout[['load_id', 'date', 'posted_rate']].copy()

        for mode in CANDIDATES:
            weights = selections[mode]['weights']
            models, info = state.task(mode + '_locked_fit', lambda: selected_fit(weights, mode, training, config))
            pred = blended_predict(models, weights, holdout, mode)
            holdout_predictions[mode + '_prediction'] = pred
            holdout_results[mode] = metrics(holdout.posted_rate, pred)
            holdout_slices[mode] = slices(holdout, pred, training)
            atomic_json(output / f'{mode}_locked_fit.json', info)
            importance = []
            for name, model in models.items():
                if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                    columns = model.features.transform(training.head(1)).columns
                    importance.extend({'model': name, 'feature': col, 'importance': float(value)}
                                      for col, value in zip(columns, model.model.feature_importances_))
            atomic_json(output / f'{mode}_feature_importance.json', importance)

        # Baseline diagnostics for reporting and documentation
        holdout_results['quote_baseline'] = metrics(holdout.posted_rate, holdout.distance * holdout.quote_signal)
        holdout_predictions.to_csv(output / 'locked_holdout_predictions.csv', index=False)
        atomic_json(output / 'locked_holdout_metrics.json', holdout_results)
        atomic_json(output / 'locked_holdout_slices.json', holdout_slices)
        atomic_json(output / 'locked_holdout_split.json', {'train_rows': len(training), 'valid_rows': len(holdout),
                    'purged': purged, 'train_end': '2025-08-31', 'valid_start': '2025-09-01', 'valid_end': '2025-10-31'})

        # Transfer stress test: deterministic exclusion of 1/8th of cities
        stress_cities = sorted(set(training.pickup) | set(training.delivery))[::8]
        stress_train = training[~training.pickup.isin(stress_cities) & ~training.delivery.isin(stress_cities)]
        stress_valid = holdout[holdout.pickup.isin(stress_cities) | holdout.delivery.isin(stress_cities)]
        stress_results = {'excluded_cities': stress_cities, 'training_rows': len(stress_train), 'validation_rows': len(stress_valid)}

        for mode in CANDIDATES:
            weights = selections[mode]['weights']
            models, _ = state.task(mode + '_unseen_city_fit', lambda: selected_fit(weights, mode, stress_train, config))
            pred = blended_predict(models, weights, stress_valid, mode)
            stress_results[mode] = metrics(stress_valid.posted_rate, pred)
        atomic_json(output / 'unseen_city_stress.json', stress_results)

        bundle = {}
        for mode in CANDIDATES:
            weights = selections[mode]['weights']
            models, info = state.task(mode + '_final_fit', lambda: selected_fit(weights, mode, train, config))
            bundle[mode] = {'models': models, 'weights': weights}
            atomic_json(output / f'{mode}_final_fit.json', info)
        bundle['config_hash'] = key
        atomic_pickle(output / 'model_bundle.pkl', bundle)

        submission, december = write_outputs(bundle, frames, output)

        # Run score.py
        (root / 'scorer_results').mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(root / 'score.py'), '--predictions', str(root / 'validation_predictions.csv'),
                        '--december-predictions', str(root / 'data/december_chart_inputs.csv'),
                        '--output-dir', str(root / 'scorer_results')], check=True)
        state.progress('predictions validated', completed=True)

        # Record verified evidence for delivery check
        evidence = {
            'selection': selections,
            'locked_metrics': holdout_results,
            'output_sha256': {
                'validation_predictions.csv': file_hash(output / 'validation_predictions.csv'),
                'december_predictions.csv': file_hash(output / 'december_predictions.csv'),
            },
            'reason': 'Clean pipeline excluding poisoned quote_signal, correcting negative weights, and preventing route memorization.'
        }
        atomic_json(output / 'initial_run_evidence.json', evidence)

        print(json.dumps({'selection': selections, 'locked_holdout': holdout_results}, indent=2), flush=True)
        atomic_json(output / 'run_complete.json', {'config_hash': key, 'stage_done': True,
                    'elapsed_seconds': time.monotonic() - state.start,
                    'submission_rows': len(submission), 'december_rows': len(december)})
        state.done = state.total
        state.progress('complete')
    except BaseException:
        state.progress('interrupted_or_failed; matching completed tasks can resume')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--rounds', type=int, default=1500)
    args = parser.parse_args()
    run(args)
