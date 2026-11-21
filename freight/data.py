from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CORE = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date']
EXTRA = ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'market_index', 'quote_signal']
TARGET = 'posted_rate'
NAMES = {
    'train': ['train-test.csv', 'train_test.csv'],
    'validation': ['validation.csv'],
    'template': ['validation-predictions-template.csv', 'validation_predictions_template.csv'],
    'december': ['december-chart-inputs.csv', 'december_chart_inputs.csv'],
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    tmp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def locate(root, kind):
    for folder in [Path(root), Path(root) / 'data']:
        for name in NAMES[kind]:
            path = folder / name
            if path.is_file():
                return path
    raise FileNotFoundError(f'Missing {kind}: expected one of {NAMES[kind]} in {root} or its data/ directory')


def read_inputs(root):
    paths = {kind: locate(root, kind) for kind in NAMES}
    frames = {kind: pd.read_csv(path) for kind, path in paths.items()}
    for kind in ['train', 'validation']:
        frame = frames[kind]
        required = ['load_id'] + CORE + EXTRA + ([TARGET] if kind == 'train' else [])
        if set(frame) != set(required):
            raise ValueError(f'{kind} unexpected schema: {list(frame)}')
        if frame.load_id.isna().any() or frame.load_id.duplicated().any():
            raise ValueError(f'{kind} has missing/duplicate load IDs')
        frame['date'] = pd.to_datetime(frame.date, errors='raise')
    train, validation, template = (frames[k] for k in ['train', 'validation', 'template'])
    expected = {f'TE-{i:06d}' for i in range(1, 12001)}
    if list(template) != ['load_id', 'predicted_rate'] or len(template) != 12000:
        raise ValueError('Invalid prediction template schema or length')
    if template.load_id.duplicated().any() or set(template.load_id) != expected:
        raise ValueError('Prediction template IDs do not match the contract')
    if len(validation) != 12000 or set(validation.load_id) != expected:
        raise ValueError('Validation IDs do not match the template')
    if set(train.load_id) & set(validation.load_id):
        raise ValueError('Training and submission IDs overlap')
    if train.date.max() >= validation.date.min():
        raise ValueError('Expected strictly future submission dates')
    if train.date.min() != pd.Timestamp('2025-01-01') or train.date.max() != pd.Timestamp('2025-10-31'):
        raise ValueError('Unexpected development date coverage; revise the frozen split design')
    december = frames['december']
    # Validate the original chart inputs without inventing a model output.
    from score import validate_december
    validate_december(december.assign(predicted_rate=1.0))
    return frames, paths


def feature_hash(frame):
    fields = CORE + EXTRA
    return pd.util.hash_pandas_object(frame.reindex(columns=fields), index=False)


def purge_overlap(train, valid):
    overlap = feature_hash(train).isin(set(feature_hash(valid)))
    return train.loc[~overlap].copy(), int(overlap.sum())


def audit(frames, paths):
    result = {'files': {k: {'name': p.name, 'sha256': file_hash(p)} for k, p in paths.items()}}
    for kind in ['train', 'validation']:
        frame = frames[kind]
        invalid = {}
        for col in ['distance', 'weight', 'market_index', 'quote_signal']:
            values = pd.to_numeric(frame[col], errors='coerce')
            invalid[col] = int((values.notna() & ((values <= 0) | ~np.isfinite(values))).sum())
        numeric = frame.select_dtypes(include='number')
        result[kind] = {
            'rows': len(frame), 'date_min': str(frame.date.min().date()),
            'date_max': str(frame.date.max().date()), 'days': int(frame.date.nunique()),
            'missing': {k: int(v) for k, v in frame.isna().sum().items()},
            'invalid_nonpositive_or_infinite': invalid,
            'unique_cities': len(set(frame.pickup) | set(frame.delivery)),
            'unique_routes': len(frame[['pickup', 'delivery']].drop_duplicates()),
            'equipment': {str(k): int(v) for k, v in frame.equipment.value_counts().items()},
            'monthly_rows': {str(k): int(v) for k, v in frame.groupby(frame.date.dt.to_period('M')).size().items()},
            'feature_duplicates': int(feature_hash(frame).duplicated().sum()),
            'numeric_quantiles': json.loads(numeric.quantile([0, .01, .25, .5, .75, .99, 1]).to_json()),
        }
    train, test = frames['train'], frames['validation']
    target = pd.to_numeric(train[TARGET], errors='coerce')
    result['invalid_targets'] = int((~np.isfinite(target) | (target <= 0)).sum())
    train_cities = set(train.pickup) | set(train.delivery)
    train_routes = set(zip(train.pickup, train.delivery))
    result['submission_shift'] = {
        'unseen_cities': sorted((set(test.pickup) | set(test.delivery)) - train_cities),
        'unseen_route_rows': sum(route not in train_routes for route in zip(test.pickup, test.delivery)),
        'feature_overlap_rows': int(feature_hash(test).isin(set(feature_hash(train))).sum()),
    }
    city = pd.concat([
        train[['pickup', 'pickup_lat', 'pickup_lon']].set_axis(['city', 'lat', 'lon'], axis=1),
        train[['delivery', 'delivery_lat', 'delivery_lon']].set_axis(['city', 'lat', 'lon'], axis=1),
    ])
    inconsistencies = city.groupby('city')[['lat', 'lon']].nunique().max(axis=1)
    result['cities_with_multiple_coordinates'] = inconsistencies[inconsistencies > 1].index.tolist()

    # Target statistics
    result['target_stats'] = {
        'mean': float(target.mean()),
        'median': float(target.median()),
        'min': float(target.min()),
        'max': float(target.max()),
        'p99': float(target.quantile(0.99)),
    }

    # Negative weights audit: confirm absolute values match positive weight distribution
    train_weights = pd.to_numeric(train['weight'], errors='coerce')
    neg_weights = train_weights[train_weights < 0].abs()
    pos_weights = train_weights[train_weights > 0]
    result['negative_weights_analysis'] = {
        'train_negative_count': int((train_weights < 0).sum()),
        'train_abs_neg_median': float(neg_weights.median()) if len(neg_weights) else None,
        'train_pos_median': float(pos_weights.median()),
        'validation_negative_count': int((pd.to_numeric(test['weight'], errors='coerce') < 0).sum()),
    }

    # Quote signal trap analysis: show monthly correlation with rate per mile
    train_rpm = target / train['distance']
    monthly_corr = {}
    for month_period, df in train.groupby(train.date.dt.to_period('M')):
        m_rpm = df[TARGET] / df['distance']
        monthly_corr[str(month_period)] = float(m_rpm.corr(df['quote_signal']))
    val_qs = test['quote_signal']
    result['quote_signal_trap_audit'] = {
        'monthly_train_rpm_correlation': monthly_corr,
        'validation_mean': float(val_qs.mean()),
        'validation_std': float(val_qs.std()),
        'validation_distance_correlation': float(val_qs.corr(test['distance'])),
        'is_noise_in_validation': bool(abs(val_qs.corr(test['distance'])) < 0.05),
    }

    good = np.isfinite(target) & (target > 0) & (train.distance > 0) & (train.quote_signal > 0)
    error = target[good] - train.loc[good, 'distance'] * train.loc[good, 'quote_signal']
    result['quote_diagnostic_not_backtest'] = {
        'mae': float(error.abs().mean()), 'rmse': float(np.sqrt(np.mean(error**2))),
        'median_absolute_error': float(error.abs().median()),
        'rows_over_50_percent_relative_error': int((error.abs() / target[good] > .5).sum()),
    }
    return result


def valid_labeled(frame):
    y = pd.to_numeric(frame[TARGET], errors='coerce')
    return frame.loc[np.isfinite(y) & (y > 0)].sort_values(['date', 'load_id']).reset_index(drop=True)
