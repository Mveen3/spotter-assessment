from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return {
        'rows': len(actual), 'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
        'mae': float(mean_absolute_error(actual, predicted)),
        'wape_percent': float(100 * np.abs(actual - predicted).sum() / np.abs(actual).sum()),
        'r2': float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        'bias': float(np.mean(predicted - actual)),
    }


def select_blend(oof):
    names = [c for c in oof if c.startswith('pred_')]
    candidates = [{n.removeprefix('pred_'): 1.} for n in names]
    for left, right in combinations(names, 2):
        for weight in [.25, .5, .75]:
            candidates.append({left.removeprefix('pred_'): weight, right.removeprefix('pred_'): 1 - weight})
    rows = []
    for weights in candidates:
        pred = sum(w * oof['pred_' + name].to_numpy() for name, w in weights.items())
        by_fold = [metrics(oof.loc[mask, 'actual'], pred[mask]) for mask in
                   [(oof.fold == fold).to_numpy() for fold in sorted(oof.fold.unique())]]
        rows.append({'weights': weights,
                     'mean_fold_rmse': float(np.mean([m['rmse'] for m in by_fold])),
                     'mean_fold_mae': float(np.mean([m['mae'] for m in by_fold])),
                     'fold_metrics': by_fold})
    rows.sort(key=lambda r: (r['mean_fold_rmse'], len(r['weights'])))
    return rows[0], rows


def slices(frame, pred, training):
    routes = set(zip(training.pickup, training.delivery))
    cities = set(training.pickup) | set(training.delivery)
    target_cut = float(training.posted_rate.quantile(.99))
    groups = {
        'equipment': frame.equipment.astype(str),
        'month': frame.date.dt.strftime('%Y-%m'),
        'distance': pd.cut(frame.distance, [0, 250, 750, 1500, np.inf], labels=['<=250', '250-750', '750-1500', '>1500']).astype(str),
        'route_seen': pd.Series(['seen' if r in routes else 'unseen' for r in zip(frame.pickup, frame.delivery)], index=frame.index),
        'city_seen': pd.Series(['seen' if p in cities and d in cities else 'unseen' for p, d in zip(frame.pickup, frame.delivery)], index=frame.index),
        'target_tail': pd.Series(np.where(frame.posted_rate > target_cut, 'above_training_p99', 'at_or_below_training_p99'), index=frame.index),
        'quote_available': pd.Series(np.where(frame.quote_signal.notna() & (frame.quote_signal > 0), 'available', 'missing_or_invalid'), index=frame.index),
    }
    result = []
    for dimension, values in groups.items():
        for label in sorted(values.unique()):
            mask = (values == label).to_numpy()
            result.append({'dimension': dimension, 'segment': label, **metrics(frame.posted_rate.to_numpy()[mask], np.asarray(pred)[mask])})
    return result
