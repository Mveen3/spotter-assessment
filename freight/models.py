from __future__ import annotations

import gc
import inspect
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from freight.data import TARGET, purge_overlap
from freight.features import Features

CANDIDATES = {
    'enriched': ['equipment', 'ridge_rate', 'lgb_l2', 'lgb_l1', 'lgb_rate', 'xgb_l2'],
    'core': ['equipment', 'ridge_rate', 'lgb_l2', 'lgb_l1', 'lgb_rate', 'xgb_l2'],
}


def positive(values):
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Model emitted a nonfinite prediction')
    return np.maximum(result, 1.0)


class Estimator:
    def __init__(self, name, mode, rounds=1500, device='cpu', threads=6, seed=20261117):
        self.name, self.mode, self.rounds = name, mode, int(rounds)
        self.device, self.threads, self.seed = device, threads, seed

    def fit(self, frame, early_valid=None):
        self.features = Features(self.mode)
        x = self.features.fit_transform(frame)
        y = frame[TARGET].to_numpy(float)
        self.best_rounds = self.rounds
        self.actual_device = 'cpu'

        if self.name == 'equipment':
            rpm = pd.Series(y / x.distance.to_numpy(), index=frame.index)
            self.equipment_rpm = rpm.groupby(frame.equipment).median().to_dict()
            self.default_rpm = float(rpm.median())
            return self

        if self.name == 'ridge_rate':
            cats = [c for c in ['pickup', 'delivery', 'equipment'] if c in x.columns]
            numeric = [c for c in x if c not in cats]
            self.columns = cats + numeric
            self.model = make_pipeline(
                ColumnTransformer([
                    ('cats', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats),
                    ('nums', StandardScaler(), numeric),
                ]),
                Ridge(alpha=500.),
            )
            weights = (x.distance.to_numpy() / 1000.)**2
            with threadpool_limits(limits=self.threads):
                self.model.fit(x[self.columns], y / x.distance.to_numpy(), ridge__sample_weight=weights)
            return self

        weights = None
        if self.name == 'lgb_rate':
            y = y / x.distance.to_numpy()
            weights = (x.distance.to_numpy() / 1000.)**2

        eval_set = None
        eval_weight = None
        if early_valid is not None:
            vx = self.features.transform(early_valid)
            vy = early_valid[TARGET].to_numpy(float)
            if self.name == 'lgb_rate':
                vy = vy / vx.distance.to_numpy()
                eval_weight = [(vx.distance.to_numpy() / 1000.)**2]
            eval_set = [(vx, vy)]

        if self.name.startswith('lgb'):
            self.model = lgb.LGBMRegressor(
                objective='regression_l1' if self.name == 'lgb_l1' else 'regression',
                n_estimators=self.rounds, learning_rate=.03, num_leaves=31,
                min_child_samples=50, reg_lambda=15., reg_alpha=1.,
                colsample_bytree=.9, subsample=.85, subsample_freq=1,
                max_bin=127, n_jobs=self.threads, verbosity=-1, random_state=self.seed,
                deterministic=True, force_col_wise=True,
            )
            kwargs = {'sample_weight': weights}
            if eval_set:
                kwargs.update(eval_metric='rmse', eval_sample_weight=eval_weight,
                              callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False)])
                if 'eval_X' in inspect.signature(self.model.fit).parameters:
                    kwargs.update(eval_X=eval_set[0][0], eval_y=eval_set[0][1])
                else:
                    kwargs.update(eval_set=eval_set)
                self.model.set_params(metric='rmse')
            self.model.fit(x, y, **kwargs)
            self.best_rounds = int(self.model.best_iteration_ or self.rounds)

        elif self.name == 'xgb_l2':
            from xgboost import XGBRegressor
            attempts = [(self.device, 128), (self.device, 64), ('cpu', 64)] if self.device == 'cuda' else [('cpu', 128)]
            for device, bins in attempts:
                try:
                    self.model = XGBRegressor(
                        n_estimators=self.rounds, learning_rate=.035, max_depth=6,
                        min_child_weight=50, reg_lambda=20., reg_alpha=1.,
                        subsample=.85, colsample_bytree=.9, tree_method='hist',
                        device=device, enable_categorical=True, max_cat_to_onehot=8,
                        max_bin=bins, n_jobs=self.threads, random_state=self.seed,
                        objective='reg:squarederror', eval_metric='rmse',
                        early_stopping_rounds=60 if eval_set else None,
                    )
                    self.model.fit(x, y, eval_set=eval_set, verbose=False)
                    backend = json.loads(self.model.get_booster().save_config())
                    self.actual_device = backend['learner']['generic_param']['device']
                    self.best_rounds = int(self.model.best_iteration + 1) if eval_set else self.rounds
                    break
                except Exception as error:
                    if not any(s in str(error).lower() for s in ['memory', 'cuda', 'gpu']):
                        raise
                    if device == 'cpu':
                        raise
                    print(f'GPU retry: {type(error).__name__}; retrying with CPU', flush=True)
                    gc.collect()
        else:
            raise ValueError(self.name)
        return self

    def predict(self, frame):
        x = self.features.transform(frame)
        if self.name == 'equipment':
            return positive(frame.equipment.map(self.equipment_rpm).fillna(self.default_rpm).to_numpy(float) * x.distance.to_numpy(float))
        pred = self.model.predict(x[self.columns] if hasattr(self, 'columns') else x)
        if self.name in ['ridge_rate', 'lgb_rate']:
            pred = pred * x.distance.to_numpy()
        return positive(pred)


def train_candidate(name, mode, frame, config):
    start = time.monotonic()
    options = {k: config[k] for k in ['rounds', 'device', 'threads', 'seed']}
    selected_rounds = options['rounds']
    inner_info = None
    if name.startswith(('lgb', 'xgb')):
        inner_start = frame.date.max().to_period('M').start_time
        inner_train, inner_valid = frame[frame.date < inner_start], frame[frame.date >= inner_start]
        inner_train, purged = purge_overlap(inner_train, inner_valid)
        if len(inner_train) >= 500 and len(inner_valid) >= 100:
            probe = Estimator(name, mode, **options).fit(inner_train, inner_valid)
            selected_rounds = probe.best_rounds
            inner_info = {'start': str(inner_start.date()), 'train_rows': len(inner_train),
                          'valid_rows': len(inner_valid), 'purged': purged}
            del probe
            gc.collect()
    options['rounds'] = selected_rounds
    model = Estimator(name, mode, **options).fit(frame)
    return model, {'rounds': selected_rounds, 'inner_split': inner_info,
                   'elapsed_seconds': time.monotonic() - start, 'device': model.actual_device}


def batch_predict(model, frame, batch_size=4096):
    result, offset = [], 0
    while offset < len(frame):
        try:
            result.append(model.predict(frame.iloc[offset:offset + batch_size]))
            offset += batch_size
        except (MemoryError, RuntimeError) as error:
            if not isinstance(error, MemoryError) and 'memory' not in str(error).lower():
                raise
            if batch_size <= 128:
                raise
            batch_size //= 2
            gc.collect()
    return np.concatenate(result) if result else np.array([], dtype=float)
