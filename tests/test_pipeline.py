import unittest

import numpy as np
import pandas as pd

from freight.data import CORE, purge_overlap
from freight.evaluation import select_blend
from freight.features import Features
from freight.models import CANDIDATES, Estimator
from run import split


def fixture(n=240):
    rng = np.random.default_rng(9)
    distance = rng.uniform(70, 1800, n)
    quote = rng.uniform(1.7, 2.5, n)
    return pd.DataFrame({
        'load_id': [f'T-{i}' for i in range(n)], 'pickup': np.where(np.arange(n) % 2, 'A', 'B'),
        'delivery': 'C', 'equipment': np.where(np.arange(n) % 3, 'Dry Van', 'Reefer'),
        'distance': distance, 'weight': rng.uniform(10000, 45000, n),
        'date': pd.date_range('2025-01-01', periods=n),
        'pickup_lat': 35., 'pickup_lon': -90., 'delivery_lat': 38., 'delivery_lon': -85.,
        'market_index': 1., 'quote_signal': quote, 'posted_rate': distance * quote + 50,
    })


class Contracts(unittest.TestCase):
    def test_core_cannot_use_privileged_fields(self):
        frame = fixture()
        builder = Features('core').fit(frame.iloc[:180])
        altered = frame.iloc[180:].copy()
        altered['quote_signal'] = 1e9
        altered['market_index'] = -999
        altered['pickup_lat'] = 89
        pd.testing.assert_frame_equal(builder.transform(frame.iloc[180:][CORE]), builder.transform(altered))

    def test_invalid_weight_and_unseen_city(self):
        frame = fixture()
        builder = Features('core').fit(frame.iloc[:180])
        valid = frame.iloc[180:182].copy()
        valid.loc[valid.index[0], 'weight'] = -100
        valid.loc[valid.index[1], 'weight'] = np.nan
        valid['pickup'] = 'UNSEEN'
        x = builder.transform(valid[CORE])
        self.assertEqual(x.weight_missing.sum(), 1)
        self.assertEqual(x.weight_invalid.sum(), 1)
        self.assertTrue((x.weight > 0).all())
        self.assertTrue(x.pickup.isna().all())
        self.assertTrue(x.pickup_lat_missing.eq(1).all())

    def test_split_and_duplicate_purge(self):
        frame = fixture()
        train, valid, purged = split(frame, '2025-05-01', '2025-07-01')
        self.assertLess(train.date.max(), valid.date.min())
        self.assertEqual(purged, 0)
        duplicate = pd.concat([train, valid.iloc[:1]])
        cleaned, count = purge_overlap(duplicate, valid)
        self.assertEqual(count, 1)
        self.assertEqual(len(cleaned), len(train))

    def test_all_candidates_predict_finite(self):
        frame = fixture()
        train, valid = frame.iloc[:180], frame.iloc[180:]
        for mode, candidates in CANDIDATES.items():
            for name in candidates:
                with self.subTest(mode=mode, name=name):
                    model = Estimator(name, mode, rounds=3, threads=2, device='cpu').fit(train)
                    pred = model.predict(valid[CORE] if mode == 'core' else valid)
                    self.assertEqual(len(pred), len(valid))
                    self.assertTrue(np.isfinite(pred).all())
                    self.assertTrue((pred > 0).all())

    def test_blend_uses_mean_fold_rmse(self):
        oof = pd.DataFrame({'fold': ['a', 'a', 'b', 'b'], 'actual': [100., 100., 200., 200.],
                            'pred_low': [80., 80., 180., 180.], 'pred_high': [120., 120., 220., 220.]})
        selected, _ = select_blend(oof)
        self.assertEqual(selected['weights'], {'low': .5, 'high': .5})
        self.assertEqual(selected['mean_fold_rmse'], 0.)


if __name__ == '__main__':
    unittest.main()
