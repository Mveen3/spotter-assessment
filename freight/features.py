from __future__ import annotations

import numpy as np
import pandas as pd

from freight.data import CORE


class Features:
    """Feature engineering pipeline.

    - Cleans payload weights (recovering negative sign-error inputs as abs(weight)
      while retaining weight_invalid indicators).
    - Excludes poisoned quote_signal from production modeling.
    - Avoids high-cardinality route category memorization by using pickup, delivery,
      and continuous geographic distance and bearing.
    - In core mode, extra fields are dropped and coordinates are recovered from historical city maps.
    """

    def __init__(self, mode='enriched'):
        if mode not in {'core', 'enriched'}:
            raise ValueError(mode)
        self.mode = mode

    def fit(self, frame):
        city = pd.concat([
            frame[['pickup', 'pickup_lat', 'pickup_lon']].set_axis(['city', 'lat', 'lon'], axis=1),
            frame[['delivery', 'delivery_lat', 'delivery_lon']].set_axis(['city', 'lat', 'lon'], axis=1),
        ])
        self.city = city.groupby('city')[['lat', 'lon']].median()
        self.defaults = {}
        for col, fallback in [('distance', 500.), ('weight', 32000.), ('market_index', 1.)]:
            values = pd.to_numeric(frame[col], errors='coerce')
            values = values.abs().where(np.isfinite(values) & (values > 0))
            self.defaults[col] = float(values.median()) if values.notna().any() else fallback
        raw = self._raw(frame)
        self.categories = {c: sorted(raw[c].unique()) for c in ['pickup', 'delivery', 'equipment'] if c in raw.columns}
        return self

    def _raw(self, original):
        frame = original.reindex(columns=CORE).copy() if self.mode == 'core' else original.copy()
        out = pd.DataFrame(index=frame.index)
        for col in ['pickup', 'delivery', 'equipment']:
            out[col] = frame[col].fillna('__MISSING__').astype(str).str.strip()

        # Clean distance
        d_raw = pd.to_numeric(frame.get('distance', pd.Series(np.nan, index=frame.index)), errors='coerce')
        good_d = np.isfinite(d_raw) & (d_raw > 0)
        out['distance_missing'] = (~good_d).astype(float)
        out['distance'] = d_raw.where(good_d).fillna(self.defaults['distance']).astype(float)

        # Clean weight: negative weights match positive weight distribution; use abs(weight) + invalid flag
        w_raw = pd.to_numeric(frame.get('weight', pd.Series(np.nan, index=frame.index)), errors='coerce')
        out['weight_missing'] = w_raw.isna().astype(float)
        out['weight_invalid'] = (w_raw.notna() & (w_raw <= 0)).astype(float)
        clean_w = w_raw.abs()
        out['weight'] = clean_w.where(clean_w > 0).fillna(self.defaults['weight']).astype(float)

        # Market index for enriched mode
        if self.mode == 'enriched':
            m_raw = pd.to_numeric(frame.get('market_index', pd.Series(np.nan, index=frame.index)), errors='coerce')
            good_m = np.isfinite(m_raw) & (m_raw > 0)
            out['market_index_missing'] = (~good_m).astype(float)
            out['market_index'] = m_raw.where(good_m).fillna(self.defaults['market_index']).astype(float)
            out['market_distance'] = out['market_index'] * out['distance']

        # Coordinates
        for side in ['pickup', 'delivery']:
            for coord, limit in [('lat', 90), ('lon', 180)]:
                col = f'{side}_{coord}'
                v = pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors='coerce')
                v = v.where(np.isfinite(v) & v.between(-limit, limit))
                v = v.fillna(out[side].map(self.city[coord]))
                out[col + '_missing'] = v.isna().astype(float)
                out[col] = v.fillna(float(self.city[coord].median())).astype(float)

        d = out.distance
        out['log_distance'] = np.log(d)
        out['inverse_distance'] = 1 / d
        out['sqrt_distance'] = np.sqrt(d)
        out['weight_per_mile'] = out.weight / d
        out['lat_delta'] = out.delivery_lat - out.pickup_lat
        out['lon_delta'] = out.delivery_lon - out.pickup_lon

        # Haversine distance and road circuity ratio
        p, q = np.radians(out.pickup_lat), np.radians(out.delivery_lat)
        h = np.sin((q - p) / 2)**2 + np.cos(p) * np.cos(q) * np.sin(np.radians(out.lon_delta) / 2)**2
        out['geo_miles'] = 3958.8 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        out['road_geo_ratio'] = d / out.geo_miles.clip(lower=1)

        dates = pd.to_datetime(frame.date, errors='raise')
        out['weekday'] = dates.dt.dayofweek.astype(float)
        out['weekend'] = (dates.dt.dayofweek >= 5).astype(float)
        out['month_start'] = dates.dt.is_month_start.astype(float)
        out['month_end'] = dates.dt.is_month_end.astype(float)

        # Smooth periodic bases
        for period, phase, harmonics in [
            (365.25, dates.dt.dayofyear - 1, 2),
            (7., dates.dt.dayofweek, 2),
        ]:
            for k in range(1, harmonics + 1):
                for name, fn in [('sin', np.sin), ('cos', np.cos)]:
                    out[f'calendar_{period:g}_{k}_{name}'] = fn(2 * np.pi * k * phase / period)

        return out

    def transform(self, frame):
        out = self._raw(frame)
        for col, values in self.categories.items():
            out[col] = pd.Categorical(out[col].where(out[col].isin(values)), categories=values)
        return out

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)
