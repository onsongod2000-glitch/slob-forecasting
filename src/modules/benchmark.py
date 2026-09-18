from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ForecastResult:
    forecast: float
    converged: bool = True
    note: str = ""


class NaiveForecaster:
    name = "Naive"

    def forecast(self, series: np.ndarray, horizon: int) -> ForecastResult:
        if len(series) == 0:
            return ForecastResult(0.0, converged=False, note="empty series")
        return ForecastResult(float(series[-1]))


class CrostonForecaster:
    """Classic Croston (1972): two exponentially-smoothed streams (demand
    size z, inter-demand interval p), updated only at non-zero periods.
    Forecast = z / p."""

    name = "Croston"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def _fit(self, series: np.ndarray):
        nz_idx = np.nonzero(series > 0)[0]
        if len(nz_idx) == 0:
            return 0.0, 1.0, False
        z = series[nz_idx[0]]
        p = nz_idx[0] + 1 if nz_idx[0] > 0 else 1.0
        last_idx = nz_idx[0]
        for idx in nz_idx[1:]:
            interval = idx - last_idx
            z = self.alpha * series[idx] + (1 - self.alpha) * z
            p = self.alpha * interval + (1 - self.alpha) * p
            last_idx = idx
        return float(z), float(p), True

    def forecast(self, series: np.ndarray, horizon: int) -> ForecastResult:
        z, p, ok = self._fit(series)
        if not ok or p <= 0:
            return ForecastResult(0.0, converged=False, note="no non-zero demand in history")
        return ForecastResult(z / p)


class SBAForecaster(CrostonForecaster):
    """Syntetos-Boylan Approximation: Croston with a (1 - alpha/2) bias
    correction (Syntetos & Boylan, 2005)."""

    name = "SBA"

    def forecast(self, series: np.ndarray, horizon: int) -> ForecastResult:
        z, p, ok = self._fit(series)
        if not ok or p <= 0:
            return ForecastResult(0.0, converged=False, note="no non-zero demand in history")
        correction = 1 - self.alpha / 2
        return ForecastResult(correction * z / p)


class TSBForecaster:
    """Teunter-Syntetos-Babai (2011): updates a demand-occurrence
    probability p_t every period (including zeros), and a demand-size
    stream z_t only when demand occurs. Forecast = p_t * z_t."""

    name = "TSB"

    def __init__(self, alpha: float = 0.1, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta

    def forecast(self, series: np.ndarray, horizon: int) -> ForecastResult:
        if len(series) == 0 or (series > 0).sum() == 0:
            return ForecastResult(0.0, converged=False, note="no non-zero demand in history")
        nz_idx = np.nonzero(series > 0)[0]
        z = series[nz_idx[0]]
        p = 1.0 / (nz_idx[0] + 1)
        for t in range(len(series)):
            occurred = series[t] > 0
            p = self.alpha * float(occurred) + (1 - self.alpha) * p
            if occurred:
                z = self.beta * series[t] + (1 - self.beta) * z
        return ForecastResult(float(p * z))


class ARIMASKUForecaster:
    """Per-SKU ARIMA(1,1,1). Falls back to naive (last value) on
    convergence failure"""

    name = "ARIMA"

    def __init__(self, order=(1, 1, 1)):
        self.order = order

    def forecast(self, series: np.ndarray, horizon: int) -> ForecastResult:
        if len(series) < 10:
            return ForecastResult(float(series[-1]) if len(series) else 0.0,
                                   converged=False, note="series too short")
        try:
            from statsmodels.tsa.arima.model import ARIMA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(series.astype(float), order=self.order)
                fitted = model.fit()
                pred = fitted.forecast(steps=horizon)
                value = float(pred[-1])
                if not np.isfinite(value):
                    raise ValueError("non-finite forecast")
                return ForecastResult(max(value, 0.0))
        except Exception as exc: 
            return ForecastResult(float(series[-1]), converged=False, note=str(exc)[:120])


FORECASTER_REGISTRY = {
    "Naive": NaiveForecaster,
    "Croston": CrostonForecaster,
    "SBA": SBAForecaster,
    "TSB": TSBForecaster,
    "ARIMA": ARIMASKUForecaster,
}


class DemandBenchmarkRunner:
    """Runs every bencmark forecaster over a set of SKUs using a
    single-origin H-step-ahead design."""

    def __init__(self, horizon: int = 4):
        self.horizon = horizon
        self.forecasters = {name: cls() for name, cls in FORECASTER_REGISTRY.items()}

    def run(self, sku_series: Dict[str, np.ndarray], origin_week_idx: int) -> List[Dict]:
        rows = []
        target_idx = origin_week_idx + self.horizon
        for sku, series in sku_series.items():
            if target_idx >= len(series):
                continue
            history = series[:origin_week_idx]
            actual = float(series[target_idx])
            for name, forecaster in self.forecasters.items():
                result = forecaster.forecast(history, self.horizon)
                rows.append({
                    "SKU": sku,
                    "method": name,
                    "actual": actual,
                    "forecast": result.forecast,
                    "abs_error": abs(actual - result.forecast),
                    "sq_error": (actual - result.forecast) ** 2,
                    "converged": result.converged,
                    "note": result.note,
                })
        return rows