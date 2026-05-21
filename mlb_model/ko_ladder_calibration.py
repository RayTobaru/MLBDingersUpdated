
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for KO ladder calibration") from exc

try:
    import joblib
except Exception as exc:  # pragma: no cover
    raise RuntimeError("joblib is required for KO ladder calibration") from exc


THRESHOLDS = (4, 5, 6, 7, 8)


def _clip_prob(x):
    arr = pd.to_numeric(x, errors="coerce").astype(float)
    arr = np.clip(arr, 1e-6, 1 - 1e-6)
    return arr


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def _brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def _logloss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _ece(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges, right=True)
    total = len(p)
    out = 0.0
    for b in range(1, bins + 1):
        mask = idx == b
        if not mask.any():
            continue
        out += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(out)


@dataclass
class ThresholdCalibrator:
    threshold: int
    pred_col: str
    actual_col: str
    coef_: float
    intercept_: float
    rows: int
    positives: int

    def predict(self, p_raw):
        x = _logit(_clip_prob(p_raw))
        z = self.intercept_ + self.coef_ * x
        return _sigmoid(z)


def fit_threshold_calibrator(df: pd.DataFrame, threshold: int) -> tuple[ThresholdCalibrator, dict]:
    pred_col = f"P(K≥{threshold})"
    actual_col = f"actual_k_ge{threshold}"
    if pred_col not in df.columns:
        raise KeyError(f"{pred_col} not found")
    if actual_col not in df.columns:
        raise KeyError(f"{actual_col} not found")

    use = df[[pred_col, actual_col]].copy()
    use[pred_col] = _clip_prob(use[pred_col])
    use[actual_col] = pd.to_numeric(use[actual_col], errors="coerce").fillna(0).astype(int)
    use = use.dropna(subset=[pred_col, actual_col])

    rows = int(len(use))
    pos = int(use[actual_col].sum())

    # Fallback to identity-ish calibrator if there is no class variation.
    if rows == 0 or pos == 0 or pos == rows:
        cal = ThresholdCalibrator(
            threshold=threshold,
            pred_col=pred_col,
            actual_col=actual_col,
            coef_=1.0,
            intercept_=0.0,
            rows=rows,
            positives=pos,
        )
        p_raw = use[pred_col].to_numpy(dtype=float) if rows else np.array([], dtype=float)
        p_cal = p_raw.copy()
    else:
        x = _logit(use[pred_col].to_numpy(dtype=float)).reshape(-1, 1)
        y = use[actual_col].to_numpy(dtype=int)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(x, y)
        cal = ThresholdCalibrator(
            threshold=threshold,
            pred_col=pred_col,
            actual_col=actual_col,
            coef_=float(model.coef_[0][0]),
            intercept_=float(model.intercept_[0]),
            rows=rows,
            positives=pos,
        )
        p_raw = use[pred_col].to_numpy(dtype=float)
        p_cal = cal.predict(p_raw)

    metrics = {
        "threshold": threshold,
        "rows": rows,
        "positives": pos,
        "base_rate": float(use[actual_col].mean()) if rows else np.nan,
        "avg_pred_raw": float(np.mean(p_raw)) if rows else np.nan,
        "avg_pred_cal": float(np.mean(p_cal)) if rows else np.nan,
        "brier_raw": _brier(use[actual_col], p_raw) if rows else np.nan,
        "brier_cal": _brier(use[actual_col], p_cal) if rows else np.nan,
        "logloss_raw": _logloss(use[actual_col], p_raw) if rows else np.nan,
        "logloss_cal": _logloss(use[actual_col], p_cal) if rows else np.nan,
        "ece_raw": _ece(use[actual_col], p_raw) if rows else np.nan,
        "ece_cal": _ece(use[actual_col], p_cal) if rows else np.nan,
        "coef": cal.coef_,
        "intercept": cal.intercept_,
    }
    return cal, metrics


def fit_all_ko_ladder_calibrators(df: pd.DataFrame, thresholds: Iterable[int] = THRESHOLDS):
    calibrators = {}
    rows = []
    for t in thresholds:
        cal, metrics = fit_threshold_calibrator(df, int(t))
        calibrators[int(t)] = cal
        rows.append(metrics)
    summary = pd.DataFrame(rows)
    return calibrators, summary


def apply_ko_ladder_calibrators(df: pd.DataFrame, calibrators: dict[int, ThresholdCalibrator]) -> pd.DataFrame:
    out = df.copy()
    for t, cal in calibrators.items():
        pred_col = f"P(K≥{t})"
        if pred_col not in out.columns:
            continue
        out[f"P_cal(K≥{t})"] = cal.predict(out[pred_col])
    return out


def save_calibrators(calibrators: dict[int, ThresholdCalibrator], path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrators, p)


def load_calibrators(path) -> dict[int, ThresholdCalibrator]:
    return joblib.load(path)


__all__ = [
    "THRESHOLDS",
    "ThresholdCalibrator",
    "fit_threshold_calibrator",
    "fit_all_ko_ladder_calibrators",
    "apply_ko_ladder_calibrators",
    "save_calibrators",
    "load_calibrators",
]
