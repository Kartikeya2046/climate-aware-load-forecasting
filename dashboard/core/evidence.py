"""Offline backtests that produce the evidence used for validation, calibration and intervals.

    python -m core.evidence                 # backtests -> evidence/evidence.json
    python -m core.evidence --weights-study # weight evaluation -> evidence/weights_study.json

Everything is computed from the repository datasets on periods the models never trained on:
* Atlanta, Birmingham, Las Vegas, Phoenix: calendar year 2023 (labels), after the 2016-2021
  training and 2022 validation years of the two-branch models; also unseen by the Dayton LSTM.
* Dayton, Pittsburgh: the Dayton LSTM's chronological test split (last 15 % of the merged
  Dayton rows, 2014-06 -> 2015-12). Pittsburgh was never used for training.

Modes: one_step (observed history every hour, the research setting), forecast (168 h
free-running windows started weekly), forecast_no_history (TCN only: pre-start residual history
unknown and set to 0, i.e. load assumed at the calendar baseline).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import CITIES, DASHBOARD_DIR, EXP, clean_like_training, repository_record, to_model_frame
from .models import TwoBranchTCN, md5, registry

EVIDENCE_PATH = DASHBOARD_DIR / "evidence" / "evidence.json"
WEIGHTS_STUDY_PATH = DASHBOARD_DIR / "evidence" / "weights_study.json"
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
BUCKETS = {"1-24h": (0, 24), "25-72h": (24, 72), "73-168h": (72, 168)}
WINDOW = 168


def dayton_split_labels() -> dict[str, pd.Timestamp]:
    """Reproduce the Dayton LSTM notebook split (inner merge, 70/15/15 by row) as label timestamps."""
    e = pd.read_csv(EXP / "03_dayton_duquesne/data/processed/FINAL_MERGED_DAY_LOAD_2006_2016.csv", parse_dates=["datetime"])
    w = pd.read_csv(EXP / "03_dayton_duquesne/data/processed/weather_DAY_2006-2016.csv", parse_dates=["datetime"])
    m = pd.merge(e.sort_values("datetime"), w.sort_values("datetime"), on="datetime", how="inner")
    n = len(m)
    tr, va = int(n * 0.70), int(n * 0.15)
    return {"train_start": m.datetime.iloc[0], "train_end": m.datetime.iloc[tr - 1],
            "test_start": m.datetime.iloc[tr + va], "test_end": m.datetime.iloc[-1]}


def holdout(city: str) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    if city in ("dayton", "pittsburgh"):
        s = dayton_split_labels()
        return s["test_start"], s["test_end"], "Dayton LSTM chronological test split (last 15 % of merged Dayton rows)"
    return pd.Timestamp("2023-01-01 00:00"), pd.Timestamp("2023-12-31 23:00"), "calendar year 2023 (after 2016-2021 training, 2022 validation)"


def training_reference(model_id: str) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    """City and label period whose weather the model's weather relationship was learned from."""
    a = registry()[model_id]
    if isinstance(a, TwoBranchTCN):
        return a.source_city, pd.Timestamp("2016-01-01 00:00"), pd.Timestamp("2021-12-31 23:00")
    s = dayton_split_labels()
    return a.source_city, s["train_start"], s["train_end"]


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    ok = np.isfinite(actual) & np.isfinite(pred) & (actual > 0)
    a, p = actual[ok], pred[ok]
    if len(a) < 2:
        return {"n": int(len(a))}
    err = a - p
    return {"n": int(len(a)), "MAE_MW": float(np.mean(np.abs(err))), "RMSE_MW": float(np.sqrt(np.mean(err ** 2))),
            "MAPE_percent": float(np.mean(np.abs(err / a)) * 100), "R2": float(1 - np.sum(err ** 2) / np.sum((a - a.mean()) ** 2)),
            "nMAE": float(np.mean(np.abs(err)) / a.mean()), "mean_actual_MW": float(a.mean())}


def quantiles(x: np.ndarray) -> dict | None:
    x = x[np.isfinite(x)]
    return {str(q): float(np.quantile(x, q)) for q in QUANTILES} if len(x) >= 20 else None


def summarize(actual, pred, naive, hours, t_index) -> dict:
    """actual/pred/naive: (n_windows, horizon) or (1, n); hours: horizon index per column."""
    a, p, nv = actual.ravel(), pred.ravel(), naive.ravel()
    both = np.isfinite(a) & np.isfinite(p) & np.isfinite(nv)
    out = metrics(a, p)
    out["naive"] = metrics(np.where(both, a, np.nan), np.where(both, nv, np.nan))
    m_both = metrics(np.where(both, a, np.nan), np.where(both, p, np.nan))
    if "MAE_MW" in m_both and out["naive"].get("MAE_MW"):
        out["skill_mae"] = 1 - m_both["MAE_MW"] / out["naive"]["MAE_MW"]
        out["skill_rmse"] = 1 - m_both["RMSE_MW"] / out["naive"]["RMSE_MW"]
    loge = np.log(a / p)
    out["log_error_quantiles"] = quantiles(loge)
    hb = np.broadcast_to(hours, actual.shape).ravel()
    out["by_horizon"] = {b: {**metrics(a[(hb >= lo) & (hb < hi)], p[(hb >= lo) & (hb < hi)]),
                             "log_error_quantiles": quantiles(loge[(hb >= lo) & (hb < hi)])}
                         for b, (lo, hi) in BUCKETS.items() if ((hb >= lo) & (hb < hi)).any()}
    # chronological halves: stability + out-of-sample interval coverage (quantiles fit on H1, tested on H2)
    tt = np.broadcast_to(t_index, actual.shape).ravel()
    mid = np.nanmedian(tt)
    h1, h2 = tt <= mid, tt > mid
    out["halves"] = {"H1": metrics(a[h1], p[h1]), "H2": metrics(a[h2], p[h2])}
    q1 = quantiles(loge[h1])
    if q1:
        e2 = loge[h2][np.isfinite(loge[h2])]
        out["calibration"] = {"nominal": 0.90, "fit_on": "H1", "tested_on": "H2", "n": int(len(e2)),
                              "coverage": float(np.mean((e2 >= q1["0.05"]) & (e2 <= q1["0.95"]))) if len(e2) else None}
    return out


def window_totals(actual, pred, length) -> dict | None:
    """log(sum actual / sum pred) over complete windows of `length` hours."""
    n = actual.shape[1] // length
    if n == 0:
        return None
    a = actual[:, :n * length].reshape(actual.shape[0], n, length)
    p = pred[:, :n * length].reshape(pred.shape[0], n, length)
    ok = np.isfinite(a).all(-1) & np.isfinite(p).all(-1)
    return quantiles(np.log(a.sum(-1)[ok] / p.sum(-1)[ok]))


def per_window(lab, idx, actual, pred, naive) -> list[dict]:
    rows = []
    for k in range(len(idx)):
        m, nv = metrics(actual[k], pred[k]), metrics(actual[k], naive[k])
        rows.append({"start_label": str(lab.index[idx[k, 0]]), "n": m.get("n", 0), "MAE_MW": m.get("MAE_MW"),
                     "MAPE_percent": m.get("MAPE_percent"), "naive_MAE_MW": nv.get("MAE_MW"), "naive_MAPE_percent": nv.get("MAPE_percent")})
    return rows


def run_one_step(adapter, lab, city, s, e) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero((lab.index >= s) & (lab.index <= e))
    if isinstance(adapter, TwoBranchTCN):
        base = adapter.baseline_log(lab.index, city)
        r = np.log(lab.load_mw.where(lab.load_mw > 0).to_numpy("float64")) - base
        pred = np.exp(base[pos] + adapter.one_step(adapter.weather_matrix(lab), r, pos))
    else:
        pred = adapter.one_step(adapter.matrix(lab), pos)
    return pos, pred


def run_forecast(adapter, lab, city, starts, no_history=False) -> np.ndarray:
    if isinstance(adapter, TwoBranchTCN):
        base = adapter.baseline_log(lab.index, city)
        r = np.log(lab.load_mw.where(lab.load_mw > 0).to_numpy("float64")) - base
        if no_history:
            r = np.zeros_like(r)
        # recursive() copies each window's pre-start history and masks everything from its start on
        res = adapter.recursive(adapter.weather_matrix(lab), r, np.asarray(starts), WINDOW)
        idx = np.asarray(starts)[:, None] + np.arange(WINDOW)[None, :]
        out = np.exp(base[idx] + res)
        return out
    X = adapter.matrix(lab)
    return adapter.recursive(X, np.asarray(starts), WINDOW)


def backtest(model_id: str, city: str) -> dict:
    adapter = registry()[model_id]
    adapter.ensure_loaded()
    lab = to_model_frame(clean_like_training(repository_record(city)), city)
    s, e, desc = holdout(city)
    load = lab.load_mw.where(lab.load_mw > 0).to_numpy("float64")
    t_num = (lab.index - lab.index[0]).total_seconds().to_numpy() / 3600
    res = {"model_id": model_id, "city": city, "holdout": {"start_label": str(s), "end_label": str(e), "description": desc}}

    pos, pred = run_one_step(adapter, lab, city, s, e)
    act = load[pos][None, :]
    naive = load[pos - 24][None, :]
    one = summarize(act, pred[None, :], naive, np.zeros(1), t_num[pos][None, :])
    one["naive_definition"] = "same hour previous day (observed)"
    blocks = len(pos) // WINDOW * WINDOW
    one["total_log_error_quantiles"] = {
        "24h": window_totals(act[:, :blocks].reshape(-1, 24), pred[None, :blocks].reshape(-1, 24), 24),
        "168h": window_totals(act[:, :blocks].reshape(-1, WINDOW), pred[None, :blocks].reshape(-1, WINDOW), WINDOW)}
    one["per_window"] = per_window(lab, pos[:blocks].reshape(-1, WINDOW), act[0, :blocks].reshape(-1, WINDOW),
                                   pred[:blocks].reshape(-1, WINDOW), naive[0, :blocks].reshape(-1, WINDOW))
    res["one_step"] = one

    starts = np.arange(pos[0], pos[-1] - WINDOW + 2, WINDOW)
    idx = starts[:, None] + np.arange(WINDOW)[None, :]
    modes = [("forecast", False)] + ([("forecast_no_history", True)] if isinstance(adapter, TwoBranchTCN) else [])
    for name, no_hist in modes:
        fc = run_forecast(adapter, lab, city, starts, no_history=no_hist)
        a2, nv2 = load[idx], load[idx - WINDOW]
        out = summarize(a2, fc, nv2, np.arange(WINDOW)[None, :], t_num[idx])
        out["naive_definition"] = "same hour previous week (observed before the window starts)"
        out["windows"] = int(len(starts))
        out["total_log_error_quantiles"] = {"24h": window_totals(a2[:, :24], fc[:, :24], 24), "168h": window_totals(a2, fc, WINDOW)}
        out["per_window"] = per_window(lab, idx, a2, fc, nv2)
        res[name] = out
    return res


def benchmarks(city: str) -> dict:
    """Reference forecasts that need no learned weather model."""
    lab = to_model_frame(clean_like_training(repository_record(city)), city)
    s, e, _ = holdout(city)
    pos = np.flatnonzero((lab.index >= s) & (lab.index <= e))
    load = lab.load_mw.where(lab.load_mw > 0).to_numpy("float64")
    out = {"naive_24h": metrics(load[pos], load[pos - 24]), "naive_168h": metrics(load[pos], load[pos - 168])}
    for a in registry().values():
        if isinstance(a, TwoBranchTCN) and city in a.baseline_cities and a.ensure_loaded():
            out["calendar_baseline_only"] = metrics(load[pos], np.exp(a.baseline_log(lab.index, city)[pos]))
            out["calendar_baseline_only"]["model"] = f"{city} baseline MLP ({a.id})"
            break
    return out


def build_evidence() -> dict:
    t0 = time.time()
    reg = registry()
    ev = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "description": __doc__.strip().splitlines()[0],
          "artifacts_md5": {}, "models": {}, "benchmarks": {}, "research_reported": {}}
    for mid, a in reg.items():
        ok = a.ensure_loaded()
        ev["artifacts_md5"][mid] = {k: md5(p) for k, p in a.artifacts.items() if p.exists()}
        if getattr(a, "research_metrics", None) and a.research_metrics.exists():
            ev["research_reported"][mid] = json.loads(a.research_metrics.read_text())
        if not ok:
            print(f"skip {mid}: {a.load_error}", file=sys.stderr)
            continue
        for city in CITIES:
            supported, _ = a.supports_location(city)
            rec = repository_record(city)
            if not supported or any(v not in rec.columns for v in a.weather_inputs):
                continue
            t = time.time()
            ev["models"].setdefault(mid, {})[city] = backtest(mid, city)
            m = ev["models"][mid][city]
            print(f"{mid:14s} {city:11s} one_step MAPE {m['one_step'].get('MAPE_percent', float('nan')):6.2f}%  "
                  f"forecast MAPE {m['forecast'].get('MAPE_percent', float('nan')):6.2f}%  ({time.time() - t:.0f}s)", flush=True)
    for city in CITIES:
        ev["benchmarks"][city] = benchmarks(city)
    ev["runtime_seconds"] = round(time.time() - t0, 1)
    return ev


def load_evidence() -> dict:
    return json.loads(EVIDENCE_PATH.read_text()) if EVIDENCE_PATH.exists() else {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-study", action="store_true")
    args = ap.parse_args()
    if args.weights_study:
        from .weights_study import run
        WEIGHTS_STUDY_PATH.write_text(json.dumps(run(load_evidence()), indent=1))
        print("wrote", WEIGHTS_STUDY_PATH)
    else:
        EVIDENCE_PATH.parent.mkdir(exist_ok=True)
        EVIDENCE_PATH.write_text(json.dumps(build_evidence(), indent=1))
        print("wrote", EVIDENCE_PATH)
