"""Evaluate fitness-weight sets against held-out windows (run via `python -m core.evidence --weights-study`).

Protocol (no information from the evaluated window enters its own score):
* Each holdout is split chronologically. Validation skill is computed from the FIRST-half weekly
  windows only; selection is then evaluated on every SECOND-half weekly window.
* Per window and candidate: validation (first-half skill), climate similarity of that window's
  weather against the model's training weather, temporal compatibility (date domain), feature
  compatibility and data quality (100: repository data). Calibration is left unavailable because
  it is itself estimated from the half split.
* The selected model is the fitness argmax; regret = its window MAPE minus the best candidate's.
Only locations with at least two scorable candidates are informative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import climate
from .data import config
from .evidence import holdout, training_reference
from .models import registry
from .schema import ComponentScore
from .scoring import COMPONENTS, fitness, skill_to_score, temporal_score


def _skill(rows):
    m = np.nanmean([r["MAE_MW"] for r in rows if r["MAE_MW"] is not None])
    n = np.nanmean([r["naive_MAE_MW"] for r in rows if r["naive_MAE_MW"] is not None])
    return 1 - m / n  # MAE skill only: per-window RMSE is not stored


def run(ev: dict) -> dict:
    reg = registry()
    sets = {"configured": dict(config()["fitness_weights"]),
            "validation_only": {c: float(c == "validation_performance") for c in COMPONENTS},
            "climate_only": {c: float(c == "climate_similarity") for c in COMPONENTS},
            "equal": {c: 1 / len(COMPONENTS) for c in COMPONENTS}}
    rng = np.random.default_rng(0)
    for k in range(200):
        w = rng.dirichlet(np.ones(len(COMPONENTS)))
        sets[f"random_{k:03d}"] = dict(zip(COMPONENTS, w.round(3)))
    cases = []
    for city in {c for m in ev["models"].values() for c in m}:
        cands = [mid for mid, m in ev["models"].items() if city in m]
        if len(cands) < 2:
            continue
        for mode in ("one_step", "forecast"):
            per = {mid: ev["models"][mid][city][mode]["per_window"] for mid in cands}
            nwin = min(len(v) for v in per.values())
            half = nwin // 2
            for k in range(half, nwin):
                start = pd.Timestamp(per[cands[0]][k]["start_label"])
                win = None
                comps, mape = {}, {}
                for mid in cands:
                    a = reg[mid]
                    rc, rs, re_ = training_reference(mid)
                    if win is None:
                        win = climate.reference_weather(city, str(start - pd.Timedelta(hours=a.lookback)), str(start + pd.Timedelta(hours=167)))
                    clim = climate.similarity_to_training(win, rc, str(rs), str(re_), a.weather_inputs, False)
                    sk = _skill(per[mid][:half])
                    t, _ = temporal_score(a.family, mode, 168, 1.0, start, start + pd.Timedelta(hours=167), 1.0)
                    comps[mid] = {"validation_performance": skill_to_score(sk), "climate_similarity": clim.get("overall"),
                                  "feature_compatibility": 100.0, "input_data_quality": 100.0, "temporal_compatibility": t,
                                  "calibration_reliability": None}
                    mape[mid] = per[mid][k]["MAPE_percent"]
                if any(v is None for v in mape.values()):
                    continue
                cases.append({"city": city, "mode": mode, "start_label": str(start), "components": comps, "mape": mape})
    results = {}
    for name, w in sets.items():
        regret, hits = [], 0
        for c in cases:
            fit = {}
            for mid, comp in c["components"].items():
                cs = [ComponentScore(n, comp[n], w[n]) for n in COMPONENTS]
                fit[mid] = fitness(cs) or 0
            sel = sorted(fit, key=lambda m: (-fit[m], m))[0]
            best = min(c["mape"], key=c["mape"].get)
            regret.append(c["mape"][sel] - c["mape"][best])
            hits += sel == best
        results[name] = {"weights": w, "mean_regret_mape_points": float(np.mean(regret)) if regret else None,
                         "selection_accuracy": hits / len(cases) if cases else None}
    order = sorted(results, key=lambda n: (results[n]["mean_regret_mape_points"], n))
    by_city = {}
    for c in cases:
        best = min(c["mape"], key=c["mape"].get)
        by_city.setdefault(f"{c['city']}/{c['mode']}", {}).setdefault(best, 0)
        by_city[f"{c['city']}/{c['mode']}"][best] += 1
    return {"protocol": __doc__.strip(), "n_cases": len(cases), "best_model_counts": by_city,
            "results": {n: results[n] for n in ["configured", "validation_only", "climate_only", "equal"]},
            "configured_rank_among_all_sets": order.index("configured") + 1, "n_weight_sets": len(sets),
            "best_set": {"name": order[0], **results[order[0]]}}
