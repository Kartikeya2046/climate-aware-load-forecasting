"""Model eligibility and the Model Fitness Score.

The Model Fitness Score is a weighted rubric on a 0-100 scale. It ranks candidate models for a
request; it is NOT a probability that a prediction is correct. Weights come from config.toml and
are hypotheses evaluated by core/weights_study.py.

Component definitions (each 0-100; None = unavailable, its weight is redistributed):
  validation_performance   100 · 2^(skill - 1) with skill = ½·skill_MAE + ½·skill_RMSE and
                           skill_X = 1 - X / X of the naive benchmark on the same holdout hours (one-step:
                           same hour previous day; forecast: same hour previous week). 100 = perfect,
                           50 = no better than the naive benchmark, each unit of skill lost halves the
                           score (monotone, so models worse than naive stay distinguishable). Evidence must
                           exist for the requested location and inference mode, else unavailable.
  climate_similarity       project climate-similarity score of the input weather against the model's
                           training weather (see climate.py); point proximity for single observations.
  feature_compatibility    100 · mean over required inputs of 1.00 supplied / 0.95 unit-converted /
                           0.90 derived from solar geometry. A missing mandatory input = ineligible.
  input_data_quality       100 · (1 - min(1, missing + ½·forward-filled + implausible + duplicate + gap
                           fractions)) over the model's input columns.
  temporal_compatibility   100 · history factor · horizon factor · date-domain factor · coverage, with
                           history = ½ + ½·observed-history completeness, horizon = min(1, 168 h / H)
                           for free-running forecasts, domain = max(0.5, 1 - 0.1 · years outside the
                           evaluated period), coverage = predicted / requested hours.
  calibration_reliability  ½ · 100·max(0, 1 - 4·|coverage90 - 0.90|)  (interval fitted on the first half
                           of the holdout, tested on the second)  +  ½ · 100·min(MAE_H1, MAE_H2)/max(…).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import ComponentScore, ModelScore

COMPONENTS = ["validation_performance", "climate_similarity", "feature_compatibility",
              "input_data_quality", "temporal_compatibility", "calibration_reliability"]
LABELS = {"validation_performance": "Validation performance", "climate_similarity": "Climate similarity",
          "feature_compatibility": "Feature compatibility", "input_data_quality": "Input data quality",
          "temporal_compatibility": "Temporal & horizon compatibility", "calibration_reliability": "Calibration & reliability"}
EVALUATED_YEARS = {"Two-branch TCN": (2016, 2023), "LSTM": (2006, 2015)}


def evidence_key(mode: str, history: str) -> str:
    if mode == "one_step":
        return "one_step"
    return "forecast_no_history" if history == "baseline_assumption" else "forecast"


def skill_to_score(skill: float) -> float:
    return float(100 * 2 ** (min(skill, 1.0) - 1))


def validation_score(ev: dict | None) -> tuple[float | None, str]:
    if not ev or "skill_mae" not in ev:
        return None, "no held-out evidence for this location and inference mode"
    skill = 0.5 * ev["skill_mae"] + 0.5 * ev["skill_rmse"]
    s = skill_to_score(skill)
    return s, (f"MAPE {ev['MAPE_percent']:.2f} %, MAE {ev['MAE_MW']:.1f} MW, R² {ev['R2']:.3f} on {ev['n']:,} held-out hours; "
               f"naive benchmark MAPE {ev['naive']['MAPE_percent']:.2f} % → skill {skill:+.3f}")


def calibration_score(ev: dict | None) -> tuple[float | None, str]:
    cal = (ev or {}).get("calibration") or {}
    h = (ev or {}).get("halves") or {}
    if cal.get("coverage") is None or "MAE_MW" not in h.get("H1", {}) or "MAE_MW" not in h.get("H2", {}):
        return None, "no residual evidence"
    cov = cal["coverage"]
    c = 100 * max(0.0, 1 - 4 * abs(cov - cal["nominal"]))
    m1, m2 = h["H1"]["MAE_MW"], h["H2"]["MAE_MW"]
    st = 100 * min(m1, m2) / max(m1, m2)
    return (c + st) / 2, f"90 % interval covered {cov * 100:.1f} % of second-half hours; MAE first/second half {m1:.1f}/{m2:.1f} MW"


def feature_score(required: list[str], converted: set[str], derived: set[str]) -> tuple[float, str]:
    parts = [0.90 if v in derived else 0.95 if v in converted else 1.0 for v in required]
    notes = [f"{len(required)} inputs"]
    if converted & set(required):
        notes.append(f"unit-converted: {', '.join(sorted(converted & set(required)))}")
    if derived & set(required):
        notes.append(f"derived: {', '.join(sorted(derived & set(required)))}")
    return 100 * float(np.mean(parts)), "; ".join(notes)


def quality_score(frame: pd.DataFrame, required: list[str], quality: dict) -> tuple[float, str]:
    if frame is None or frame.empty:
        return 0.0, "no rows"
    cols = quality.get("columns", {})
    n = max(quality.get("rows_hourly", len(frame)), 1)
    miss = float(frame.reindex(columns=required).isna().mean().mean())
    filled = float(np.mean([cols.get(v, {}).get("filled", 0) / n for v in required]))
    invalid = float(np.mean([cols.get(v, {}).get("invalid", 0) / n for v in required]))
    dup = quality.get("duplicate_timestamps", 0) / max(quality.get("rows_uploaded", n), 1)
    gaps = quality.get("missing_hours", 0) / n
    pen = miss + 0.5 * filled + invalid + dup + gaps
    return 100 * (1 - min(1.0, pen)), (f"missing {miss:.1%}, forward-filled {filled:.1%}, implausible {invalid:.1%}, "
                                       f"duplicate {dup:.1%}, gaps {gaps:.1%}")


def temporal_score(family: str, mode: str, horizon_h: int, history_completeness: float,
                   start: pd.Timestamp, end: pd.Timestamp, coverage: float) -> tuple[float, str]:
    hist = 0.5 + 0.5 * history_completeness
    hor = 1.0 if mode == "one_step" else min(1.0, 168 / max(horizon_h, 1))
    lo, hi = EVALUATED_YEARS[family]
    outside = max(0.0, lo - (start.year + start.dayofyear / 366)) + max(0.0, (end.year + end.dayofyear / 366) - (hi + 1))
    dom = max(0.5, 1 - 0.1 * outside)
    s = 100 * hist * hor * dom * coverage
    return s, (f"history {history_completeness:.0%} observed (×{hist:.2f}); horizon {horizon_h} h "
               f"{'one-step' if mode == 'one_step' else 'free-running'} (×{hor:.2f}); "
               f"dates vs evaluated {lo}–{hi} (×{dom:.2f}); coverage {coverage:.0%}")


def fitness(components: list[ComponentScore]) -> float | None:
    avail = [c for c in components if c.score is not None and c.weight > 0]
    total = sum(c.weight for c in avail)
    for c in components:
        c.effective_weight = (c.weight / total) if (c.score is not None and total) else 0.0
    return float(sum(c.score * c.weight for c in avail) / total) if total else None


def rank(scores: list[ModelScore], near_tie: float = 1.0) -> list[ModelScore]:
    """Deterministic: fitness desc, then validation score desc, then model id. Ineligible never rank."""
    def val(s):
        v = next((c.score for c in s.components if c.name == "validation_performance"), None)
        return -1 if v is None else v
    ranked = sorted([s for s in scores if s.eligibility != "ineligible" and s.fitness is not None],
                    key=lambda s: (-s.fitness, -val(s), s.model_id))
    for s in scores:
        s.rank, s.selected, s.near_tie = None, False, False
    for i, s in enumerate(ranked, 1):
        s.rank = i
    if ranked:
        ranked[0].selected = True
        for s in ranked[1:]:
            s.near_tie = ranked[0].fitness - s.fitness < near_tie
    return ranked + [s for s in scores if s.rank is None]
