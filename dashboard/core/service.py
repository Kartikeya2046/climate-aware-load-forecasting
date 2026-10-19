"""End-to-end prediction service: inputs -> context -> eligibility -> scores -> selection ->
inference -> output checks -> intervals -> aggregation. The UI calls only `prepare_inputs`
and `run_prediction`; everything returned uses explicit units and local-standard timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import climate
from .data import CITIES, config, record_span, repository_record
from .evidence import load_evidence, training_reference
from .models import Adapter, TwoBranchTCN, registry
from .schema import ComponentScore, Issue, LogEntry, ModelScore, PredictionRequest, PredictionResponse, VARIABLES
from .scoring import (COMPONENTS, LABELS, calibration_score, evidence_key, feature_score, fitness, quality_score,
                      rank, temporal_score, validation_score)
from .validation import derive, repository_context

MAX_HISTORY = max(TwoBranchTCN.history_hours, 24)


@dataclass
class Prepared:
    frame: pd.DataFrame
    issues: list[Issue] = field(default_factory=list)
    log: list[LogEntry] = field(default_factory=list)
    history_hours_observed: int = 0
    ground_truth_hours: int = 0


def weather_lead_hours(city: str) -> int:
    """Hours of weather needed after the last predicted hour: the training data paired load(UTC t)
    with weather(local t), so a prediction for local hour L reads weather up to L + offset - 1."""
    return CITIES[city].utc_offset - 1


def prepare_inputs(req: PredictionRequest) -> Prepared:
    """Assemble the model input frame (context + window) and log every assumption."""
    city, start, end = req.location_id, req.start, req.end
    f = req.weather.copy()
    issues, log = [], []
    if req.context_source == "repository":
        ctx = repository_context(city, start, MAX_HISTORY, end)
        if ctx is None:
            lo, hi = record_span(city)
            issues.append(Issue("warning", "no_repository_context",
                                f"The recorded {CITIES[city].name} data ({lo:%Y-%m-%d} – {hi:%Y-%m-%d}) does not cover the "
                                f"{MAX_HISTORY} h before {start:%Y-%m-%d %H:%M}; no recorded history is used."))
        else:
            hist = ctx.loc[:start - pd.Timedelta(hours=1)]
            f = f.combine_first(hist) if not f.empty else hist
            truth = ctx["load_mw"].loc[start:end]
            if req.scenario:
                truth = truth.iloc[:0]  # recorded load happened under different weather: not ground truth for a scenario
            if "load_mw" not in f or f["load_mw"].loc[start:end].isna().all():
                f.loc[truth.index, "load_mw"] = truth
            log.append(LogEntry("context", "repository history",
                                f"recorded weather + load for {len(hist)} h before the start (cleaned as in training); "
                                f"recorded load in the window kept as ground truth ({int(truth.notna().sum())} h)", len(hist)))
    f = f.sort_index().asfreq("h")
    lead = weather_lead_hours(city)
    need_to = end + pd.Timedelta(hours=lead)
    weather_cols = [c for c in f.columns if c != "load_mw"]
    if f.index[-1] < need_to:
        f = f.reindex(pd.date_range(f.index[0], need_to, freq="h"))
    tail = f.loc[end + pd.Timedelta(hours=1):need_to, weather_cols]
    if lead and tail.isna().any().any():
        n = int(tail.isna().any(axis=1).sum())
        f.loc[end + pd.Timedelta(hours=1):need_to, weather_cols] = f.loc[:need_to, weather_cols].ffill().loc[end + pd.Timedelta(hours=1):]
        issues.append(Issue("recoverable", "weather_lead",
                            f"The trained models read weather up to {lead} h after each predicted hour (timestamp alignment of "
                            f"the training data). {n} hour(s) after {end:%Y-%m-%d %H:%M} were not supplied and hold the last observation."))
        log.append(LogEntry("context", "weather lead: hold last observation", f"{n} h after the window end", n))
    f.index.name = "timestamp"
    load = f["load_mw"] if "load_mw" in f else pd.Series(np.nan, index=f.index)
    hist = load.loc[start - pd.Timedelta(hours=MAX_HISTORY):start - pd.Timedelta(hours=1)]
    return Prepared(f, issues, log, int(hist.notna().sum()), int(load.loc[start:end].notna().sum()))


def history_completeness(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, hours: int, mode: str) -> float:
    load = frame["load_mw"] if "load_mw" in frame else pd.Series(np.nan, index=frame.index)
    lo = start - pd.Timedelta(hours=hours)
    hi = end - pd.Timedelta(hours=1) if mode == "one_step" else start - pd.Timedelta(hours=1)
    return float(load.reindex(pd.date_range(lo, hi, freq="h")).notna().mean())


def eligibility(a: Adapter, req: PredictionRequest, frame: pd.DataFrame) -> tuple[str, list[str], str]:
    """-> (status, reasons, history policy)."""
    reasons: list[str] = []
    if not a.ensure_loaded():
        return "ineligible", [f"artifact check failed: {a.load_error}"], "observed"
    ok, why = a.supports_location(req.location_id)
    if not ok:
        return "ineligible", [why], "observed"
    status = "eligible"
    if req.location_id not in a.trained_on + a.evaluated_on:
        status = "conditional"
        reasons.append(f"cross-domain: {why}")
    missing = [VARIABLES[v].label for v in a.weather_inputs if v not in frame or frame[v].loc[req.start:req.end].isna().all()]
    if missing:
        return "ineligible", reasons + [f"missing required input(s): {', '.join(missing)}"], "observed"
    off = CITIES[req.location_id].utc_offset  # first window reads local weather [start + off - lookback, start + off - 1]
    first = pd.date_range(req.start + pd.Timedelta(hours=off - a.lookback), req.start + pd.Timedelta(hours=off - 1), freq="h")
    ok_rows = frame.reindex(index=first, columns=a.weather_inputs).notna().all(axis=1)
    if not ok_rows.all():
        n = int((~ok_rows).sum())
        return "ineligible", reasons + [f"the model window needs weather for the {a.lookback} h before the start; {n} of those hours are missing "
                                        "(include them in the CSV or use recorded context)"], "observed"
    comp = history_completeness(frame, req.start, req.end, a.history_hours, req.prediction_mode)
    policy = "observed"
    if req.prediction_mode == "one_step":
        if comp == 0:
            return "ineligible", reasons + ["one-hour-ahead mode needs observed load for the history and every predicted hour; none supplied"], policy
        if comp < 1:
            reasons.append(f"observed load covers {comp:.0%} of the required hours; hours without a complete window are skipped")
    else:
        if comp < 1:
            if isinstance(a, TwoBranchTCN):
                policy = "baseline_assumption"
                status = "conditional"
                reasons.append(f"observed load covers {comp:.0%} of the {a.history_hours} h history; unknown residual history is set to 0 "
                               "(load assumed equal to the calendar baseline)")
            else:
                return "ineligible", reasons + [f"needs {a.history_hours} h of observed load before the start ({comp:.0%} supplied)"], policy
        horizon = int((req.end - req.start) / pd.Timedelta(hours=1)) + 1
        if horizon > 168:
            reasons.append(f"horizon {horizon} h exceeds the 168 h free-running windows used for evaluation")
    return status, reasons, policy


def score_model(a: Adapter, req: PredictionRequest, frame: pd.DataFrame, status: str, reasons: list[str],
                policy: str, ev: dict, coverage: float, converted: set[str], derived: set[str]) -> ModelScore:
    w = config()["fitness_weights"]
    city = req.location_id
    e = ev.get("models", {}).get(a.id, {}).get(city, {}).get(evidence_key(req.prediction_mode, policy))
    comps = {}
    comps["validation_performance"] = validation_score(e)
    ref_city, rs, re_ = training_reference(a.id)
    window = frame.loc[req.start - pd.Timedelta(hours=a.lookback):req.end]
    clim = climate.similarity_to_training(window, ref_city, str(rs), str(re_), a.weather_inputs, req.scenario)
    comps["climate_similarity"] = (clim.get("overall"), f"{clim.get('method')}: {clim.get('reference', clim.get('note', ''))}")
    comps["feature_compatibility"] = feature_score(a.weather_inputs, converted, derived)
    comps["input_data_quality"] = quality_score(window, a.weather_inputs, req.quality) if req.input_mode == "csv" else (100.0, "manual entry: values range-checked")
    horizon = int((req.end - req.start) / pd.Timedelta(hours=1)) + 1
    comp = history_completeness(frame, req.start, req.end, a.history_hours, req.prediction_mode)
    comps["temporal_compatibility"] = temporal_score(a.family, req.prediction_mode, horizon, comp, req.start, req.end, coverage)
    comps["calibration_reliability"] = calibration_score(e)
    cs = []
    for name in COMPONENTS:
        s, detail = comps[name]
        conf = "unavailable" if s is None else ("low" if name == "climate_similarity" and clim.get("confidence") == "low" else "normal")
        cs.append(ComponentScore(name, None if s is None else round(float(s), 2), w[name], detail=detail, confidence=conf))
    fit = fitness(cs)
    metrics = {k: e[k] for k in ("MAE_MW", "RMSE_MW", "MAPE_percent", "R2", "n") if e and k in e}
    if e:
        metrics["naive_MAPE_percent"] = e["naive"]["MAPE_percent"]
        metrics["evidence_mode"] = evidence_key(req.prediction_mode, policy)
        metrics["holdout"] = ev["models"][a.id][city]["holdout"]["description"]
    return ModelScore(a.id, a.display_name, a.family, a.version, status, reasons, cs,
                      None if fit is None else round(fit, 2), metrics=metrics,
                      training_domain={"trained_on": a.trained_on, "evaluated_on": a.evaluated_on,
                                       "reference_weather": clim.get("reference"), "history_policy": policy, "climate": clim})


def intervals(df: pd.DataFrame, e: dict | None, mode: str) -> tuple[pd.DataFrame, str | None]:
    """Empirical 90 % interval from held-out log errors log(actual/pred) for the same model, city and mode."""
    df = df.assign(lower_mw=np.nan, upper_mw=np.nan)
    if not e or not e.get("log_error_quantiles"):
        return df, None
    if mode == "one_step":
        q = e["log_error_quantiles"]
        df["lower_mw"], df["upper_mw"] = df.pred_mw * np.exp(q["0.05"]), df.pred_mw * np.exp(q["0.95"])
    else:
        h = np.arange(len(df))
        for b, (lo, hi) in {"1-24h": (0, 24), "25-72h": (24, 72), "73-168h": (72, 10 ** 9)}.items():
            q = (e.get("by_horizon", {}).get(b) or {}).get("log_error_quantiles")
            if q:
                m = (h >= lo) & (h < hi)
                df.loc[m, "lower_mw"] = df.pred_mw[m] * np.exp(q["0.05"])
                df.loc[m, "upper_mw"] = df.pred_mw[m] * np.exp(q["0.95"])
    return df, "empirical 5–95 % quantiles of held-out log errors (same model, location and mode)"


def aggregate(hourly: pd.DataFrame, freq: str) -> pd.DataFrame:
    """hourly -> daily: energy summed (MWh), power averaged (MW), peak = max hourly MW."""
    if freq == "hourly":
        return hourly
    g = hourly.resample("D")
    out = pd.DataFrame({"energy_mwh": g.energy_mwh.sum(min_count=1), "avg_mw": g.pred_mw.mean(), "peak_mw": g.pred_mw.max(),
                        "hours": g.pred_mw.count()})
    if "actual_mw" in hourly and hourly.actual_mw.notna().any():
        out["actual_energy_mwh"] = g.actual_mw.sum(min_count=1)
    return out


def summarize(df: pd.DataFrame, e: dict | None, mode: str) -> dict:
    p = df.dropna(subset=["pred_mw"])
    if p.empty:
        return {}
    total = float(p.energy_mwh.sum())
    s = {"total_energy_mwh": total, "average_power_mw": float(p.pred_mw.mean()), "peak_power_mw": float(p.pred_mw.max()),
         "peak_timestamp": str(p.pred_mw.idxmax()), "min_power_mw": float(p.pred_mw.min()), "hours_predicted": int(len(p)),
         "hours_requested": int(len(df)), "interval_hours": 1.0,
         "energy_definition": "energy per hour (MWh) = predicted hourly average load (MW) × 1 h; total = sum over predicted hours"}
    tq = (e or {}).get("total_log_error_quantiles", {})
    key = "24h" if len(df) <= 24 else "168h" if len(df) <= 168 else None
    if key and tq.get(key) and len(p) == len(df):
        s["total_energy_interval_mwh"] = [total * float(np.exp(tq[key]["0.05"])), total * float(np.exp(tq[key]["0.95"]))]
        s["total_interval_basis"] = f"5–95 % of held-out {key} window-total log errors"
    if "actual_mw" in p and p.actual_mw.notna().all():
        a = p.actual_mw
        s["observed_total_energy_mwh"] = float(a.sum())
        s["observed_MAE_MW"] = float((a - p.pred_mw).abs().mean())
        s["observed_MAPE_percent"] = float(((a - p.pred_mw).abs() / a).mean() * 100)
    return s


def run_prediction(req: PredictionRequest, prepared: Prepared | None = None) -> PredictionResponse:
    log = list(req.processing_log)
    issues = list(req.issues)
    city = req.location_id
    if city not in CITIES:
        return PredictionResponse(req.request_id, city, "unsupported_location",
                                  "No trained model has a validated or technically supported pathway for this location. "
                                  "The two-branch models need a city baseline trained on that city's load; the LSTM needs evidence for the location.",
                                  None, None, [], None, {}, {}, ["Unsupported location: prediction refused rather than producing an unreliable estimate."], log, issues)
    if req.end < req.start:
        return PredictionResponse(req.request_id, city, "error", "End is before start.", None, None, [], None, {}, {}, [], log, issues)
    prepared = prepared or prepare_inputs(req)
    frame = prepared.frame
    log += prepared.log
    issues += prepared.issues
    ev = load_evidence()
    if not ev:
        issues.append(Issue("warning", "no_evidence", "evidence/evidence.json is missing: validation and calibration components are unavailable."))
    converted = {v for v, u in req.units.items() if v in VARIABLES and VARIABLES[v].units.get(u, (1, 0)) != (1, 0)}
    derived = {e.action.split(" ")[0] for e in log if e.step == "feature engineering"}

    scores: list[ModelScore] = []
    preds: dict[str, pd.DataFrame] = {}
    for a in registry().values():
        status, reasons, policy = eligibility(a, req, frame)
        coverage = 0.0
        if status != "ineligible":
            try:
                p = a.predict(frame, city, req.start, req.end, req.prediction_mode, policy)
                coverage = p.predicted / max(p.requested, 1)
                if p.predicted == 0:
                    status, reasons = "ineligible", reasons + ["no hour could be predicted (incomplete model window)"]
                else:
                    preds[a.id] = p.frame
                    reasons += p.notes
                    if not np.isfinite(p.frame.pred_mw.dropna()).all() or (p.frame.pred_mw.dropna() <= 0).any():
                        status, reasons = "ineligible", reasons + ["output failed the finite/positive check"]
            except Exception as exc:  # noqa: BLE001
                status, reasons = "ineligible", reasons + [f"inference failed: {type(exc).__name__}: {exc}"]
        if status == "ineligible":
            scores.append(ModelScore(a.id, a.display_name, a.family, a.version, status, reasons,
                                     [ComponentScore(n, None, config()["fitness_weights"][n], detail="not scored (ineligible)", confidence="unavailable") for n in COMPONENTS],
                                     None, training_domain={"trained_on": a.trained_on, "evaluated_on": a.evaluated_on}))
        else:
            scores.append(score_model(a, req, frame, status, reasons, policy, ev, coverage, converted, derived))
    scores = rank(scores, config()["selection"]["near_tie_points"])
    winner = next((s for s in scores if s.selected), None)
    if winner is None:
        return PredictionResponse(req.request_id, city, "no_eligible_model",
                                  "No candidate model passed the eligibility checks for this request.", None, None, scores, None, {}, {},
                                  explain(scores, None), log, issues)

    policy = winner.training_domain["history_policy"]
    e = ev.get("models", {}).get(winner.model_id, {}).get(city, {}).get(evidence_key(req.prediction_mode, policy))
    hourly, basis = intervals(preds[winner.model_id], e, req.prediction_mode)
    hourly["energy_mwh"] = hourly.pred_mw * 1.0
    weather_cols = [c for c in frame.columns if c != "load_mw"]
    hourly = hourly.join(frame[weather_cols], how="left")
    lo_rec, hi_rec = repository_record(city).load_mw.quantile([0.0, 1.0])
    out_of_range = int(((hourly.pred_mw < 0.5 * lo_rec) | (hourly.pred_mw > 1.5 * hi_rec)).sum())
    if out_of_range:
        issues.append(Issue("warning", "output_range", f"{out_of_range} predicted hour(s) lie outside 0.5×–1.5× the recorded load range "
                                                       f"({lo_rec:,.0f}–{hi_rec:,.0f} MW); values are not clipped."))
    summary = summarize(hourly, e, req.prediction_mode)
    summary["interval_basis"] = basis
    summary["prediction_mode"] = req.prediction_mode
    summary["history_policy"] = policy
    summary["scenario"] = req.scenario
    summary["benchmarks"] = ev.get("benchmarks", {}).get(city, {})
    alts = {k: v for k, v in preds.items() if k != winner.model_id} if req.compute_alternatives else {}
    return PredictionResponse(req.request_id, city, "ok", "Prediction completed.", winner.model_id, winner.fitness, scores,
                              hourly, alts, summary, explain(scores, winner), log, issues,
                              climate=winner.training_domain.get("climate", {}))


def analog_ranking(weather: pd.DataFrame) -> list[dict]:
    """Project climate-similarity score of the input weather vs each supported city's recorded weather (same months ±1)."""
    rows = []
    for c in CITIES:
        lo, hi = record_span(c)
        cl = climate.similarity_to_training(weather, c, str(lo), str(hi), climate.COMMON, False)
        rows.append({"city": c, "similarity": cl.get("overall"), "classification": cl.get("classification"),
                     "method": cl.get("method"), "confidence": cl.get("confidence")})
    return sorted(rows, key=lambda r: -1 if r["similarity"] is None else -r["similarity"])


def run_analog(req: PredictionRequest, name: str = "") -> tuple[PredictionRequest, PredictionResponse]:
    """Weather from any city: predict "this weather on the grid of the most similar supported city".
    Cities are tried in order of climate similarity; the first one with an eligible model is used, and
    the normal fitness ranking picks the model there. Differences in infrastructure, population etc.
    are not modelled: the MW scale is the analog city's."""
    name = name or "Uploaded location"
    ranking = analog_ranking(req.weather.loc[req.start - pd.Timedelta(hours=MAX_HISTORY):req.end].drop(columns="load_mw", errors="ignore"))
    has_load = "load_mw" in req.weather and req.weather["load_mw"].notna().any()
    tried, first = [], None
    for r in ranking:
        c = r["city"]
        log = list(req.processing_log) + [LogEntry("analog", f"analog city: {CITIES[c].name}",
                                                   f"climate similarity {r['similarity']:.1f}/100; derived radiation uses its coordinates")]
        issues = list(req.issues) + [Issue("warning", "analog", f"{name} is not a trained location. Predicted as if this weather occurred on "
                                                                f"{CITIES[c].name}'s grid ({CITIES[c].utility}); its load scale, infrastructure and "
                                                                "population are assumed.")]
        rc = replace(req, location_id=c, weather=derive(req.weather.copy(), c, log), context_source="provided" if has_load else "none",
                     processing_log=log, issues=issues)
        resp = run_prediction(rc)
        tried.append({"city": c, "status": resp.status, "model": resp.selected_model,
                      "reason": "" if resp.status == "ok" else " ".join(resp.selection_explanation) or resp.message})
        if resp.status == "ok":
            break
        first = first or (r, rc, resp)
    else:  # nothing eligible anywhere: report against the most similar city
        r, rc, resp = first
    resp.analog = {"name": name, "city": rc.location_id, "similarity": r["similarity"], "classification": r["classification"],
                   "ranking": ranking, "tried": tried}
    return rc, resp


def explain(scores: list[ModelScore], winner: ModelScore | None) -> list[str]:
    out = []
    if winner:
        top = sorted([c for c in winner.components if c.score is not None], key=lambda c: -c.score * c.effective_weight)[:3]
        out.append(f"Selected {winner.display_name} with a Model Fitness Score of {winner.fitness:.1f}/100 "
                   f"({winner.eligibility}). Largest contributions: " +
                   ", ".join(f"{LABELS[c.name]} {c.score:.0f} × {c.effective_weight:.0%}" for c in top) + ".")
        un = [LABELS[c.name] for c in winner.components if c.score is None]
        if un:
            out.append(f"Unavailable components (weights redistributed, not scored as perfect): {', '.join(un)}.")
    for s in scores:
        if s is winner:
            continue
        if s.eligibility == "ineligible":
            out.append(f"{s.display_name}: excluded — {'; '.join(s.reasons)}.")
        elif winner:
            gaps = sorted(((LABELS[c.name], (wc.score or 0) - (c.score or 0)) for c, wc in zip(s.components, winner.components)
                           if c.score is not None and wc.score is not None), key=lambda t: -t[1])[:2]
            lower = ", ".join(f"{n} (−{d:.0f})" for n, d in gaps if d > 0)
            out.append(f"{s.display_name}: {s.fitness:.1f}/100 (rank {s.rank}{', near tie' if s.near_tie else ''}); "
                       + (f"lower mainly on {lower}." if lower else "no component is lower; ranked by the deterministic tie-break."))
    return out
