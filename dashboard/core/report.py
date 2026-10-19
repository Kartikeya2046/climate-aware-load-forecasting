"""Exports: hourly predictions CSV, JSON response and a self-contained HTML report."""
from __future__ import annotations

import html
import json

import pandas as pd

from .data import CITIES
from .models import registry
from .schema import PredictionRequest, PredictionResponse
from .scoring import LABELS

LIMITATIONS = [
    "All registered models are one-hour-ahead models that read observed load history. Free-running multi-hour forecasts "
    "compound errors; their held-out accuracy is reported separately and is much lower than the one-hour-ahead setting.",
    "The research datasets pair load labelled in UTC with weather labelled in local standard time. The dashboard reproduces "
    "this alignment exactly because the models were trained on it; consequently a prediction reads weather up to "
    "(UTC offset − 1) hours after the predicted hour.",
    "The Model Fitness Score is a documented weighted rubric for ranking models; it is not a probability of correctness.",
    "Prediction intervals are empirical quantiles of held-out errors for the same model, location and mode; they assume "
    "future errors behave like the 2023 (or Dayton test-split) errors.",
    "The weather-induced adjustment of the two-branch models is the TCN residual term, which also contains autoregressive "
    "residual-history effects (it is not a pure weather effect).",
]


def predictions_frame(resp: PredictionResponse) -> pd.DataFrame:
    df = resp.predictions.copy()
    cols = ["pred_mw", "energy_mwh", "lower_mw", "upper_mw", "baseline_mw", "adjustment_mw", "actual_mw"]
    weather = [c for c in df.columns if c not in cols + ["residual_log"]]
    out = df[cols + weather].rename(columns={"pred_mw": "predicted_avg_load_mw", "energy_mwh": "predicted_energy_mwh",
                                             "lower_mw": "interval90_lower_mw", "upper_mw": "interval90_upper_mw",
                                             "adjustment_mw": "weather_adjustment_mw", "actual_mw": "observed_load_mw"})
    out.index.name = "timestamp_local_standard_time"
    return out


def predictions_csv(resp: PredictionResponse) -> bytes:
    return predictions_frame(resp).to_csv(float_format="%.4f").encode("utf-8")


def response_json(resp: PredictionResponse) -> bytes:
    return json.dumps(resp.to_dict(), indent=1, default=str).encode("utf-8")


def _table(rows: list[list], head: list[str]) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def report_html(req: PredictionRequest, resp: PredictionResponse) -> bytes:
    city = CITIES.get(req.location_id)
    s = resp.summary
    win = next((x for x in resp.scores if x.selected), None)
    info = registry()[win.model_id].info() if win else {}
    f = lambda v, d=1: "—" if v is None else f"{v:,.{d}f}"
    parts = [f"<h1>Electricity consumption prediction report</h1>",
             f"<p class=meta>Request {html.escape(resp.request_id)} · {html.escape(city.name if city else req.location_id)} · "
             f"{req.start:%Y-%m-%d %H:%M} → {req.end:%Y-%m-%d %H:%M} local standard time · input: {html.escape(req.input_mode)}"
             f"{' (scenario simulation)' if req.scenario else ''} · mode: {html.escape(req.prediction_mode)}</p>",
             f"<h2>Result</h2><p>Status: <b>{html.escape(resp.status)}</b> — {html.escape(resp.message)}</p>"]
    if resp.analog:
        a = resp.analog
        parts.append(f"<h2>Analog city</h2><p>{html.escape(a['name'])} is not a trained location. Its weather was most similar to "
                     f"<b>{html.escape(CITIES[a['city']].name)}</b> ({a['similarity']:.1f}/100) among cities with an eligible model; the prediction "
                     "is this weather on that city's grid, so its load scale, infrastructure and population are assumed.</p>")
        parts.append(_table([[CITIES[x["city"]].name, f(x["similarity"]), x["classification"]] for x in a["ranking"]],
                            ["Trained city", "Climate similarity", "Class"]))
    if s:
        rows = [["Total predicted energy", f"{f(s['total_energy_mwh'])} MWh"],
                ["Average predicted load", f"{f(s['average_power_mw'])} MW"],
                ["Peak predicted load", f"{f(s['peak_power_mw'])} MW at {s['peak_timestamp']}"],
                ["Hours predicted / requested", f"{s['hours_predicted']} / {s['hours_requested']}"]]
        if s.get("total_energy_interval_mwh"):
            lo, hi = s["total_energy_interval_mwh"]
            rows.append(["Total energy 90 % interval", f"{f(lo)} – {f(hi)} MWh ({s.get('total_interval_basis')})"])
        if s.get("observed_total_energy_mwh") is not None:
            rows.append(["Observed total energy", f"{f(s['observed_total_energy_mwh'])} MWh (MAPE {f(s['observed_MAPE_percent'], 2)} %)"])
        rows.append(["Units", s["energy_definition"]])
        parts.append(_table(rows, ["Quantity", "Value"]))
    if win:
        parts.append(f"<h2>Selected model</h2><p><b>{html.escape(win.display_name)}</b> · version {html.escape(win.version)} · "
                     f"Model Fitness Score <b>{f(win.fitness)}/100</b> ({html.escape(win.eligibility)})</p>")
        parts.append(_table([[LABELS[c.name], f(c.score), f"{c.weight:.0%}", f"{c.effective_weight:.0%}", c.confidence, c.detail]
                             for c in win.components], ["Component", "Score", "Weight", "Effective", "Confidence", "Evidence"]))
        parts.append("<p>Artifacts: " + html.escape(", ".join(info.get("artifacts", {}).values())) + "</p>")
    parts.append("<h2>All candidates</h2>" + _table(
        [[x.rank or "—", x.display_name, x.eligibility, f(x.fitness), "; ".join(x.reasons) or "—"] for x in resp.scores],
        ["Rank", "Model", "Eligibility", "Fitness", "Notes"]))
    parts.append("<h2>Selection rationale</h2><ul>" + "".join(f"<li>{html.escape(t)}</li>" for t in resp.selection_explanation) + "</ul>")
    if resp.issues:
        parts.append("<h2>Data quality and warnings</h2>" + _table([[i.severity, i.code, i.message] for i in resp.issues], ["Severity", "Code", "Message"]))
    parts.append("<h2>Processing log</h2>" + _table([[e.step, e.action, e.detail, e.rows_affected] for e in resp.processing_log],
                                                     ["Step", "Action", "Detail", "Rows"]))
    parts.append("<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(t)}</li>" for t in LIMITATIONS) + "</ul>")
    css = ("body{font:14px/1.5 system-ui,sans-serif;max-width:980px;margin:32px auto;padding:0 16px;color:#1f2933}"
           "h1{font-size:24px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #d9e2ec;padding-bottom:4px}"
           ".meta{color:#52606d}table{border-collapse:collapse;width:100%;margin:8px 0}"
           "th,td{border:1px solid #d9e2ec;padding:5px 8px;text-align:left;vertical-align:top}th{background:#f0f4f8}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Prediction report {html.escape(resp.request_id)}</title>"
            f"<style>{css}</style></head><body>{''.join(parts)}</body></html>").encode("utf-8")
