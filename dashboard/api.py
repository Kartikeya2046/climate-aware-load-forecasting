"""HTTP API for the web frontend (dashboard/web). Run from inside dashboard/:

    python -m uvicorn api:app --port 8000

A thin layer over dashboard/core: it only parses requests, calls the same service functions the
Streamlit app calls, and serialises what they return. No model, validation or scoring logic here.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import climate
from core.data import CITIES, DASHBOARD_DIR, config, record_span, repository_record
from core.evidence import WEIGHTS_STUDY_PATH, load_evidence, training_reference
from core.models import discovered_artifacts, registry
from core.report import LIMITATIONS, predictions_csv, report_html, response_json
from core.schema import VARIABLES, Issue, PredictionRequest
from core.scoring import COMPONENTS, LABELS
from core.service import MAX_HISTORY, aggregate, prepare_inputs, run_analog, run_prediction, weather_lead_hours
from core.validation import manual_frame, process, propose_mapping, read_csv, suggest_unit

app = FastAPI(title="Energy Intelligence API")
LOCK = threading.Lock()  # ponytail: one global lock serialises inference (5-10 s each); a worker queue if traffic ever needs it
# In-memory stores shared by all visitors, addressed only by unguessable ids that each browser keeps for itself.
# ponytail: oldest entries are evicted at these caps and everything is lost on restart; a database if results must persist.
UPLOADS: dict[str, dict] = {}
PENDING: dict[str, dict] = {}
RESULTS: dict[str, dict] = {}
KEEP = {id(UPLOADS): 8, id(PENDING): 16, id(RESULTS): 32}
DEFAULT_START = {"dayton": "2015-07-06", "pittsburgh": "2015-07-06"}


def remember(store: dict, key: str, value: dict) -> None:
    store[key] = value
    while len(store) > KEEP[id(store)]:
        store.pop(next(iter(store)))


def frame(df: pd.DataFrame) -> dict:
    """DataFrame -> columnar JSON; timestamps are naive local standard time."""
    out = {"t": [t.strftime("%Y-%m-%dT%H:%M") for t in df.index]}
    for c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        out[c] = [None if not math.isfinite(v) else round(v, 4) for v in x]
    return out


def js(o):
    if isinstance(o, pd.DataFrame):
        return frame(o)
    if isinstance(o, dict):
        return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        o = o.tolist()
    if isinstance(o, (list, tuple, set)):
        return [js(x) for x in o]
    if is_dataclass(o):
        return js(asdict(o))
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def ok(payload) -> JSONResponse:
    return JSONResponse(js(payload))


def models():
    with LOCK:
        reg = registry()
        for a in reg.values():
            a.ensure_loaded()
    return reg


@app.on_event("startup")
def warm():
    threading.Thread(target=models, daemon=True).start()


# ------------------------------------------------------------------------------------ reference data
@app.get("/api/meta")
def meta():
    reg = models()
    cities = []
    for c, v in CITIES.items():
        lo, hi = record_span(c)
        rows = []
        for a in reg.values():
            good, why = a.supports_location(c) if a.loaded else (False, a.load_error)
            dom = "trained here" if c in a.trained_on else "evaluated here" if c in a.evaluated_on else "outside research domain"
            rows.append({"id": a.id, "applicable": bool(good), "domain": dom if good else None, "reason": why})
        cities.append({**asdict(v), "span": [str(lo), str(hi)], "default_start": DEFAULT_START.get(c, "2023-07-10"), "models": rows})
    ev = load_evidence()
    for per_city in ev.get("models", {}).values():  # weekly window lists are large and unused by the UI
        for e in per_city.values():
            for mode in e.values():
                if isinstance(mode, dict):
                    mode.pop("per_window", None)
    return ok({
        "cities": cities,
        "models": [{**a.info(), "decomposes": a.decomposes} for a in reg.values()],
        "variables": {k: {"label": v.label, "unit": v.unit, "units": list(v.units), "factors": v.units, "lo": v.lo, "hi": v.hi, "help": v.help,
                          "derivable": v.derivable} for k, v in VARIABLES.items()},
        "weights": config()["fitness_weights"], "labels": LABELS, "components": COMPONENTS,
        "limits": {"max_horizon_hours": config()["validation"]["max_horizon_hours"],
                   "max_upload_mb": config()["validation"]["max_upload_mb"], "max_history_hours": MAX_HISTORY,
                   "near_tie_points": config()["selection"]["near_tie_points"]},
        "evidence": {k: ev.get(k) for k in ("generated_utc", "models", "benchmarks", "research_reported")},
        "weights_study": json.loads(WEIGHTS_STUDY_PATH.read_text()) if WEIGHTS_STUDY_PATH.exists() else None,
        "limitations": LIMITATIONS, "artifacts": discovered_artifacts(),
    })


@app.get("/api/manual-fields")
def manual_fields(city: str, month: int):
    """Union of the inputs needed by the models applicable at `city`, prefilled with the recorded monthly median."""
    if city not in CITIES:
        raise HTTPException(400, "Manual entry needs a trained city.")
    need: dict[str, list[str]] = {}
    for a in models().values():
        if a.loaded and a.supports_location(city)[0]:
            for v in a.weather_inputs:
                need.setdefault(v, []).append(a.display_name)
    rec = repository_record(city)
    m = rec[rec.index.month == month]
    out = []
    for v in need:
        var = VARIABLES[v]
        if var.derivable:
            continue
        default = float(m[v].median()) if v in m and m[v].notna().any() else (var.lo + var.hi) / 2
        out.append({"variable": v, "default": round(default, 2), "needed_by": need[v]})
    return ok(out)


@app.get("/api/sample")
def sample(city: str, start: str, end: str):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    rec = repository_record(city).loc[s - pd.Timedelta(hours=MAX_HISTORY + 24):e + pd.Timedelta(hours=weather_lead_hours(city))]
    if rec.empty:
        raise HTTPException(404, "The selected timeframe is outside the recorded period.")
    return Response(rec.to_csv().encode(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="sample_{city}_{s:%Y%m%d}.csv"'})


# ------------------------------------------------------------------------------------ input -> validation
def describe_upload(uid: str) -> dict:
    u = UPLOADS[uid]
    ing = u["ingested"]
    cols = []
    if not any(i.severity == "fatal" for i in ing.issues):
        prop = propose_mapping(list(ing.raw.columns))
        for c in ing.raw.columns:
            p = prop[c]
            unit, warn = suggest_unit(p["variable"], ing.raw[c]) if p["variable"] in VARIABLES else (None, None)
            cols.append({"name": c, "variable": p["variable"], "status": p["status"], "sample": list(ing.raw[c].head(3).astype(str)),
                         "unit": unit, "unit_warning": warn})
    return {"upload_id": uid, "name": u["name"], "size": len(u["bytes"]), "rows": len(ing.raw), "columns": cols,
            "head": ing.raw.head(8).to_dict("records"), "issues": ing.issues}


@app.post("/api/upload")
async def upload(request: Request, filename: str):
    data = await request.body()
    if len(data) > (config()["validation"]["max_upload_mb"] + 1) * 1024 * 1024:
        raise HTTPException(413, "File too large.")
    ing = read_csv(data, filename)
    uid = "u" + hashlib.sha256(data).hexdigest()[:20]
    remember(UPLOADS, uid, {"name": filename, "bytes": data, "ingested": ing})
    return ok(describe_upload(uid))


class ValidateBody(BaseModel):
    city: str
    start: str
    end: str
    mode: str = "forecast"
    aggregation: str = "hourly"
    source: str  # "recorded" | "manual" | "csv"
    values: dict[str, float] = {}
    units: dict[str, str] = {}
    use_recorded: bool = True
    upload_id: str | None = None
    mapping: dict[str, str | None] = {}
    ts_kind: str = "local_standard"
    context: str = "provided"


def pending_payload(pid: str | None, p: dict) -> dict:
    req, prep = p.get("request"), p.get("prepared")
    preview = None
    if req is not None:
        f = prep.frame if prep is not None else req.weather
        preview = f.loc[req.start - pd.Timedelta(hours=6):req.end].head(60).round(3)
    return {"pending_id": pid, "issues": p["issues"], "log": p["log"], "preview": preview,
            "fatal": any(i.severity == "fatal" for i in p["issues"]),
            "needs_confirmation": any(i.severity == "recoverable" for i in p["issues"]),
            "history_hours_observed": getattr(prep, "history_hours_observed", None),
            "ground_truth_hours": getattr(prep, "ground_truth_hours", None),
            "rows": None if req is None else int(len(prep.frame if prep is not None else req.weather))}


@app.post("/api/validate")
def validate(b: ValidateBody):
    start, end = pd.Timestamp(b.start), pd.Timestamp(b.end)
    hours = int((end - start) / pd.Timedelta(hours=1)) + 1
    max_h = config()["validation"]["max_horizon_hours"]
    if not 0 < hours <= max_h:
        raise HTTPException(400, f"The end must be after the start and the timeframe at most {max_h} hours.")
    city = b.city
    if b.source != "csv" and city not in CITIES:
        raise HTTPException(400, "Climate matching needs a weather time series: upload a CSV, or choose a trained city.")
    models()
    proc = raw = name = None
    if b.source == "manual":
        lo, hi = record_span(city)
        use_rec = b.use_recorded and lo + pd.Timedelta(hours=MAX_HISTORY) <= start <= hi + pd.Timedelta(hours=1)
        values = {}
        for v, x in b.values.items():
            scale, off = VARIABLES[v].units[b.units.get(v, VARIABLES[v].unit)]
            values[v] = x * scale + off
        f, issues, log = manual_frame(city, start, end, values, 0 if use_rec else MAX_HISTORY)
        req = PredictionRequest(city, start, end, "manual", f, context_source="repository" if use_rec else "none",
                                prediction_mode=b.mode, aggregation=b.aggregation, units=b.units, scenario=True,
                                processing_log=log, issues=issues, quality={"rows_hourly": len(f)})
    else:
        if b.source == "recorded":  # same path as downloading the sample CSV and uploading it
            rec = repository_record(city).loc[start - pd.Timedelta(hours=MAX_HISTORY + 24):end + pd.Timedelta(hours=weather_lead_hours(city))]
            if rec.loc[start:end].drop(columns="load_mw").dropna(how="all").empty:
                lo, hi = record_span(city)
                raise HTTPException(400, f"No recorded weather for {CITIES[city].name} in this timeframe. Records run {lo:%Y-%m-%d} to {hi:%Y-%m-%d}.")
            raw, name = rec.to_csv().encode(), f"recorded_{city}.csv"
            ing = read_csv(raw, name)
            mapping = {c: p["variable"] for c, p in propose_mapping(list(ing.raw.columns)).items()}
            units, ts_kind, ctx = {}, "local_standard", "provided"
        else:
            u = UPLOADS.get(b.upload_id or "")
            if not u:
                raise HTTPException(400, "Upload the CSV again; the server no longer has it.")
            ing, raw, name = u["ingested"], u["bytes"], u["name"]
            mapping, units, ts_kind = b.mapping, b.units, b.ts_kind
            ctx = "provided" if city == "auto" else b.context
        ts_cols = [c for c, v in mapping.items() if v == "timestamp"]
        used = [v for v in mapping.values() if v and v != "timestamp"]
        if len(ts_cols) != 1:
            raise HTTPException(400, "Map exactly one column to the timestamp.")
        if len(used) != len(set(used)):
            raise HTTPException(400, "Two columns are mapped to the same variable.")
        proc = process(ing.raw, dict(mapping), units, city if city in CITIES else "", ts_cols[0], ts_kind)
        log = ing.log + proc.log
        if not proc.fatal:
            win = proc.frame.loc[start:end]
            if win.empty or win.drop(columns="load_mw", errors="ignore").isna().all().all():
                proc.issues.append(Issue("fatal", "timeframe", f"The data ({proc.quality['span'][0]} → {proc.quality['span'][1]}) has no weather "
                                                               f"inside the requested timeframe {start} → {end}."))
        if proc.fatal:
            return ok(pending_payload(None, {"issues": proc.issues, "log": log}))
        req = PredictionRequest(city, start, end, "csv", proc.frame, context_source=ctx, prediction_mode=b.mode,
                                aggregation=b.aggregation, units=units, processing_log=log, issues=proc.issues, quality=proc.quality)
    prep = None if city == "auto" else prepare_inputs(req)  # analog inputs are prepared per candidate city in run_analog
    p = {"request": req, "prepared": prep, "processed": proc, "raw": raw, "name": name, "source": b.source,
         "issues": req.issues + (prep.issues if prep else []), "log": req.processing_log + (prep.log if prep else [])}
    remember(PENDING, req.request_id, p)
    return ok(pending_payload(req.request_id, p))


# ------------------------------------------------------------------------------------ run + results
class RunBody(BaseModel):
    pending_id: str
    name: str = ""
    confirmed: bool = False


def result_payload(rid: str) -> dict:
    r = RESULTS[rid]
    req, resp, proc = r["request"], r["response"], r["processed"]
    preds = resp.predictions
    proj = climate.project(preds) if preds is not None else None
    return {
        "id": rid, "source": r["source"], "upload_name": r["name"],
        "request": {"location_id": req.location_id, "start": str(req.start), "end": str(req.end), "input_mode": req.input_mode,
                    "prediction_mode": req.prediction_mode, "scenario": req.scenario, "context_source": req.context_source,
                    "aggregation": req.aggregation},
        "status": resp.status, "message": resp.message, "selected_model": resp.selected_model,
        "selected_fitness": resp.selected_fitness, "scores": resp.scores, "predictions": preds,
        "alternatives": {k: v[["pred_mw"]] for k, v in resp.alternatives.items()},
        "daily": aggregate(preds, "daily") if preds is not None else None,
        "summary": resp.summary, "selection_explanation": resp.selection_explanation,
        "processing_log": resp.processing_log, "issues": resp.issues, "analog": resp.analog or None,
        "projection": None if proj is None else [[round(a, 3), round(b, 3)] for a, b in proj.to_numpy()[:: max(1, len(proj) // 400)]],
        "quality": proc.quality if proc else None, "mapping": proc.mapping if proc else None,
        "has_processed": bool(proc and proc.frame is not None), "has_original": r["raw"] is not None and r["source"] == "csv",
    }


@app.post("/api/run")
def run(b: RunBody):
    p = PENDING.get(b.pending_id)
    if not p:
        raise HTTPException(400, "Validate the inputs again; the server no longer has them.")
    if any(i.severity == "fatal" for i in p["issues"]):
        raise HTTPException(400, "Fatal validation errors block the run.")
    if any(i.severity == "recoverable" for i in p["issues"]) and not b.confirmed:
        raise HTTPException(400, "Confirm the corrections before running.")
    req = p["request"]
    with LOCK:
        if req.location_id == "auto":
            req, resp = run_analog(req, b.name.strip())
        else:
            resp = run_prediction(req, p["prepared"])
    remember(RESULTS, resp.request_id, {"request": req, "response": resp, "processed": p["processed"], "raw": p["raw"],
                                        "name": p["name"], "source": p["source"]})
    return ok(result_payload(resp.request_id))


def get_result(rid: str) -> dict:
    if rid not in RESULTS:
        raise HTTPException(404, "This result is no longer on the server. Run the prediction again.")
    return RESULTS[rid]


@app.get("/api/result/{rid}")
def result(rid: str):
    get_result(rid)
    return ok(result_payload(rid))


@app.get("/api/result/{rid}/export/{kind}")
def export(rid: str, kind: str):
    r = get_result(rid)
    req, resp, proc = r["request"], r["response"], r["processed"]
    if kind == "csv" and resp.predictions is not None:
        body, mime, name = predictions_csv(resp), "text/csv", f"predictions_{rid}.csv"
    elif kind == "html":
        body, mime, name = report_html(req, resp), "text/html", f"report_{rid}.html"
    elif kind == "json":
        body, mime, name = response_json(resp), "application/json", f"response_{rid}.json"
    elif kind == "processed" and proc and proc.frame is not None:
        body, mime, name = proc.frame.to_csv().encode(), "text/csv", "processed_input.csv"
    elif kind == "original" and r["raw"] is not None:
        body, mime, name = r["raw"], "text/csv", "original_upload.csv"
    else:
        raise HTTPException(404, "Nothing to export for this result.")
    return Response(body, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/result/{rid}/climate/{model_id}")
def climate_detail(rid: str, model_id: str):
    """Input vs training-weather distributions and correlation structure for one scored candidate."""
    resp = get_result(rid)["response"]
    s = next((x for x in resp.scores if x.model_id == model_id), None)
    cl = (s.training_domain.get("climate") or {}) if s else {}
    if not cl.get("variables") or resp.predictions is None:
        raise HTTPException(404, "No climate comparison for this model.")
    rc, rs, re_ = training_reference(model_id)
    inp = resp.predictions
    ref = climate.seasonal(climate.reference_weather(rc, str(rs), str(re_)), set(inp.index.month))
    hist = {}
    for v in cl["variables"]:
        a, b = inp[v].dropna().to_numpy(), ref[v].dropna().to_numpy()
        edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=36)
        hist[v] = {"edges": edges, "input": np.histogram(a, edges, density=True)[0], "reference": np.histogram(b, edges, density=True)[0]}
    corr = None
    if cl["method"] == "distribution":
        vv = [v for v in cl["variables"] if inp[v].std() > 0 and ref[v].std() > 0]
        if len(vv) >= 2:
            corr = {"variables": vv, "input": inp[vv].dropna().corr().to_numpy(), "reference": ref[vv].dropna().corr().to_numpy()}
    return ok({"reference_city": rc, "months": sorted(set(ref.index.month)), "hist": hist, "corr": corr})


@app.get("/api/climate/space")
def climate_space():
    """Fixed PCA space of the six cities' recorded weather and the city-to-city similarity matrix."""
    _, pca, pts = climate.pca_space()
    ids = list(CITIES)
    mat = climate.city_matrix()
    z = [[100.0] * len(ids) for _ in ids]
    for k, v in mat.items():
        a, b = (ids.index(x) for x in k.split("|"))
        z[a][b] = z[b][a] = round(v["overall"], 1)
    return ok({"points": {c: pts[pts.city == c][["PC1", "PC2"]].iloc[::5].round(3).to_numpy() for c in ids},
               "explained": pca.explained_variance_ratio_, "cities": ids, "matrix": z})


DIST = DASHBOARD_DIR / "web" / "dist"
if DIST.exists():  # production: the built frontend is served from the same port
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
