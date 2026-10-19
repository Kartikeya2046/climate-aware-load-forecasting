"""Energy Intelligence Dashboard (Streamlit UI). Run from the repository root:

    streamlit run dashboard/app.py

All inference, validation and scoring live in dashboard/core; this file only lays out pages.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

import charts
from core import climate
from core.data import CITIES, config, record_span, repository_record
from core.evidence import WEIGHTS_STUDY_PATH, load_evidence
from core.models import discovered_artifacts, registry
from core.report import LIMITATIONS, predictions_csv, predictions_frame, report_html, response_json
from core.schema import VARIABLES, PredictionRequest
from core.scoring import COMPONENTS, LABELS
from core.service import MAX_HISTORY, aggregate, prepare_inputs, run_analog, run_prediction, weather_lead_hours
from core.validation import manual_frame, process, propose_mapping, read_csv, suggest_unit

st.set_page_config(page_title="Energy Intelligence Dashboard", page_icon="⚡", layout="wide")
st.markdown("""<style>
.kpi{border:1px solid #e7e6e2;border-radius:10px;padding:14px 16px;background:#fcfcfb;height:100%}
.kpi .l{color:#52514e;font-size:13px}.kpi .v{color:#0b0b0b;font-size:26px;font-weight:650;line-height:1.25}
.kpi .s{color:#52514e;font-size:12px}.hero .v{font-size:40px}.hero{border-color:#2a78d6}
</style>""", unsafe_allow_html=True)

S = st.session_state
NAMES = {c: v.name for c, v in CITIES.items()}
DEFAULT_WINDOW = {"dayton": "2015-07-06", "pittsburgh": "2015-07-06"}


@st.cache_resource(show_spinner="Loading and health-checking the trained models…")
def models():
    reg = registry()
    for a in reg.values():
        a.ensure_loaded()
    return reg


def kpi(label, value, sub="", hero=False):
    st.markdown(f"<div class='kpi{' hero' if hero else ''}'><div class='l'>{label}</div><div class='v'>{value}</div>"
                f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def fmt_energy(mwh):
    return f"{mwh / 1000:,.2f} GWh" if abs(mwh) >= 10_000 else f"{mwh:,.0f} MWh"


def result():
    return S.get("result")


def reset():
    for k in list(S.keys()):
        del S[k]


def city_changed():
    c = S.get("city")
    if c in CITIES:
        d = pd.Timestamp(DEFAULT_WINDOW.get(c, "2023-07-10"))
        S.start_date, S.start_hour, S.end_date, S.end_hour = d.date(), 0, (d + pd.Timedelta(days=2)).date(), 23
    for k in ("pending", "ingested", "upload_name"):
        S.pop(k, None)


def analog_text(resp):
    a = resp.analog
    if resp.status != "ok":
        return (f"**{a['name']}** is not a trained city, and no trained city had an eligible model for this request (closest climate: "
                f"**{NAMES[a['city']]}**, {a['similarity']:.1f}/100). See the reasons below and the Climate Similarity page.")
    return (f"**{a['name']}** is not a trained city. Most similar climate with an eligible model: **{NAMES[a['city']]}** "
            f"({a['similarity']:.1f}/100, {a['classification'].lower()}); predicted as this weather on {NAMES[a['city']]}'s grid.")


def no_result(msg="Run a prediction in the Prediction Workspace first."):
    st.info(msg, icon="ℹ️")
    st.page_link(PAGES["workspace"], label="Open the Prediction Workspace", icon="🧭")


# ----------------------------------------------------------------------------------- overview
def page_overview():
    st.title("Energy Intelligence Dashboard")
    st.caption("Climate-aware electricity consumption prediction with transparent, evidence-based model selection. "
               "Weather (manual or CSV) is validated, every trained model is checked for eligibility and scored, "
               "and the best-scoring model predicts hourly load for the requested timeframe.")
    r = result()
    if not r:
        no_result("No prediction yet. Choose a location, a timeframe and weather input in the Prediction Workspace.")
        c = st.columns(3)
        reg = models()
        c[0].metric("Registered models", len(reg), help="Two-branch TCN (Atlanta, Las Vegas weather branches) and the Dayton LSTM")
        c[1].metric("Supported locations", len(CITIES))
        c[2].metric("Models loaded", sum(a.loaded for a in reg.values()))
        return
    req, resp = r["request"], r["response"]
    city = NAMES.get(req.location_id, req.location_id)
    if resp.analog:
        city = f"{resp.analog['name']} (analog: {city})"
    st.subheader(f"{city} · {req.start:%Y-%m-%d %H:%M} → {req.end:%Y-%m-%d %H:%M} (local standard time)")
    if resp.analog:
        st.info(analog_text(resp), icon="🧭")
    if resp.status != "ok":
        st.error(f"**{resp.status.replace('_', ' ').title()}** — {resp.message}", icon="⛔")
        for t in resp.selection_explanation:
            st.write("•", t)
        return
    s = resp.summary
    win = next(x for x in resp.scores if x.selected)
    target = S.get("display_target", "Total energy (MWh)")
    c = st.columns([1.6, 1, 1, 1])
    with c[0]:
        if target.startswith("Average"):
            kpi("Average predicted load", f"{s['average_power_mw']:,.0f} MW", f"over {s['hours_predicted']} h", hero=True)
        else:
            iv = s.get("total_energy_interval_mwh")
            kpi("Total predicted energy", fmt_energy(s["total_energy_mwh"]),
                f"90 % interval {fmt_energy(iv[0])} – {fmt_energy(iv[1])}" if iv else "interval unavailable for this horizon", hero=True)
    with c[1]:
        kpi("Average per interval", f"{s['average_power_mw']:,.0f} MWh", "per 1-hour interval (= average MW)")
    with c[2]:
        kpi("Peak predicted load", f"{s['peak_power_mw']:,.0f} MW", pd.Timestamp(s["peak_timestamp"]).strftime("%a %Y-%m-%d %H:%M"))
    with c[3]:
        n_ok = sum(x.eligibility != "ineligible" for x in resp.scores)
        kpi("Eligible candidates", f"{n_ok} / {len(resp.scores)}", f"{sum(x.eligibility == 'conditional' for x in resp.scores)} conditional")
    c = st.columns([1.6, 1, 1, 1])
    with c[0]:
        kpi("Selected model", win.display_name, f"{win.family} · {win.version}")
    with c[1]:
        kpi("Model Fitness Score", f"{win.fitness:.1f} / 100", "weighted rubric, not a probability")
    with c[2]:
        fat = sum(i.severity == "fatal" for i in resp.issues)
        warn = sum(i.severity != "fatal" for i in resp.issues)
        kpi("Data quality", "✅ OK" if not warn else f"⚠️ {warn} notice(s)", "see Data Quality page")
    with c[3]:
        mode = "One hour ahead" if req.prediction_mode == "one_step" else "Free-running forecast"
        kpi("Prediction mode", mode, "scenario simulation" if req.scenario else req.context_source + " history")
    if s.get("observed_MAPE_percent") is not None:
        st.success(f"Observed load is available for every predicted hour: MAPE {s['observed_MAPE_percent']:.2f} %, "
                   f"MAE {s['observed_MAE_MW']:,.1f} MW; observed total {fmt_energy(s['observed_total_energy_mwh'])}.", icon="📏")
    df = resp.predictions
    st.plotly_chart(charts.timeseries(df, {}, {}, "power", win.display_name), use_container_width=True)
    if req.prediction_mode != "one_step":
        ev = load_evidence().get("benchmarks", {}).get(req.location_id, {})
        b = ev.get("calendar_baseline_only") or ev.get("naive_168h")
        if win.metrics.get("MAPE_percent") and b and win.metrics["MAPE_percent"] > b["MAPE_percent"]:
            st.warning(f"Held-out evidence: in free-running mode this model's MAPE was {win.metrics['MAPE_percent']:.1f} %, worse than the "
                       f"{'calendar baseline alone' if 'model' in b else 'same-hour-last-week benchmark'} ({b['MAPE_percent']:.1f} %). "
                       "The models are most reliable one hour ahead with observed load.", icon="⚠️")


# ----------------------------------------------------------------------------------- workspace
def required_inputs(city):
    reg = models()
    need = {}
    for a in reg.values():
        if a.loaded and a.supports_location(city)[0]:
            for v in a.weather_inputs:
                need.setdefault(v, []).append(a.display_name)
    return need


def page_workspace():
    st.title("Prediction workspace")
    reg = models()
    st.subheader("1 · Location")
    opts = ["auto"] + list(CITIES)
    if "city" not in S:
        S.city = "auto"
        city_changed()
    city = st.selectbox("Location", opts, key="city", on_change=city_changed,
                        format_func=lambda c: f"{NAMES[c]} — {CITIES[c].pair}" if c in CITIES else "Any city — match by climate (upload its weather)")
    if city == "auto":
        st.text_input("City name (optional, used in labels)", key="custom_name", placeholder="e.g. Kirk land")
        st.info("Upload the city's weather CSV. It is compared with the recorded climate of every trained city (project climate-similarity "
                "score); the most similar city with an eligible model is used, and the prediction is that weather on that city's grid "
                "(its load scale, infrastructure and population are assumed).", icon="🧭")
    else:
        lo, hi = record_span(city)
        cols = st.columns([1.2, 2])
        cols[0].markdown(f"**{NAMES[city]}**  \n{CITIES[city].utility}  \nResearch pair: {CITIES[city].pair}  \n"
                         f"Recorded data: {lo:%Y-%m-%d} → {hi:%Y-%m-%d}  \nUTC offset (standard): −{CITIES[city].utc_offset} h")
        rows = []
        for a in reg.values():
            ok, why = a.supports_location(city) if a.loaded else (False, a.load_error)
            dom = "trained here" if city in a.trained_on else "evaluated here" if city in a.evaluated_on else "outside research domain"
            rows.append({"Model": a.display_name, "Technically applicable": "yes" if ok else "no", "Domain": dom if ok else "—", "Reason": why})
        cols[1].dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("2 · Timeframe")
    c = st.columns([1, .6, 1, .6, 1.4])
    S.setdefault("start_date", pd.Timestamp("2023-07-10").date()); S.setdefault("end_date", pd.Timestamp("2023-07-12").date())
    S.setdefault("start_hour", 0); S.setdefault("end_hour", 23)
    c[0].date_input("Start date", key="start_date")
    c[1].selectbox("Start hour", range(24), key="start_hour", format_func=lambda h: f"{h:02d}:00")
    c[2].date_input("End date", key="end_date")
    c[3].selectbox("End hour", range(24), key="end_hour", format_func=lambda h: f"{h:02d}:00")
    start = pd.Timestamp(S.start_date) + pd.Timedelta(hours=S.start_hour)
    end = pd.Timestamp(S.end_date) + pd.Timedelta(hours=S.end_hour)
    hours = int((end - start) / pd.Timedelta(hours=1)) + 1
    max_h = config()["validation"]["max_horizon_hours"]
    c[4].markdown(f"**{max(hours, 0)} hourly intervals**  \nresolution: 1 h (models are hourly); timestamps are local standard time (no DST)")
    tf_ok = 0 < hours <= max_h
    if not tf_ok:
        st.error(f"The end must be after the start and the timeframe at most {max_h} hours.", icon="⛔")
    c = st.columns(3)
    c[0].radio("Prediction mode", ["forecast", "one_step"], key="mode", horizontal=False,
               format_func=lambda m: "Forecast / scenario (free-running)" if m == "forecast" else "One hour ahead (needs observed load every hour)",
               help="All models were trained to predict the next hour from observed load history. One-hour-ahead mode is the evaluated research "
                    "setting; forecast mode feeds each prediction back as history and is markedly less accurate (see Model Selection).")
    c[1].radio("Primary result", ["Total energy (MWh)", "Energy per interval (MWh)", "Average power (MW)"], key="display_target",
               help="All three are computed; this chooses the headline. Hourly MWh = average MW × 1 h.")
    c[2].radio("Analytics aggregation", ["hourly", "daily"], key="aggregation", horizontal=True)

    st.subheader("3 · Weather input")
    mode_in = st.radio("Input method", ["manual", "csv"], key="input_mode", horizontal=True,
                       format_func=lambda m: "Manual entry (one weather scenario)" if m == "manual" else "Upload CSV")
    pending = None
    if mode_in == "manual":
        pending = manual_section(city, start, end, tf_ok)
    else:
        pending = csv_section(city, start, end, tf_ok)
    if pending:
        S.pending = pending
    validation_section(city)


def manual_section(city, start, end, tf_ok):
    if city == "auto":
        st.info("Climate matching needs a weather time series: upload a CSV, or choose a trained city for manual entry.", icon="ℹ️")
        return None
    if S.get("mode") == "one_step":
        st.warning("One-hour-ahead mode needs observed load for every hour; a manual scenario has none. Switch to forecast mode or upload a CSV.", icon="⚠️")
    need = required_inputs(city)
    rec = repository_record(city)
    month = rec[rec.index.month == start.month]
    st.caption("Fields are the union of inputs required by the models applicable here. Values are prefilled with the recorded median "
               f"for {start:%B} in {NAMES[city]} (a documented starting point, edit freely). Radiation inputs are derived from solar geometry.")
    values, units = {}, {}
    vars_ = [v for v in need if not VARIABLES[v].derivable]
    with st.form("manual"):
        cols = st.columns(3)
        for i, v in enumerate(vars_):
            var = VARIABLES[v]
            with cols[i % 3]:
                u = st.selectbox(f"{var.label} unit", list(var.units), key=f"u_{v}", label_visibility="collapsed")
                scale, off = var.units[u]
                default = float(month[v].median()) if v in month and month[v].notna().any() else (var.lo + var.hi) / 2
                x = st.number_input(f"{var.label} ({u})", value=round((default - off) / scale, 2), key=f"m_{v}_{u}",
                                    help=f"{var.help} Needed by: {', '.join(need[v])}. Valid range {var.lo}–{var.hi} {var.unit}.")
                values[v], units[v] = x * scale + off, u
        lo, hi = record_span(city)
        has_rec = lo + pd.Timedelta(hours=MAX_HISTORY) <= start <= hi + pd.Timedelta(hours=1)
        use_rec = st.checkbox(f"Use recorded {NAMES[city]} weather and load before the start as model history",
                              value=has_rec, disabled=not has_rec,
                              help="Available when the start lies in the recorded period. Without it the scenario weather is also used for the "
                                   "look-back window and load history is unknown (two-branch TCN: residual history assumed 0; LSTM: ineligible).")
        go = st.form_submit_button("Validate inputs", type="primary", disabled=not tf_ok)
    if not go:
        return None
    frame, issues, log = manual_frame(city, start, end, values, 0 if use_rec else MAX_HISTORY)
    req = PredictionRequest(city, start, end, "manual", frame, context_source="repository" if use_rec else "none",
                            prediction_mode=S.mode, aggregation=S.aggregation, units=units, scenario=True,
                            processing_log=log, issues=issues, quality={"rows_hourly": len(frame)})
    return build_pending(req, None)


def csv_section(city, start, end, tf_ok):
    up = st.file_uploader("Weather CSV (hourly or finer; one timestamp column)", type=["csv", "txt"], key="upload",
                          help=f"Limit {config()['validation']['max_upload_mb']} MB. Columns are matched to known variable names and aliases; "
                               "anything unrecognised must be mapped explicitly or is ignored.")
    if city in CITIES:
        with st.expander("Need a file to try the workflow? Download a sample built from the project's recorded data"):
            lo_s = start - pd.Timedelta(hours=MAX_HISTORY + 24)
            hi_s = end + pd.Timedelta(hours=weather_lead_hours(city))
            rec = repository_record(city).loc[lo_s:hi_s]
            if rec.empty:
                st.caption("The selected timeframe is outside the recorded period.")
            else:
                st.download_button("Download sample CSV (recorded weather + observed load)", rec.to_csv().encode(),
                                   f"sample_{city}_{start:%Y%m%d}.csv", "text/csv")
                st.caption(f"{len(rec)} hourly rows from {rec.index[0]} to {rec.index[-1]}, local standard time. "
                           "Includes the look-back history the models need before the start.")
    up = up or S.get("_injected_upload")  # test seam: AppTest cannot drive st.file_uploader
    if not up:
        return None
    data = up.getvalue()
    if S.get("upload_name") != (up.name, len(data)):
        S.upload_name = (up.name, len(data))
        S.ingested = read_csv(data, up.name)
        S.raw_bytes = data
        S.pop("pending", None)
    ing = S.ingested
    for i in ing.issues:
        (st.error if i.severity == "fatal" else st.warning)(i.message, icon="⛔" if i.severity == "fatal" else "⚠️")
    if any(i.severity == "fatal" for i in ing.issues):
        return None
    st.markdown(f"**{up.name}** · {len(ing.raw):,} rows · {ing.raw.shape[1]} columns")
    st.dataframe(ing.raw.head(8), use_container_width=True)
    prop = propose_mapping(list(ing.raw.columns))
    st.markdown("**Column mapping** — confirm what each column means. Only exact names/aliases are pre-selected; nothing unknown is guessed.")
    options = ["— ignore —", "timestamp"] + list(VARIABLES)
    fmt = lambda o: o if o in ("— ignore —", "timestamp") else f"{VARIABLES[o].label} [{o}]"
    mapping, units, notes = {}, {}, []
    head = st.columns([2, 3, 2, 3])
    for h, t in zip(head, ["Column", "Maps to", "Unit", "Sample / note"]):
        h.caption(t)
    for col in ing.raw.columns:
        p = prop[col]
        default = p["variable"] if p["variable"] else "— ignore —"
        row = st.columns([2, 3, 2, 3])
        row[0].markdown(f"`{col}`")
        choice = row[1].selectbox("map", options, index=options.index(default), key=f"map_{col}", format_func=fmt, label_visibility="collapsed")
        mapping[col] = None if choice == "— ignore —" else choice
        sample = ", ".join(ing.raw[col].head(3).astype(str))
        note = {"ambiguous": "⚠️ ambiguous — several columns claim this variable; choose one",
                "ambiguous_timestamp": "⚠️ several timestamp-like columns; choose one", "unmapped": "not recognised — ignored unless mapped"}.get(p["status"], "")
        if choice in VARIABLES:
            sug, warn = suggest_unit(choice, ing.raw[col])
            ulist = list(VARIABLES[choice].units)
            units[choice] = row[2].selectbox("unit", ulist, index=ulist.index(sug), key=f"unit_{col}_{choice}", label_visibility="collapsed")
            if warn:
                note = "⚠️ " + warn
        row[3].caption(f"{sample[:60]}  {note}")
    ts_cols = [c for c, v in mapping.items() if v == "timestamp"]
    dup = pd.Series([v for v in mapping.values() if v and v != "timestamp"]).duplicated()
    c = st.columns(2)
    ts_kind = c[0].radio("Timestamps are", ["local_standard", "utc"], horizontal=True, key="ts_kind",
                         format_func=lambda k: "Local standard time (no DST)" if k == "local_standard" else "UTC",
                         help="Values with an explicit offset (e.g. +00:00, Z) are converted automatically.")
    ctx = "provided" if city == "auto" else c[1].radio("Load history for the models", ["provided", "repository", "none"], key="ctx", horizontal=True,
                     format_func=lambda k: {"provided": "Rows before the start in this file", "repository": "Recorded project data", "none": "None"}[k],
                     help="The models read observed load before each prediction (TCN: 360 h, LSTM: 24 h). Include a load column and history rows, "
                          "or use the recorded data when the dates fall in the recorded period.")
    errors = []
    if len(ts_cols) != 1:
        errors.append("Map exactly one column to 'timestamp'.")
    if dup.any():
        errors.append("Two columns are mapped to the same variable.")
    for e in errors:
        st.error(e, icon="⛔")
    if not st.button("Validate & preview", type="primary", disabled=bool(errors) or not tf_ok):
        return None
    proc = process(ing.raw, {c: v for c, v in mapping.items()}, units, city if city in CITIES else "", ts_cols[0], ts_kind)
    S.processed = proc
    if proc.fatal:
        return {"request": None, "prepared": None, "processed": proc, "issues": proc.issues, "log": ing.log + proc.log}
    win = proc.frame.loc[start:end]
    if win.empty or win.drop(columns="load_mw", errors="ignore").isna().all().all():
        from core.schema import Issue
        proc.issues.append(Issue("fatal", "timeframe", f"The uploaded data ({proc.quality['span'][0]} → {proc.quality['span'][1]}) has no weather "
                                                       f"inside the requested timeframe {start} → {end}."))
        return {"request": None, "prepared": None, "processed": proc, "issues": proc.issues, "log": ing.log + proc.log}
    req = PredictionRequest(city, start, end, "csv", proc.frame, context_source=ctx, prediction_mode=S.mode, aggregation=S.aggregation,
                            units=units, processing_log=ing.log + proc.log, issues=proc.issues, quality=proc.quality)
    return build_pending(req, proc)


def build_pending(req, proc):
    if req.location_id == "auto":  # inputs are prepared per analog city in run_analog
        return {"request": req, "prepared": None, "processed": proc, "issues": req.issues, "log": req.processing_log}
    prep = prepare_inputs(req)
    return {"request": req, "prepared": prep, "processed": proc, "issues": req.issues + prep.issues, "log": req.processing_log + prep.log}


def validation_section(city):
    p = S.get("pending")
    if not p:
        return
    st.subheader("4 · Validation")
    groups = {"fatal": [], "recoverable": [], "warning": []}
    for i in p["issues"]:
        groups[i.severity].append(i)
    c = st.columns(3)
    c[0].metric("Fatal errors", len(groups["fatal"]))
    c[1].metric("Corrections to confirm", len(groups["recoverable"]))
    c[2].metric("Warnings", len(groups["warning"]))
    for i in groups["fatal"]:
        st.error(i.message, icon="⛔")
    for i in groups["recoverable"]:
        st.warning(f"Correction: {i.message}", icon="🛠️")
    for i in groups["warning"]:
        st.info(i.message, icon="⚠️")
    with st.expander(f"Processing log ({len(p['log'])} steps)"):
        st.dataframe(pd.DataFrame([vars(e) for e in p["log"]]), hide_index=True, use_container_width=True)
    if p["request"] is None:
        return
    prep = p["prepared"]
    req = p["request"]
    if prep is None:
        with st.expander("Preview of the processed weather (local standard time)", expanded=False):
            st.dataframe(req.weather.loc[req.start - pd.Timedelta(hours=24):req.end].round(3), use_container_width=True, height=260)
    else:
        with st.expander("Preview of the model input data (processed, local standard time)", expanded=False):
            st.caption(f"{len(prep.frame)} hourly rows incl. look-back context · observed load history before the start: "
                       f"{prep.history_hours_observed} h · observed load inside the timeframe: {prep.ground_truth_hours} h")
            st.dataframe(prep.frame.loc[req.start - pd.Timedelta(hours=24):req.end].round(3), use_container_width=True, height=260)
    fatal = bool(groups["fatal"])
    confirm = True
    if groups["recoverable"]:
        confirm = st.checkbox("I have reviewed the corrections and assumptions above and want to continue", key="confirm")
    if st.button("Run prediction", type="primary", disabled=fatal or not confirm):
        with st.spinner("Checking eligibility, scoring candidates and running inference…"):
            if req.location_id == "auto":
                req, resp = run_analog(req, S.get("custom_name", "").strip())
            else:
                resp = run_prediction(req, prep)
        S.result = {"request": req, "response": resp, "processed": p.get("processed"), "raw_bytes": S.get("raw_bytes") if req.input_mode == "csv" else None,
                    "upload_name": S.get("upload_name")}
        if resp.analog:
            st.info(analog_text(resp), icon="🧭")
        if resp.status == "ok":
            st.toast(f"Prediction complete · {resp.selected_model} selected ({resp.selected_fitness:.1f}/100)", icon="✅")
            st.success("Prediction complete. Open Overview, Model Selection or Energy Analytics.", icon="✅")
        else:
            st.error(f"{resp.status}: {resp.message}", icon="⛔")
            for t in resp.selection_explanation:
                st.write("•", t)


# ----------------------------------------------------------------------------------- model selection
def page_selection():
    st.title("Model selection")
    st.caption("Stage 1 checks eligibility (artifacts, loading, inputs, history, location pathway). Stage 2 scores eligible and conditional "
               "models with the Model Fitness Score — a documented weighted rubric (0–100), not a probability of correctness.")
    r = result()
    if not r:
        return no_result()
    resp = r["response"]
    w = config()["fitness_weights"]
    rows = []
    for s in resp.scores:
        row = {"Rank": s.rank or "—", "Model": s.display_name, "Family": s.family, "Eligibility": s.eligibility,
               "Fitness": s.fitness, "Selected": "✓ selected" if s.selected else ("near tie" if s.near_tie else "")}
        for c in s.components:
            row[LABELS[c.name]] = c.score
        row["Held-out MAPE %"] = s.metrics.get("MAPE_percent")
        row["Version"] = s.version
        rows.append(row)
    df = pd.DataFrame(rows)
    show = st.multiselect("Show eligibility", ["eligible", "conditional", "ineligible"], default=["eligible", "conditional", "ineligible"])
    pc = {LABELS[n]: st.column_config.ProgressColumn(LABELS[n], min_value=0, max_value=100, format="%.0f") for n in COMPONENTS}
    st.dataframe(df[df.Eligibility.isin(show)], hide_index=True, use_container_width=True,
                 column_config={"Fitness": st.column_config.ProgressColumn("Fitness", min_value=0, max_value=100, format="%.1f"), **pc})
    st.caption("Weights: " + " · ".join(f"{LABELS[k]} {v:.0%}" for k, v in w.items()) +
               ". Unavailable components are excluded and the remaining weights renormalised (never scored as perfect).")
    st.plotly_chart(charts.fitness_bars([{"name": s.display_name, "fitness": s.fitness, "selected": s.selected} for s in resp.scores]),
                    use_container_width=True)
    st.subheader("Why this model")
    for t in resp.selection_explanation:
        st.write("•", t)
    ev = load_evidence()
    st.subheader("Candidate details")
    for s in resp.scores:
        icon = "✅" if s.selected else ("🟡" if s.eligibility == "conditional" else "⛔" if s.eligibility == "ineligible" else "⚪")
        with st.expander(f"{icon} {s.display_name} — {s.eligibility}{f' · {s.fitness:.1f}/100' if s.fitness is not None else ''}"):
            if s.reasons:
                st.markdown("**Eligibility notes:** " + "; ".join(s.reasons))
            if s.fitness is not None:
                st.plotly_chart(charts.components_chart(s), use_container_width=True, key=f"cc_{s.model_id}")
                st.dataframe(pd.DataFrame([{"Component": LABELS[c.name], "Score": c.score, "Weight": c.weight, "Effective weight": round(c.effective_weight, 3),
                                            "Confidence": c.confidence, "Evidence / calculation": c.detail} for c in s.components]),
                             hide_index=True, use_container_width=True)
                st.caption(f"Fitness = Σ effective weight × score = {s.fitness:.2f}")
            m = ev.get("models", {}).get(s.model_id, {}).get(r["request"].location_id)
            if m:
                st.markdown(f"**Held-out evidence at {NAMES[r['request'].location_id]}** — {m['holdout']['description']}")
                st.dataframe(pd.DataFrame([{"Mode": k, "MAPE %": v.get("MAPE_percent"), "MAE MW": v.get("MAE_MW"), "RMSE MW": v.get("RMSE_MW"),
                                            "R²": v.get("R2"), "Naive MAPE %": v["naive"].get("MAPE_percent"), "Naive": v.get("naive_definition"),
                                            "Interval coverage (H2)": (v.get("calibration") or {}).get("coverage"), "Hours": v.get("n")}
                                           for k, v in m.items() if isinstance(v, dict) and "naive" in v]), hide_index=True, use_container_width=True)
            rep = ev.get("research_reported", {}).get(s.model_id)
            if rep:
                st.caption("Reported by the research notebook: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rep.items()))
            info = models()[s.model_id].info()
            st.caption("Artifacts: " + ", ".join(info["artifacts"].values()))
    with st.expander("Reference benchmarks at this location (no learned weather model)"):
        b = ev.get("benchmarks", {}).get(r["request"].location_id, {})
        st.dataframe(pd.DataFrame([{"Benchmark": k, **{x: v.get(x) for x in ("MAPE_percent", "MAE_MW", "R2", "n")}} for k, v in b.items()]),
                     hide_index=True, use_container_width=True)
    with st.expander("Weight evaluation on held-out windows"):
        if WEIGHTS_STUDY_PATH.exists():
            import json
            wsd = json.loads(WEIGHTS_STUDY_PATH.read_text())
            st.markdown(f"{wsd['n_cases']} second-half weekly windows, validation evidence from first-half windows only. "
                        f"Configured weights rank {wsd['configured_rank_among_all_sets']} of {wsd['n_weight_sets']} weight sets by mean regret.")
            st.dataframe(pd.DataFrame([{"Weight set": k, "Mean regret (MAPE points)": v["mean_regret_mape_points"],
                                        "Selection accuracy": v["selection_accuracy"]} for k, v in wsd["results"].items()]), hide_index=True)
            st.caption("With few candidates per location and one family dominating, the study cannot discriminate between weight sets; "
                       "the weights remain documented hypotheses.")
        else:
            st.caption("Run `python -m core.evidence --weights-study` to generate it.")
    with st.expander("All model artifacts discovered in the repository"):
        st.dataframe(pd.DataFrame(discovered_artifacts()), hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------------- analytics
def page_analytics():
    st.title("Energy analytics")
    r = result()
    if not r or r["response"].status != "ok":
        return no_result()
    req, resp = r["request"], r["response"]
    win = next(x for x in resp.scores if x.selected)
    df = resp.predictions
    c = st.columns([2.2, 1, 1, 1.6])
    rng = c[0].slider("Date range", min_value=df.index[0].to_pydatetime(), max_value=df.index[-1].to_pydatetime(),
                      value=(df.index[0].to_pydatetime(), df.index[-1].to_pydatetime()), step=pd.Timedelta(hours=1).to_pytimedelta(),
                      format="YYYY-MM-DD HH:mm")
    agg = c[1].radio("Aggregation", ["hourly", "daily"], index=["hourly", "daily"].index(S.get("aggregation", "hourly")), horizontal=True)
    val = c[2].radio("Value", ["power", "energy"], horizontal=True, format_func=lambda v: "MW" if v == "power" else "MWh")
    names = {s.model_id: s.display_name for s in resp.scores}
    alt_ids = c[3].multiselect("Compare alternatives (computed)", list(resp.alternatives), format_func=lambda m: names[m],
                               help="Only eligible alternatives whose predictions were actually computed are listed.")
    d = df.loc[rng[0]:rng[1]]
    alts = {m: resp.alternatives[m].loc[rng[0]:rng[1]] for m in alt_ids}
    t = st.tabs(["Consumption", "Profiles", "Weather", "Weather vs load", "Baseline vs adjustment", "Observed vs predicted", "Model comparison"])
    with t[0]:
        if agg == "daily":
            daily = aggregate(d, "daily")
            st.plotly_chart(charts.daily_bars(daily), use_container_width=True)
            st.dataframe(daily.round(1), use_container_width=True)
        else:
            st.plotly_chart(charts.timeseries(d, alts, names, val, win.display_name), use_container_width=True)
        st.plotly_chart(charts.cumulative(d), use_container_width=True)
        with st.expander("Table view"):
            st.dataframe(predictions_frame(resp).loc[rng[0]:rng[1]].round(2), use_container_width=True)
    with t[1]:
        a, b = st.columns(2)
        a.plotly_chart(charts.hourly_profile(d), use_container_width=True)
        b.plotly_chart(charts.hour_dow_heatmap(d), use_container_width=True)
    wcols = [v for v in VARIABLES if v in d and v != "load_mw" and d[v].notna().any()]
    with t[2]:
        v = st.selectbox("Weather variable", wcols, format_func=lambda x: f"{VARIABLES[x].label} ({VARIABLES[x].unit})")
        st.plotly_chart(charts.weather_series(d, v, f"{VARIABLES[v].label} ({VARIABLES[v].unit})"), use_container_width=True)
        fig = charts.correlation(d, wcols + ["pred_mw"], {**{k: VARIABLES[k].label for k in VARIABLES}, "pred_mw": "Predicted load"},
                                 "Correlation of weather variables and predicted load (window)")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Correlations need at least two varying variables (a constant scenario has none).")
    with t[3]:
        cols = st.columns(3)
        for col, v in zip(cols, ["temperature_c", "relative_humidity_percent", "dew_point_c"]):
            if v in d:
                col.plotly_chart(charts.scatter_vs(d, v, f"{VARIABLES[v].label} ({VARIABLES[v].unit})"), use_container_width=True)
        if req.scenario:
            st.caption("Scenario simulation: weather is constant, so variation in predicted load comes from the calendar and the model's dynamics.")
    with t[4]:
        if d.baseline_mw.notna().any():
            f1, f2 = charts.decomposition(d)
            st.plotly_chart(f1, use_container_width=True)
            st.plotly_chart(f2, use_container_width=True)
            st.caption("Two-branch combination: log E = log E_baseline + r (TCN). In MW: E = E_baseline + ΔE with ΔE = E_baseline·(eʳ − 1). "
                       "r is the TCN's next-hour residual, learned from weather and residual history.")
        else:
            st.info(f"{win.display_name} predicts load directly; it has no baseline / weather-adjustment decomposition.")
    with t[5]:
        if d.actual_mw.notna().any() and not req.scenario:
            st.plotly_chart(charts.actual_vs_predicted(d), use_container_width=True)
            e = (d.actual_mw - d.pred_mw).dropna()
            st.caption(f"Observed hours: {len(e)} · MAE {e.abs().mean():,.1f} MW · MAPE {(e.abs() / d.actual_mw[e.index]).mean() * 100:.2f} % · bias {e.mean():+,.1f} MW")
        else:
            st.info("No observed load in the timeframe (or a scenario simulation), so no ground-truth comparison is shown.")
    with t[6]:
        if not resp.alternatives:
            st.info("No other eligible model produced predictions for this request.")
        else:
            rows = [{"Model": win.display_name + " (selected)", "Total MWh": d.energy_mwh.sum(), "Mean MW": d.pred_mw.mean(), "Peak MW": d.pred_mw.max()}]
            for m, a in resp.alternatives.items():
                a = a.loc[rng[0]:rng[1]]
                rows.append({"Model": names[m], "Total MWh": a.pred_mw.sum(), "Mean MW": a.pred_mw.mean(), "Peak MW": a.pred_mw.max()})
            st.dataframe(pd.DataFrame(rows).round(1), hide_index=True, use_container_width=True)
            st.plotly_chart(charts.timeseries(d, {m: resp.alternatives[m].loc[rng[0]:rng[1]] for m in resp.alternatives}, names, "power", win.display_name),
                            use_container_width=True, key="cmp")


# ----------------------------------------------------------------------------------- climate
def page_climate():
    st.title("Climate similarity")
    st.caption("Project methodology (from the Atlanta–Birmingham and Las Vegas–Phoenix analyses): per-variable SMD, normalised Wasserstein and KS "
               "similarities (60 %), correlation-structure similarity (20 %) and standardised multivariate similarity (20 %). In model selection the "
               "input weather is compared with each model's training weather for the same months (±1). It carries 25 % of the fitness weight.")
    r = result()
    if r and r["response"].analog:
        a = r["response"].analog
        st.subheader(f"{a['name']} vs the trained cities' recorded climate (same months ±1)")
        tried = {t["city"]: t for t in a["tried"]}
        rows = [{"City": NAMES[x["city"]], "Similarity": x["similarity"], "Class": x["classification"], "Confidence": x["confidence"],
                 "Outcome": ("✅ used · " + tried[x["city"]]["model"]) if tried.get(x["city"], {}).get("status") == "ok"
                 else f"skipped: {tried[x['city']]['reason']}" if x["city"] in tried else "—"} for x in a["ranking"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=100, format="%.1f")})
    if r and r["response"].scores:
        req, resp = r["request"], r["response"]
        st.subheader(f"Input weather vs candidate training climates · {NAMES.get(req.location_id, req.location_id)}")
        rows = []
        for s in resp.scores:
            cl = s.training_domain.get("climate") or {}
            rows.append({"Model": s.display_name, "Training location": ", ".join(NAMES[c] for c in s.training_domain.get("trained_on", [])),
                         "Similarity": cl.get("overall"), "Class": cl.get("classification"), "Method": cl.get("method"),
                         "Confidence": cl.get("confidence"), "Reference": cl.get("reference")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=100, format="%.1f")})
        scored = [s for s in resp.scores if (s.training_domain.get("climate") or {}).get("variables")]
        if scored:
            pick = st.selectbox("Inspect model", [s.model_id for s in scored], format_func=lambda m: next(x.display_name for x in scored if x.model_id == m))
            s = next(x for x in scored if x.model_id == pick)
            cl = s.training_domain["climate"]
            if cl.get("note"):
                st.warning(cl["note"], icon="⚠️")
            st.dataframe(pd.DataFrame([{"Variable": VARIABLES[v].label, **{k: x.get(k) for k in ("score", "SMD", "normalized_Wasserstein", "KS",
                                        "input_mean", "reference_mean", "value", "training_percentile")}} for v, x in cl["variables"].items()]).round(3),
                         hide_index=True, use_container_width=True)
            if cl["method"] == "distribution":
                st.caption(f"Distribution {cl['distribution_score']:.1f} · correlation {cl['correlation_score'] if cl['correlation_score'] is None else round(cl['correlation_score'], 1)} · "
                           f"multivariate {cl['multivariate_score']:.1f} (distance {cl['multivariate_distance']:.2f}) → overall {cl['overall']:.1f}")
            from core.evidence import training_reference
            rc, rs, re_ = training_reference(pick)
            ref = climate.seasonal(climate.reference_weather(rc, str(rs), str(re_)), set(resp.predictions.index.month))
            inp = resp.predictions
            v = st.selectbox("Variable", list(cl["variables"]), format_func=lambda x: VARIABLES[x].label)
            st.plotly_chart(charts.distributions(inp[v].dropna(), ref[v].dropna(), VARIABLES[v].label, f"{NAMES[rc]} training ({sorted(set(ref.index.month))})"),
                            use_container_width=True)
            if cl["method"] == "distribution":
                vv = list(cl["variables"])
                lab = {k: VARIABLES[k].label for k in VARIABLES}
                a, b = st.columns(2)
                f1 = charts.correlation(inp, vv, lab, "Input weather correlations")
                f2 = charts.correlation(ref, vv, lab, f"{NAMES[rc]} training-weather correlations")
                if f1:
                    a.plotly_chart(f1, use_container_width=True)
                if f2:
                    b.plotly_chart(f2, use_container_width=True)
        inp_w = resp.predictions
    else:
        no_result()
        inp_w = None
    st.subheader("Supported cities in one weather feature space")
    sc, pca, pts = climate.pca_space()
    proj = climate.project(inp_w) if inp_w is not None else None
    st.plotly_chart(charts.pca_plot(pts, proj, NAMES, pca.explained_variance_ratio_), use_container_width=True)
    st.caption("StandardScaler + PCA(2) fitted once on a fixed sample of 3,000 recorded hours per city (8 shared variables); input hours are projected "
               "with the same transformation. Two components cannot capture every climate characteristic.")
    with st.spinner("Computing city-to-city similarity…"):
        st.plotly_chart(charts.similarity_matrix(climate.city_matrix(), NAMES), use_container_width=True)


# ----------------------------------------------------------------------------------- data quality
def page_quality():
    st.title("Data quality & processing")
    r = result() or {}
    p = S.get("pending") or {}
    proc = r.get("processed") or p.get("processed")
    resp = r.get("response")
    req = r.get("request") or p.get("request")
    if not (proc or req):
        return no_result("Validate inputs in the Prediction Workspace first.")
    if req is not None and req.input_mode == "manual":
        st.info("Manual entry: one range-checked observation applied as a scenario; no file processing.", icon="✍️")
    if proc:
        q = proc.quality
        name = (r.get("upload_name") or S.get("upload_name") or ("—", 0))
        c = st.columns(5)
        c[0].metric("File", name[0], f"{name[1] / 1e3:,.0f} kB", delta_color="off")
        c[1].metric("Rows uploaded", f"{q.get('rows_uploaded', 0):,}")
        c[2].metric("Hourly rows after processing", f"{q.get('rows_hourly', 0):,}")
        c[3].metric("Duplicate timestamps", q.get("duplicate_timestamps", 0))
        c[4].metric("Missing hours (gaps)", q.get("missing_hours", 0))
        if q.get("columns"):
            t = pd.DataFrame(q["columns"]).T.rename_axis("variable").reset_index()
            t["variable"] = t.variable.map(lambda v: VARIABLES[v].label)
            st.dataframe(t, hide_index=True, use_container_width=True)
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(x=t.variable, y=t.missing, marker=dict(color=charts.BLUE, cornerradius=4), name="missing before correction",
                                   hovertemplate="%{x}: %{y} missing<extra></extra>"))
            st.plotly_chart(charts._style(fig, "Missing values by variable (before forward fill)", "hours", None, 300), use_container_width=True)
        st.markdown("**Feature mapping**")
        st.dataframe(pd.DataFrame([{"Column": c, "Variable": v or "ignored"} for c, v in proc.mapping.items()]), hide_index=True, use_container_width=True)
    issues = (resp.issues if resp else p.get("issues", []))
    log = (resp.processing_log if resp else p.get("log", []))
    st.markdown("**Conditions found**")
    st.dataframe(pd.DataFrame([vars(i) for i in issues]) if issues else pd.DataFrame({"message": ["none"]}), hide_index=True, use_container_width=True)
    st.markdown("**Processing log** (unit conversions, imputation, derived features, context)")
    st.dataframe(pd.DataFrame([vars(e) for e in log]), hide_index=True, use_container_width=True)
    if proc and proc.frame is not None:
        with st.expander("Inspect the processed dataset"):
            st.dataframe(proc.frame.round(3), use_container_width=True, height=320)
        c = st.columns(2)
        c[0].download_button("Download processed data (CSV)", proc.frame.to_csv().encode(), "processed_input.csv", "text/csv")
        raw = r.get("raw_bytes") or S.get("raw_bytes")
        if raw:
            c[1].download_button("Download the original upload (unchanged)", raw, "original_upload.csv", "text/csv")


# ----------------------------------------------------------------------------------- report
def page_report():
    st.title("Prediction report")
    r = result()
    if not r:
        return no_result()
    req, resp = r["request"], r["response"]
    c = st.columns(3)
    if resp.predictions is not None:
        c[0].download_button("⬇ Predictions (CSV)", predictions_csv(resp), f"predictions_{resp.request_id}.csv", "text/csv", type="primary")
    c[1].download_button("⬇ Report (HTML)", report_html(req, resp), f"report_{resp.request_id}.html", "text/html")
    c[2].download_button("⬇ Full response (JSON)", response_json(resp), f"response_{resp.request_id}.json", "application/json")
    st.components.v1.html(report_html(req, resp).decode(), height=1400, scrolling=True)
    with st.expander("Model limitations"):
        for t in LIMITATIONS:
            st.write("•", t)


PAGES = {
    "overview": st.Page(page_overview, title="Overview", icon="📊", url_path="overview", default=True),
    "workspace": st.Page(page_workspace, title="Prediction Workspace", icon="🧭", url_path="workspace"),
    "selection": st.Page(page_selection, title="Model Selection", icon="🏆", url_path="model-selection"),
    "analytics": st.Page(page_analytics, title="Energy Analytics", icon="📈", url_path="analytics"),
    "climate": st.Page(page_climate, title="Climate Similarity", icon="🌦️", url_path="climate"),
    "quality": st.Page(page_quality, title="Data Quality", icon="🧪", url_path="data-quality"),
    "report": st.Page(page_report, title="Prediction Report", icon="📄", url_path="report"),
}

with st.sidebar:
    st.markdown("### ⚡ Energy Intelligence")
    r = result()
    if r:
        resp = r["response"]
        st.caption(f"Last request {resp.request_id}: **{resp.status}**" + (f" · {resp.selected_model} ({resp.selected_fitness:.1f})" if resp.selected_model else ""))
    if st.button("Reset workspace", help="Clear inputs, validation results and the current prediction"):
        reset()
        st.rerun()
    st.caption("Timestamps: local standard time · power MW · energy MWh")

models()
st.navigation(list(PAGES.values())).run()
