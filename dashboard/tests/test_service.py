"""Service ("API") integration with real artifacts and recorded data, aggregation and exports."""
import json

import numpy as np
import pandas as pd
import pytest

from core.data import repository_record
from core.report import predictions_csv, report_html, response_json
from core.schema import PredictionRequest
from core.service import aggregate, run_analog, run_prediction, summarize
from core.validation import manual_frame


def req(city, s, e, mode="one_step", ctx="provided", with_load=True, hist=400):
    s, e = pd.Timestamp(s), pd.Timestamp(e)
    w = repository_record(city).loc[s - pd.Timedelta(hours=hist):e + pd.Timedelta(hours=10)]
    if not with_load:
        w = w.drop(columns="load_mw")
    return PredictionRequest(city, s, e, "csv", w, context_source=ctx, prediction_mode=mode, quality={"rows_hourly": len(w)})


@pytest.mark.parametrize("city,expected", [("birmingham", "tcn_atlanta"), ("atlanta", "tcn_atlanta"), ("phoenix", "tcn_las_vegas"),
                                           ("las_vegas", "tcn_las_vegas"), ("dayton", "lstm_dayton")])
def test_selection_from_evidence(reg, city, expected):
    s = "2015-07-06" if city == "dayton" else "2023-07-10"
    r = run_prediction(req(city, s, pd.Timestamp(s) + pd.Timedelta(hours=47)))
    assert r.status == "ok" and r.selected_model == expected
    assert sum(x.selected for x in r.scores) == 1
    assert all(x.rank is None for x in r.scores if x.eligibility == "ineligible")
    p = r.predictions
    assert len(p) == 48 and np.isfinite(p.pred_mw).all() and (p.pred_mw > 0).all()
    assert r.summary["observed_MAPE_percent"] < 10


def test_energy_aggregation_units(reg):
    r = run_prediction(req("phoenix", "2023-07-10", "2023-07-12 23:00"))
    p = r.predictions
    assert np.allclose(p.energy_mwh, p.pred_mw * 1.0)  # 1-hour intervals: MWh = MW x 1 h
    assert r.summary["total_energy_mwh"] == pytest.approx(p.energy_mwh.sum())
    assert r.summary["average_power_mw"] == pytest.approx(p.pred_mw.mean())
    d = aggregate(p, "daily")
    assert len(d) == 3 and d.energy_mwh.sum() == pytest.approx(p.energy_mwh.sum()) and (d.hours == 24).all()
    lo, hi = r.summary["total_energy_interval_mwh"]
    assert lo < r.summary["total_energy_mwh"] < hi
    assert (p.lower_mw <= p.pred_mw).all() and (p.pred_mw <= p.upper_mw).all()


def test_summary_peak_and_partial_coverage():
    df = pd.DataFrame({"pred_mw": [1.0, 3.0, np.nan], "energy_mwh": [1.0, 3.0, np.nan]},
                      index=pd.date_range("2023-01-01", periods=3, freq="h"))  # SYNTHETIC values
    s = summarize(df, None, "forecast")
    assert s["total_energy_mwh"] == 4 and s["peak_power_mw"] == 3 and s["hours_predicted"] == 2 and s["hours_requested"] == 3
    assert "total_energy_interval_mwh" not in s  # never invented for incomplete coverage


def test_forecast_with_recorded_context_and_lead_assumption(reg):
    r = run_prediction(req("birmingham", "2023-03-01", "2023-03-01 23:00", mode="forecast", ctx="repository", with_load=False, hist=0))
    assert r.status == "ok" and r.selected_model == "tcn_atlanta"
    assert r.summary["observed_MAPE_percent"] is not None  # recorded load kept as ground truth


def test_one_step_without_observed_load_is_ineligible(reg):
    r = run_prediction(req("phoenix", "2023-07-10", "2023-07-10 23:00", with_load=False))
    assert r.status == "no_eligible_model"
    assert all(s.eligibility == "ineligible" for s in r.scores)


def test_manual_scenario_without_history(reg):
    s, e = pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-01 23:00")
    f, issues, log = manual_frame("phoenix", s, e, {v: x for v, x in {
        "temperature_c": 40, "dew_point_c": 5, "relative_humidity_percent": 12, "station_pressure_kpa": 96.5, "wind_speed_ms": 3,
        "wind_direction_deg": 250, "horizontal_visibility_km": 16, "precipitable_water_mm": 15, "hourly_precipitation_cm": 0,
        "total_sky_cover_tenths": 1, "opaque_sky_cover_tenths": 0}.items()}, 400)
    r = run_prediction(PredictionRequest("phoenix", s, e, "manual", f, context_source="none", prediction_mode="forecast",
                                         scenario=True, issues=issues, processing_log=log))
    assert r.status == "ok" and r.selected_model == "tcn_las_vegas"
    win = next(x for x in r.scores if x.selected)
    assert win.eligibility == "conditional" and win.training_domain["history_policy"] == "baseline_assumption"
    clim = next(c for c in win.components if c.name == "climate_similarity")
    assert clim.confidence == "low"  # point proximity, not a distribution score
    lstm = next(x for x in r.scores if x.model_id == "lstm_dayton")
    assert lstm.eligibility == "ineligible"
    assert "observed_MAPE_percent" not in r.summary


def test_unsupported_location_refused():
    f = repository_record("phoenix").iloc[:48]
    r = run_prediction(PredictionRequest("tucson", f.index[24], f.index[30], "csv", f))
    assert r.status == "unsupported_location" and r.predictions is None


@pytest.mark.parametrize("city,model", [("phoenix", "tcn_las_vegas"), ("birmingham", "tcn_atlanta")])
def test_unknown_city_uses_most_similar_climate(reg, city, model):
    w = repository_record(city).loc["2023-01-01":"2023-01-20"].drop(columns="load_mw")  # weather only, no load
    q = PredictionRequest("auto", pd.Timestamp("2023-01-12"), pd.Timestamp("2023-01-13 23:00"), "csv", w)
    rc, r = run_analog(q, "Somewhere")
    sims = [x["similarity"] for x in r.analog["ranking"]]
    assert sims == sorted(sims, reverse=True)
    assert r.status == "ok" and rc.location_id == r.analog["city"] and r.selected_model == model
    assert r.analog["tried"][-1]["status"] == "ok" and any(i.code == "analog" for i in r.issues)


def test_unknown_city_without_eligible_model_reports_most_similar(reg):
    w = repository_record("birmingham").loc["2023-07-01":"2023-07-14"].drop(columns="load_mw")
    rc, r = run_analog(PredictionRequest("auto", pd.Timestamp("2023-07-10"), pd.Timestamp("2023-07-10 23:00"), "csv", w,
                                         prediction_mode="one_step"))  # one-step needs load: nothing is eligible
    assert r.status == "no_eligible_model" and len(r.analog["tried"]) == 6
    assert r.analog["city"] == rc.location_id == r.analog["ranking"][0]["city"]


def test_exports(reg):
    q = req("las_vegas", "2023-07-10", "2023-07-10 23:00")
    r = run_prediction(q)
    csv = pd.read_csv(pd.io.common.BytesIO(predictions_csv(r)))
    assert {"timestamp_local_standard_time", "predicted_avg_load_mw", "predicted_energy_mwh"} <= set(csv.columns) and len(csv) == 24
    js = json.loads(response_json(r))
    assert js["selected_model"] == "tcn_las_vegas" and len(js["predictions"]) == 24
    html = report_html(q, r).decode()
    assert "Model Fitness Score" in html and "not a probability" in html
