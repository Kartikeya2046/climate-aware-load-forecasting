"""Model registry, artifact checks and inference against the real repository artifacts."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from core.data import CITIES, EXP, clean_like_training, repository_record, solar_etr, to_model_frame
from core.models import ArtifactError, DaytonLSTM, TwoBranchTCN, discovered_artifacts, load_keras, registry


def test_registry_discovers_and_loads_real_artifacts(reg):
    assert set(reg) == {"tcn_atlanta", "tcn_las_vegas", "lstm_dayton"}
    for a in reg.values():
        assert a.loaded, a.load_error
        assert not a.check_artifacts()
    found = discovered_artifacts()
    assert any(r["registered"] for r in found) and any(not r["registered"] for r in found)


def _spec(mid):
    from core.data import config
    return dict(next(s for s in config()["models"] if s["id"] == mid))


def test_missing_artifact_excludes_model():
    s = _spec("lstm_dayton")
    s["model"] = "experiments/03_dayton_duquesne/models/does_not_exist.keras"
    a = DaytonLSTM(s)
    assert not a.ensure_loaded() and "missing artifact" in a.load_error


def test_missing_preprocessing_object_excludes_model():
    s = _spec("tcn_las_vegas")
    s["residual_scaler"] = "las_vegas_to_phoenix/missing_scaler.joblib"
    a = TwoBranchTCN(s)
    assert not a.ensure_loaded() and "residual_scaler" in a.load_error


def test_corrupt_artifact_fails_to_load(tmp_path):
    bad = tmp_path / "bad.keras"
    bad.write_bytes(b"this is not a model")
    with pytest.raises(ArtifactError):
        load_keras(bad)


def test_feature_schema_mismatch_is_detected(tmp_path, reg):
    """Copy the real Atlanta result folder and tamper with experiment_config.json only."""
    s = _spec("tcn_atlanta")
    src = EXP.parent / s["result_dir"]
    dst = tmp_path / "res"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.csv", "visualizations"))
    cfg = json.loads((dst / s["experiment_config"]).read_text())
    cfg["weather_features"] = cfg["weather_features"][::-1]
    (dst / s["experiment_config"]).write_text(json.dumps(cfg))
    s["result_dir"] = str(dst)
    a = TwoBranchTCN(s)
    assert not a.ensure_loaded() and "schema" in a.load_error


def test_location_support():
    r = registry()
    assert r["tcn_atlanta"].supports_location("birmingham")[0]
    assert not r["tcn_atlanta"].supports_location("phoenix")[0]
    assert not r["tcn_las_vegas"].supports_location("dayton")[0]
    assert r["lstm_dayton"].supports_location("pittsburgh")[0]


@pytest.mark.parametrize("mid,city,col", [("tcn_atlanta", "birmingham", "predicted_birmingham_mw"),
                                          ("tcn_las_vegas", "phoenix", "predicted_phoenix_mw")])
def test_tcn_reproduces_research_predictions(reg, mid, city, col):
    """Feature transformation, inverse scaling and baseline+adjustment combination reproduce the
    research notebook's saved 2023 predictions (float16 mixed precision there vs float32 here)."""
    a = reg[mid]
    ref = pd.read_csv(a.research_predictions)
    ref["timestamp"] = pd.to_datetime(ref.timestamp.str.strip())
    ref = ref.set_index("timestamp").iloc[:720]
    off = pd.Timedelta(hours=CITIES[city].utc_offset)
    p = a.predict(clean_like_training(repository_record(city)), city, ref.index[0] - off, ref.index[-1] - off, "one_step")
    rel = np.abs(p.frame.pred_mw.to_numpy() - ref[col].to_numpy()) / ref[col].to_numpy()
    assert p.predicted == 720 and rel.max() < 2e-3 and rel.mean() < 2e-4
    base = [c for c in ref.columns if "baseline" in c][0]
    assert np.allclose(p.frame.baseline_mw, ref[base], rtol=1e-3)
    # E = E_baseline + dE with dE = E_baseline (e^r - 1)
    f = p.frame
    assert np.allclose(f.pred_mw, f.baseline_mw + f.adjustment_mw)
    assert np.allclose(f.adjustment_mw, f.baseline_mw * (np.exp(f.residual_log) - 1))


def test_lstm_reproduces_research_predictions_exactly(reg):
    a = reg["lstm_dayton"]
    ref = pd.read_csv(EXP / "03_dayton_duquesne/outputs/DUQ_predictions.csv")
    e = pd.read_csv(EXP / "03_dayton_duquesne/data/processed/FINAL_MERGED_DUQ_LOAD_2006_2016.csv", parse_dates=["datetime"])
    w = pd.read_csv(EXP / "03_dayton_duquesne/data/processed/weather_DUQ_2006-2016.csv", parse_dates=["datetime"])
    m = pd.merge(e.sort_values("datetime"), w.sort_values("datetime"), on="datetime", how="inner")
    lab = to_model_frame(repository_record("pittsburgh"), "pittsburgh")
    k = np.arange(24, 1500)
    contiguous = (m.datetime.iloc[k].to_numpy() - m.datetime.iloc[k - 24].to_numpy()) == np.timedelta64(24, "h")
    y = a.one_step(a.matrix(lab), lab.index.get_indexer(pd.DatetimeIndex(m.datetime.iloc[k])))
    assert np.allclose(y[contiguous], ref.Predicted_DUQ_Load.to_numpy()[k - 24][contiguous], atol=1e-6)


def test_sequence_generation_windows(reg):
    from core.models import _windows
    X = np.arange(10)[:, None].astype(float)
    w = _windows(X, np.array([3, 9]), 3)
    assert w[:, :, 0].tolist() == [[0, 1, 2], [6, 7, 8]]
    a = reg["tcn_atlanta"]
    r = np.arange(400, dtype=float)
    f = a.residual_features_of(r)
    i = 300
    assert f[i].tolist() == [r[i], r[i - 1], r[i - 24], r[i - 168], r[i - 24:i].mean(), pytest.approx(r[i - 24:i].std(ddof=1))]
    blk = a._residual_block(r[None, :], 400)
    assert np.allclose(blk[0], f[400 - 168:400], equal_nan=True)


def test_recursive_first_step_equals_one_step(reg):
    a = reg["tcn_las_vegas"]
    lab = to_model_frame(clean_like_training(repository_record("phoenix")), "phoenix")
    base = a.baseline_log(lab.index, "phoenix")
    r = np.log(lab.load_mw.to_numpy()) - base
    s = lab.index.get_loc(pd.Timestamp("2023-05-01 00:00"))
    one = a.one_step(a.weather_matrix(lab), r, np.array([s]))
    rh = r.copy()
    rh[s:] = np.nan
    rec = a.recursive(a.weather_matrix(lab), rh, np.array([s]), 3)
    assert one[0] == pytest.approx(rec[0, 0], abs=1e-9)
    assert np.isfinite(rec).all()


def test_solar_derivation_matches_recorded_radiation():
    for city in ("las_vegas", "atlanta"):
        rec = repository_record(city).iloc[::97].dropna(subset=["extraterrestrial_horizontal_radiation_wh_m2"])
        sol = solar_etr(rec.index, CITIES[city])
        assert (sol.iloc[:, 0] - rec.extraterrestrial_horizontal_radiation_wh_m2).abs().mean() < 10


def test_timestamp_alignment_round_trip():
    rec = repository_record("birmingham").iloc[1000:1100]
    lab = to_model_frame(rec, "birmingham")
    # load is UTC-labelled in the model frame: local 10:00 load appears at label 16:00 (UTC-6)
    t = rec.index[10]
    assert lab.load_mw[t + pd.Timedelta(hours=6)] == rec.load_mw[t]
    assert lab.temperature_c[t] == rec.temperature_c[t]
