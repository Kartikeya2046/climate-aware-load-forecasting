"""Model registry and inference adapters for the trained research models.

Two families are registered (see config.toml):
* Two-branch TCN: a city baseline MLP (calendar + trend -> log load) plus a weather-residual
  TCN trained on the source city (168 h window -> next-hour log residual).
  log E = log E_baseline + r_weather  =>  E = E_baseline + dE_weather, dE = E_baseline (e^r - 1).
* LSTM (Dayton): 24 h window of [load, 8 weather, hour, dayofweek, month] -> next-hour load.

Both are one-hour-ahead models that read observed load history. `predict` supports
`one_step` (actual load known for every preceding hour, the evaluated research setting) and
`forecast` (recursive: each prediction is fed back as history for the next hour).
All arrays below are in the training label space (see data.to_model_frame).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

from .data import CITIES, REPO_ROOT, config, label_to_local, to_model_frame

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASELINE_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "years_since_start"]
TREND_ORIGIN = pd.Timestamp("2016-01-01 00:00:00")  # first label of every 04/05 training frame (years_since_start = 0)
TRAIN_END = pd.Timestamp("2021-12-31 23:00:00")      # 04/05 chronological split (labels)
VALID_END = pd.Timestamp("2022-12-31 23:00:00")
DERIVED = {"wind_dir_sin", "wind_dir_cos", "cooling_degree_c", "heating_degree_c", "temp_x_humidity"}
LSTM_WEATHER = {"Temp_C": "temperature_c", "Dew_Point_C": "dew_point_c", "RH_percent": "relative_humidity_percent",
                "Pressure_kPa": "station_pressure_kpa", "Wind_Dir_deg": "wind_direction_deg",
                "Wind_Speed_mps": "wind_speed_ms", "Visibility_km": "horizontal_visibility_km", "Precip_cm": "hourly_precipitation_cm"}


class ArtifactError(RuntimeError):
    pass


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_keras(path: Path):
    """Load a .keras artifact. Zip archives (Keras 3) load directly. HDF5 files written by
    TF 2.10 under the mixed_float16 policy are loaded from a temporary copy whose config
    replaces the dtype policy with float32 (weights are float32 either way); the repository
    file is never modified. Outputs agree with TF 2.10 to ~1e-3 (float16 rounding)."""
    import keras

    class SpatialDropout1D(keras.layers.SpatialDropout1D):  # Keras 2 config carries extra keys
        def __init__(self, rate, trainable=True, noise_shape=None, **kw):
            super().__init__(rate, **kw)

    with open(path, "rb") as f:
        magic = f.read(4)
    if magic.startswith(b"PK"):
        return keras.models.load_model(path, compile=False)
    if not magic.startswith(b"\x89HDF"):
        raise ArtifactError(f"{path.name}: not a Keras zip or HDF5 file")

    def float32(o):
        if isinstance(o, dict):
            return {k: ("float32" if k == "dtype" and isinstance(v, dict) else float32(v)) for k, v in o.items()}
        return [float32(x) for x in o] if isinstance(o, list) else o

    tmpdir = tempfile.mkdtemp()
    try:
        tmp = Path(tmpdir) / "model.h5"
        shutil.copy(path, tmp)
        with h5py.File(tmp, "r+") as f:
            f.attrs["model_config"] = json.dumps(float32(json.loads(f.attrs["model_config"])))
        return keras.models.load_model(tmp, compile=False, custom_objects={"SpatialDropout1D": SpatialDropout1D})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def load_scaler(path: Path):
    with warnings.catch_warnings():  # pickled with scikit-learn 1.7.2; attributes are version-stable
        warnings.simplefilter("ignore")
        return joblib.load(path)


def _windows(X: np.ndarray, ends: np.ndarray, lookback: int) -> np.ndarray:
    """Stack X[p-lookback:p] for every p in ends -> (n, lookback, features)."""
    idx = ends[:, None] + np.arange(-lookback, 0)[None, :]
    return X[idx]


def _predict(model, x: np.ndarray, batch: int = 1024) -> np.ndarray:
    with warnings.catch_warnings():  # Keras-2 functional configs declare a one-element input list
        warnings.filterwarnings("ignore", message="The structure of `inputs`")
        out = [np.asarray(model(x[i:i + batch], training=False)).ravel() for i in range(0, len(x), batch)]
    return np.concatenate(out) if out else np.empty(0)


@dataclass
class Prediction:
    """Hourly result in local standard time."""
    frame: pd.DataFrame          # pred_mw, baseline_mw, adjustment_mw, actual_mw, residual_log
    requested: int
    predicted: int
    notes: list[str] = field(default_factory=list)


class Adapter:
    id: str
    family: str
    display_name: str
    version: str
    lookback: int
    history_hours: int           # hours of observed load needed before the first prediction
    weather_inputs: list[str]    # canonical weather variables the model needs
    source_city: str
    trained_on: list[str]
    evaluated_on: list[str]
    artifacts: dict[str, Path]
    decomposes: bool             # provides baseline / weather adjustment

    def __init__(self, spec: dict):
        self.spec = spec
        self.id, self.family, self.display_name, self.version = spec["id"], spec["family"], spec["display_name"], spec["version"]
        self.source_city = spec["source_city"]
        self.loaded = False
        self.load_error: str | None = None

    # -- registry -------------------------------------------------------------
    def check_artifacts(self) -> list[str]:
        return [f"missing artifact: {k} ({p.relative_to(REPO_ROOT)})" for k, p in self.artifacts.items() if not p.exists()]

    def ensure_loaded(self) -> bool:
        if self.loaded:
            return True
        if self.load_error:
            return False
        missing = self.check_artifacts()
        if missing:
            self.load_error = "; ".join(missing)
            return False
        try:
            self._load()
            self._health_check()
            self.loaded = True
        except Exception as exc:  # noqa: BLE001 - any failure excludes the model, with the reason shown
            self.load_error = f"{type(exc).__name__}: {exc}"
        return self.loaded

    def info(self) -> dict:
        return {"id": self.id, "family": self.family, "display_name": self.display_name, "version": self.version,
                "source_city": self.source_city, "trained_on": self.trained_on, "evaluated_on": self.evaluated_on,
                "lookback_hours": self.lookback, "history_hours": self.history_hours, "weather_inputs": self.weather_inputs,
                "target": "next-hour average load (MW); one-hour intervals so MW x 1 h = MWh",
                "artifacts": {k: str(p.relative_to(REPO_ROOT)).replace("\\", "/") for k, p in self.artifacts.items()},
                "loaded": self.loaded, "load_error": self.load_error}

    def supports_location(self, city: str) -> tuple[bool, str]:
        raise NotImplementedError

    # -- inference ------------------------------------------------------------
    def predict(self, frame: pd.DataFrame, city: str, start: pd.Timestamp, end: pd.Timestamp,
                mode: str = "forecast", history: str = "observed") -> Prediction:
        """frame: canonical local-standard-time hourly frame covering context + window.
        history='observed' uses load_mw where present; 'baseline_assumption' (TCN only) treats
        unknown pre-start residual history as 0, i.e. load equal to the calendar baseline."""
        lab = to_model_frame(frame, city)
        off = pd.Timedelta(hours=CITIES[city].utc_offset)
        targets = pd.date_range(start + off, end + off, freq="h")
        res = self._predict_labels(lab, city, targets, mode, history)
        res.index = label_to_local(res.index, city)
        res.index.name = "timestamp"
        n_ok = int(res.pred_mw.notna().sum())
        notes = []
        if n_ok < len(res):
            notes.append(f"{len(res) - n_ok} of {len(res)} requested hours could not be predicted (missing weather or load history in the model window).")
        return Prediction(res, len(res), n_ok, notes)


class TwoBranchTCN(Adapter):
    decomposes = True
    lookback = 168
    history_hours = 168 + 168 + 24  # window + lag_168h of its first row + 24 h rolling statistics

    def __init__(self, spec: dict):
        super().__init__(spec)
        rd = REPO_ROOT / spec["result_dir"]
        self.artifacts = {"tcn": rd / spec["tcn"], "feature_scaler": rd / spec["feature_scaler"],
                          "residual_scaler": rd / spec["residual_scaler"], "experiment_config": rd / spec["experiment_config"]}
        for city, sub in spec["baselines"].items():
            for k in ("baseline_mlp.keras", "baseline_feature_scaler.joblib", "baseline_target_scaler.joblib"):
                self.artifacts[f"{city}/{k}"] = rd / sub / k
        self.research_metrics = rd / spec["research_metrics"]
        self.research_predictions = rd / spec["research_predictions"]
        cfg = json.loads(self.artifacts["experiment_config"].read_text()) if self.artifacts["experiment_config"].exists() else {}
        self.experiment_config = cfg
        self.features = cfg.get("tcn_features") or (cfg.get("weather_features", []) + cfg.get("residual_features", []))
        self.residual_features = [f for f in self.features if "resid" in f]
        self.weather_features = [f for f in self.features if f not in self.residual_features]
        raw = [f for f in self.weather_features if f not in DERIVED]
        if {"wind_dir_sin", "wind_dir_cos"} & set(self.weather_features):
            raw.append("wind_direction_deg")
        if {"cooling_degree_c", "heating_degree_c", "temp_x_humidity"} & set(self.weather_features):
            raw += [c for c in ("temperature_c", "relative_humidity_percent") if c not in raw]
        self.weather_inputs = raw
        self.trained_on = [spec["source_city"]]
        self.evaluated_on = [cfg["target_city"]] if cfg.get("target_city") else []
        self.baseline_cities = list(spec["baselines"])

    def supports_location(self, city):
        if city in self.baseline_cities:
            return True, "city baseline model available"
        return False, f"no trained city-baseline MLP for {CITIES[city].name if city in CITIES else city} (baselines exist for {', '.join(CITIES[c].name for c in self.baseline_cities)})"

    def _load(self):
        self.tcn = load_keras(self.artifacts["tcn"])
        self.x_scaler = load_scaler(self.artifacts["feature_scaler"])
        self.r_scaler = load_scaler(self.artifacts["residual_scaler"])
        if list(self.x_scaler.feature_names_in_) != self.features:
            raise ArtifactError("TCN feature scaler schema does not match experiment_config.json")
        if tuple(self.tcn.input_shape[1:]) != (self.lookback, len(self.features)):
            raise ArtifactError(f"TCN input shape {self.tcn.input_shape} does not match {self.lookback}x{len(self.features)}")
        self.baselines = {}
        for city in self.baseline_cities:
            m = load_keras(self.artifacts[f"{city}/baseline_mlp.keras"])
            xs = load_scaler(self.artifacts[f"{city}/baseline_feature_scaler.joblib"])
            ys = load_scaler(self.artifacts[f"{city}/baseline_target_scaler.joblib"])
            if list(xs.feature_names_in_) != BASELINE_FEATURES:
                raise ArtifactError(f"{city} baseline scaler schema mismatch")
            self.baselines[city] = (m, xs, ys)

    def _health_check(self):
        y = _predict(self.tcn, np.zeros((1, self.lookback, len(self.features)), "float32"))
        for m, _, _ in self.baselines.values():
            y = np.append(y, _predict(m, np.zeros((1, len(BASELINE_FEATURES)), "float32")))
        if not np.isfinite(y).all():
            raise ArtifactError("non-finite output on health-check input")

    # -- features (identical to the training notebooks) ------------------------
    @staticmethod
    def calendar(labels: pd.DatetimeIndex) -> pd.DataFrame:
        t = labels
        return pd.DataFrame({
            "hour_sin": np.sin(2 * np.pi * t.hour / 24), "hour_cos": np.cos(2 * np.pi * t.hour / 24),
            "dow_sin": np.sin(2 * np.pi * t.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * t.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * (t.month - 1) / 12), "month_cos": np.cos(2 * np.pi * (t.month - 1) / 12),
            "is_weekend": (t.dayofweek >= 5).astype("float32"),
            "years_since_start": (t - TREND_ORIGIN).total_seconds() / (365.25 * 24 * 3600),
        }, index=labels)

    def baseline_log(self, labels: pd.DatetimeIndex, city: str) -> np.ndarray:
        m, xs, ys = self.baselines[city]
        x = xs.transform(self.calendar(labels)[BASELINE_FEATURES]).astype("float32")
        return ys.inverse_transform(_predict(m, x).reshape(-1, 1).astype("float64")).ravel()

    def weather_matrix(self, lab: pd.DataFrame) -> np.ndarray:
        w = lab.copy()
        if "wind_direction_deg" in w:
            rad = np.deg2rad(w["wind_direction_deg"])
            w["wind_dir_sin"], w["wind_dir_cos"] = np.sin(rad), np.cos(rad)
        w["cooling_degree_c"] = np.clip(w["temperature_c"] - 18, 0, None)
        w["heating_degree_c"] = np.clip(18 - w["temperature_c"], 0, None)
        w["temp_x_humidity"] = w["temperature_c"] * w["relative_humidity_percent"]
        return w.reindex(columns=self.weather_features).to_numpy("float64")

    @staticmethod
    def residual_features_of(r: np.ndarray) -> np.ndarray:
        """lag_0h, lag_1h, lag_24h, lag_168h, mean_24h, std_24h (rolling over the 24 hours before each row)."""
        s = pd.Series(r)
        prev = s.shift(1).rolling(24, min_periods=24)
        return np.column_stack([s, s.shift(1), s.shift(24), s.shift(168), prev.mean(), prev.std()])

    def _predict_labels(self, lab, city, targets, mode, history):
        lab = lab.reindex(lab.index.union(targets).union(pd.date_range(targets[0] - pd.Timedelta(hours=self.history_hours), targets[0], freq="h")))
        lab = lab.asfreq("h")
        pos = lab.index.get_indexer(targets)
        base = self.baseline_log(lab.index, city)
        r = np.log(lab["load_mw"].where(lab["load_mw"] > 0).to_numpy("float64")) - base
        W = self.weather_matrix(lab)
        if mode == "one_step":
            r_hat = self.one_step(W, r, pos)
        else:
            first = pos[0]
            r_hist = r.copy()
            r_hist[first:] = np.nan
            if history == "baseline_assumption":
                r_hist[:first] = np.nan_to_num(r_hist[:first], nan=0.0)  # unknown history = load at baseline
            r_hat = self.recursive(W, r_hist, np.array([first]), len(pos))[0]
        out = pd.DataFrame(index=targets)
        out["residual_log"] = r_hat
        out["baseline_mw"] = np.exp(base[pos])
        out["pred_mw"] = np.exp(base[pos] + r_hat)
        out["adjustment_mw"] = out.pred_mw - out.baseline_mw
        out["actual_mw"] = lab["load_mw"].to_numpy()[pos]
        return out

    def one_step(self, W: np.ndarray, r: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Next-hour residual for every label position in `ends` using observed history."""
        X = np.column_stack([W, self.residual_features_of(r)])
        ok = (ends >= self.lookback)
        ok[ok] = np.isfinite(_windows(X, ends[ok], self.lookback)).all(axis=(1, 2))
        out = np.full(len(ends), np.nan)
        if ok.any():
            xs = self._scale(_windows(X, ends[ok], self.lookback))
            out[ok] = self.r_scaler.inverse_transform(_predict(self.tcn, xs).reshape(-1, 1).astype("float64")).ravel()
        return out

    def _residual_block(self, R: np.ndarray, p: int) -> np.ndarray:
        """residual_features_of for rows p-lookback..p-1 of every series in R (n, T), vectorised."""
        L = self.lookback
        i = np.arange(p - L, p)
        prev = np.lib.stride_tricks.sliding_window_view(R[:, p - L - 24:p - 1], 24, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.stack([R[:, i], R[:, i - 1], R[:, i - 24], R[:, i - 168], prev.mean(-1), prev.std(-1, ddof=1)], axis=2)

    def _scale(self, win: np.ndarray) -> np.ndarray:
        n, L, F = win.shape
        return ((win.reshape(-1, F) - self.x_scaler.mean_) / self.x_scaler.scale_).reshape(n, L, F).astype("float32")

    def recursive(self, W: np.ndarray, r: np.ndarray, starts: np.ndarray, horizon: int) -> np.ndarray:
        """Free-running forecast of `horizon` hours from each start (batched over starts).
        r must hold the history available at forecast time (NaN from each start onward is
        expected); each predicted residual is fed back for the following hours."""
        n = len(starts)
        ctx = self.history_hours
        out = np.full((n, horizon), np.nan)
        if n == 0:
            return out
        R = np.full((n, ctx + horizon), np.nan)
        Wb = np.full((n, ctx + horizon, W.shape[1]), np.nan)
        for k, s in enumerate(starts):
            lo = s - ctx
            src = slice(max(lo, 0), min(s + horizon, len(r)))
            dst = slice(src.start - lo, src.stop - lo)
            R[k, dst] = r[src]
            R[k, ctx:] = np.nan
            Wb[k, dst] = W[src]
        alive = np.ones(n, bool)
        for h in range(horizon):
            p = ctx + h
            win = np.concatenate([Wb[:, p - self.lookback:p], self._residual_block(R, p)], axis=2)
            alive &= np.isfinite(win).all(axis=(1, 2))
            if not alive.any():
                break
            y = np.full(n, np.nan)
            y[alive] = self.r_scaler.inverse_transform(_predict(self.tcn, self._scale(win[alive])).reshape(-1, 1).astype("float64")).ravel()
            R[:, p] = y
            out[:, h] = y
        return out


class DaytonLSTM(Adapter):
    decomposes = False
    lookback = 24
    history_hours = 24

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.artifacts = {"model": REPO_ROOT / spec["model"], "scaler": REPO_ROOT / spec["scaler"]}
        self.weather_inputs = list(LSTM_WEATHER.values())
        self.trained_on = [spec["source_city"]]
        self.evaluated_on = list(spec.get("evaluated_on", []))
        self.research_metrics = None

    def supports_location(self, city):
        if city in self.trained_on + self.evaluated_on:
            return True, "within the research domain (trained or evaluated here)"
        return True, "technically applicable (needs only weather + load history); not validated here in the research"

    def _load(self):
        self.model = load_keras(self.artifacts["model"])
        self.scaler = load_scaler(self.artifacts["scaler"])
        names = list(self.scaler.feature_names_in_)
        expected = ["mw", *LSTM_WEATHER, "hour", "dayofweek", "month"]
        if names != expected:
            raise ArtifactError(f"LSTM scaler schema {names} != {expected}")
        if tuple(self.model.input_shape[1:]) != (self.lookback, len(expected)):
            raise ArtifactError(f"LSTM input shape {self.model.input_shape} mismatch")

    def _health_check(self):
        if not np.isfinite(_predict(self.model, np.zeros((1, self.lookback, 12), "float32"))).all():
            raise ArtifactError("non-finite output on health-check input")

    def matrix(self, lab: pd.DataFrame) -> np.ndarray:
        t = lab.index
        cols = [lab["load_mw"].to_numpy("float64")] + [lab[c].to_numpy("float64") if c in lab else np.full(len(lab), np.nan)
                                                       for c in LSTM_WEATHER.values()]
        cols += [t.hour.to_numpy("float64"), t.dayofweek.to_numpy("float64"), t.month.to_numpy("float64")]
        return np.column_stack(cols)

    def scale(self, X):
        return X * self.scaler.scale_ + self.scaler.min_

    def unscale_load(self, y):
        return (y - self.scaler.min_[0]) / self.scaler.scale_[0]

    def one_step(self, X: np.ndarray, ends: np.ndarray) -> np.ndarray:
        ok = ends >= self.lookback
        ok[ok] = np.isfinite(_windows(X, ends[ok], self.lookback)).all(axis=(1, 2))
        out = np.full(len(ends), np.nan)
        if ok.any():
            out[ok] = self.unscale_load(_predict(self.model, self.scale(_windows(X, ends[ok], self.lookback)).astype("float32")))
        return out

    def recursive(self, X: np.ndarray, starts: np.ndarray, horizon: int) -> np.ndarray:
        n, L = len(starts), self.lookback
        out = np.full((n, horizon), np.nan)
        B = np.full((n, L + horizon, X.shape[1]), np.nan)
        for k, s in enumerate(starts):
            src = slice(max(s - L, 0), min(s + horizon, len(X)))
            B[k, src.start - (s - L):src.stop - (s - L)] = X[src]
        B[:, L:, 0] = np.nan  # future load unknown
        alive = np.ones(n, bool)
        for h in range(horizon):
            win = B[:, h:h + L]
            alive &= np.isfinite(win).all(axis=(1, 2))
            if not alive.any():
                break
            y = np.full(n, np.nan)
            y[alive] = self.unscale_load(_predict(self.model, self.scale(win[alive]).astype("float32")))
            B[:, L + h, 0] = y
            out[:, h] = y
        return out

    def _predict_labels(self, lab, city, targets, mode, history):
        lab = lab.reindex(lab.index.union(targets).union(pd.date_range(targets[0] - pd.Timedelta(hours=self.lookback), targets[0], freq="h"))).asfreq("h")
        pos = lab.index.get_indexer(targets)
        X = self.matrix(lab)
        if mode == "one_step":
            y = self.one_step(X, pos)
        else:
            Xh = X.copy()
            Xh[pos[0]:, 0] = np.nan
            y = self.recursive(Xh, np.array([pos[0]]), len(pos))[0]
        out = pd.DataFrame(index=targets)
        out["pred_mw"] = y
        out["baseline_mw"] = np.nan
        out["adjustment_mw"] = np.nan
        out["residual_log"] = np.nan
        out["actual_mw"] = lab["load_mw"].to_numpy()[pos]
        return out


@lru_cache(maxsize=1)
def registry() -> dict[str, Adapter]:
    kinds = {"Two-branch TCN": TwoBranchTCN, "LSTM": DaytonLSTM}
    return {s["id"]: kinds[s["family"]](s) for s in config()["models"]}


def discovered_artifacts() -> list[dict]:
    """Every *.keras file in experiments/ with its registration status (transparency only)."""
    used = {p.resolve() for a in registry().values() for p in a.artifacts.values()}
    rows = []
    for p in sorted((REPO_ROOT / "experiments").rglob("*.keras")):
        if ".venv" in p.parts:
            continue
        rows.append({"artifact": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
                     "registered": p.resolve() in used})
    return rows
