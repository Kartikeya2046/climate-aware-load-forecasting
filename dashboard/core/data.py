"""Configuration, supported locations, repository datasets and shared feature helpers.

Timestamp convention (verified from the diurnal cycles of every research dataset):
the load series are labelled in UTC (load minimum at 08-13 h = 03-05 h local) while the
weather series are labelled in local standard time (temperature minimum ~05 h, maximum
~15 h). The research pipelines joined the two on the raw label, so every trained model saw
`load(UTC t)` next to `weather(local t)`. This module keeps data in true local standard time
and `to_model_frame` re-creates the training alignment exactly for inference.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import VARIABLES

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DASHBOARD_DIR.parent
EXP = REPO_ROOT / "experiments"


@lru_cache(maxsize=1)
def config() -> dict:
    with open(DASHBOARD_DIR / "config.toml", "rb") as f:
        return tomllib.load(f)


@dataclass(frozen=True)
class City:
    id: str
    name: str
    utility: str
    utc_offset: int  # hours behind UTC, standard time
    lat: float
    lon: float
    pair: str


CITIES: dict[str, City] = {c.id: c for c in [
    City("dayton", "Dayton, Ohio", "PJM DAY zone", 5, 39.90, -84.22, "Dayton ↔ Pittsburgh"),
    City("pittsburgh", "Pittsburgh, Pennsylvania", "PJM DUQ zone (Duquesne Light)", 5, 40.49, -80.23, "Dayton ↔ Pittsburgh"),
    City("atlanta", "Atlanta, Georgia", "Southern Company (Georgia)", 5, 33.64, -84.43, "Atlanta ↔ Birmingham"),
    City("birmingham", "Birmingham, Alabama", "Southern Company (Alabama)", 6, 33.56, -86.75, "Atlanta ↔ Birmingham"),
    City("las_vegas", "Las Vegas, Nevada", "Nevada Power", 8, 36.08, -115.15, "Las Vegas ↔ Phoenix"),
    City("phoenix", "Phoenix, Arizona", "Arizona Public Service", 7, 33.43, -112.02, "Las Vegas ↔ Phoenix"),
]}

_DAYTON_COLS = {"Temp_C": "temperature_c", "Dew_Point_C": "dew_point_c", "RH_percent": "relative_humidity_percent",
                "Pressure_kPa": "station_pressure_kpa", "Wind_Dir_deg": "wind_direction_deg",
                "Wind_Speed_mps": "wind_speed_ms", "Visibility_km": "horizontal_visibility_km",
                "Precip_cm": "hourly_precipitation_cm"}

_SOURCES = {
    "dayton": ("03_dayton_duquesne/data/processed/FINAL_MERGED_DAY_LOAD_2006_2016.csv",
               "03_dayton_duquesne/data/processed/weather_DAY_2006-2016.csv"),
    "pittsburgh": ("03_dayton_duquesne/data/processed/FINAL_MERGED_DUQ_LOAD_2006_2016.csv",
                   "03_dayton_duquesne/data/processed/weather_DUQ_2006-2016.csv"),
    "atlanta": ("04_atlanta_birmingham/LSTM_GPU_TRAINING/LSTM_DATA_PREP/data/atlanta_master.csv",) * 2,
    "birmingham": ("04_atlanta_birmingham/LSTM_GPU_TRAINING/LSTM_DATA_PREP/data/birmingham_master.csv",) * 2,
    "las_vegas": ("05_las_vegas_phoenix/data/las_vegas_hourly_energy.csv",
                  "05_las_vegas_phoenix/data/las_vegas_hourly_weather_2016_2023.csv"),
    "phoenix": ("05_las_vegas_phoenix/data/phoenix_hourly_energy.csv",
                "05_las_vegas_phoenix/data/phoenix_hourly_weather_2016_2023.csv"),
}


def source_files(city: str) -> list[Path]:
    return sorted({EXP / p for p in _SOURCES[city]})


@lru_cache(maxsize=None)
def repository_record(city: str) -> pd.DataFrame:
    """The city's research dataset in the canonical schema, local standard time, hourly index.
    Values are as recorded (no cleaning); missing hours are NaN rows."""
    energy_path, weather_path = (EXP / p for p in _SOURCES[city])
    if city in ("dayton", "pittsburgh"):
        e = pd.read_csv(energy_path, parse_dates=["datetime"]).rename(columns={"datetime": "t", "mw": "load_mw"})
        w = pd.read_csv(weather_path, parse_dates=["datetime"]).rename(columns={"datetime": "t", **_DAYTON_COLS})
        e["t"], w["t"] = e.t.dt.round("h"), w.t.dt.round("h")  # as in the training notebook
    else:
        e = pd.read_csv(energy_path, parse_dates=["timestamp"]).rename(columns={"timestamp": "t", "hourly_load_mw": "load_mw"})
        w = pd.read_csv(weather_path, parse_dates=["timestamp"]).rename(columns={"timestamp": "t"})
    e = e[["t", "load_mw"]].drop_duplicates("t").set_index("t").sort_index()
    w = w[["t"] + [c for c in VARIABLES if c in w.columns and c != "load_mw"]].drop_duplicates("t").set_index("t").sort_index()
    e.index = e.index - pd.Timedelta(hours=CITIES[city].utc_offset)  # UTC label -> local standard time
    idx = pd.date_range(min(e.index.min(), w.index.min()), max(e.index.max(), w.index.max()), freq="h")
    out = w.reindex(idx).join(e.reindex(idx))
    out.index.name = "timestamp"
    return out.apply(pd.to_numeric, errors="coerce")


def record_span(city: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    r = repository_record(city)
    both = r.load_mw.notna() & r.temperature_c.notna()
    return r.index[both][0], r.index[both][-1]


def to_model_frame(frame: pd.DataFrame, city: str) -> pd.DataFrame:
    """Local-standard-time frame -> the label space the models were trained in:
    row label t carries weather observed at local t and load observed at UTC t (local t - offset)."""
    off = pd.Timedelta(hours=CITIES[city].utc_offset)
    out = frame.drop(columns="load_mw", errors="ignore").copy()
    load = frame["load_mw"] if "load_mw" in frame else pd.Series(np.nan, index=frame.index)
    shifted = load.copy()
    shifted.index = shifted.index + off
    full = out.index.union(shifted.index)
    out = out.reindex(full)
    out["load_mw"] = shifted.reindex(full)
    return out.asfreq("h")


def label_to_local(index: pd.DatetimeIndex, city: str) -> pd.DatetimeIndex:
    """Model label of a load value -> the local-standard hour that load belongs to."""
    return index - pd.Timedelta(hours=CITIES[city].utc_offset)


def clean_like_training(frame: pd.DataFrame) -> pd.DataFrame:
    """Training-pipeline cleaning for repository data: out-of-range weather -> NaN, then
    past-only forward fill; load must be positive and is never imputed."""
    out = frame.copy()
    for c in out.columns:
        if c == "load_mw":
            out[c] = out[c].where(out[c] > 0)
        else:
            v = VARIABLES[c]
            out[c] = out[c].where(out[c].between(v.lo, v.hi))
    weather = [c for c in out.columns if c != "load_mw"]
    out[weather] = out[weather].ffill()
    return out


def solar_etr(index: pd.DatetimeIndex, city: City | tuple[float, float, int]) -> pd.DataFrame:
    """Extraterrestrial horizontal and direct-normal radiation (Wh/m²) integrated over each
    local-standard hour [t, t+1h), NOAA solar-position equations, 1-minute steps.
    Matches the NSRDB values in the research data to ~5 Wh/m² MAE (see tests)."""
    lat, lon, off = (city.lat, city.lon, city.utc_offset) if isinstance(city, City) else city
    t = index.values.astype("datetime64[m]")[:, None] + np.arange(60).astype("timedelta64[m]")[None, :]
    ts = pd.DatetimeIndex(t.ravel())
    doy = ts.dayofyear.values.reshape(t.shape)
    hr = (ts.hour.values + ts.minute.values / 60).reshape(t.shape)
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.deg2rad((hr * 60 + eot + 4 * lon + 60 * off) / 4 - 180)
    cz = np.sin(np.deg2rad(lat)) * np.sin(decl) + np.cos(np.deg2rad(lat)) * np.cos(decl) * np.cos(ha)
    i0 = 1367 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))
    return pd.DataFrame({
        "extraterrestrial_horizontal_radiation_wh_m2": (np.clip(cz, 0, None) * i0).mean(axis=1),
        "extraterrestrial_direct_normal_radiation_wh_m2": np.where(cz > 0, i0, 0).mean(axis=1),
    }, index=index)
