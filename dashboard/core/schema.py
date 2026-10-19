"""Canonical variables, units, aliases and the data contracts shared by every layer.

All frames inside the application use one canonical schema: an hourly, timezone-naive
index in the city's *local standard time* (no daylight saving) and the column names below.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Variable:
    name: str
    label: str
    unit: str
    lo: float  # physically plausible range used by the training pipelines (values outside -> invalid)
    hi: float
    help: str
    aliases: tuple[str, ...] = ()
    units: dict[str, tuple[float, float]] = field(default_factory=dict)  # unit label -> (scale, offset) to canonical
    derivable: bool = False  # can be computed exactly from timestamp + coordinates


VARIABLES: dict[str, Variable] = {v.name: v for v in [
    Variable("temperature_c", "Temperature", "°C", -60, 60, "Dry-bulb air temperature.",
             ("temperature", "temp", "tempc", "airtemperature", "airtemp", "drybulb", "drybulbtemperature", "tmp", "t2m"),
             {"°C": (1, 0), "°F": (5 / 9, -160 / 9), "K": (1, -273.15)}),
    Variable("dew_point_c", "Dew point", "°C", -80, 50, "Dew-point temperature.",
             ("dewpoint", "dewpointc", "dewpointtemperature", "dewp", "td"),
             {"°C": (1, 0), "°F": (5 / 9, -160 / 9), "K": (1, -273.15)}),
    Variable("relative_humidity_percent", "Relative humidity", "%", 0, 100, "Relative humidity, 0–100 %.",
             ("relativehumidity", "humidity", "rh", "rhpercent", "relhum"),
             {"%": (1, 0), "fraction (0-1)": (100, 0)}),
    Variable("station_pressure_kpa", "Station pressure", "kPa", 80, 110,
             "Station (not sea-level) atmospheric pressure.",
             ("pressure", "stationpressure", "pressurekpa", "pres", "atmosphericpressure", "stnpressure"),
             {"kPa": (1, 0), "hPa / mbar": (0.1, 0), "inHg": (3.386389, 0), "Pa": (0.001, 0)}),
    Variable("wind_direction_deg", "Wind direction", "°", 0, 360, "Direction the wind blows from, degrees.",
             ("winddirection", "winddir", "wd", "wdir", "winddirectiondeg", "winddirdeg"),
             {"°": (1, 0)}),
    Variable("wind_speed_ms", "Wind speed", "m/s", 0, 75, "Wind speed.",
             ("windspeed", "ws", "wspd", "windspeedmps", "windspeedms"),
             {"m/s": (1, 0), "km/h": (1 / 3.6, 0), "mph": (0.44704, 0), "knots": (0.514444, 0)}),
    Variable("horizontal_visibility_km", "Visibility", "km", 0, 100, "Horizontal visibility.",
             ("visibility", "vis", "visibilitykm", "horizontalvisibility"),
             {"km": (1, 0), "mi": (1.609344, 0), "m": (0.001, 0)}),
    Variable("hourly_precipitation_cm", "Precipitation", "cm", 0, 100, "Liquid precipitation in the hour.",
             ("precipitation", "precip", "precipcm", "rain", "rainfall", "prcp", "hourlyprecipitation"),
             {"cm": (1, 0), "mm": (0.1, 0), "in": (2.54, 0)}),
    Variable("precipitable_water_mm", "Precipitable water", "mm", 0, 100, "Total-column precipitable water.",
             ("precipitablewater", "pw", "pwat"),
             {"mm": (1, 0), "cm": (10, 0)}),
    Variable("total_sky_cover_tenths", "Total sky cover", "tenths", 0, 10, "Cloud cover in tenths of sky.",
             ("totalskycover", "cloudcover", "totalcloudcover", "tcc", "skycover"),
             {"tenths": (1, 0), "%": (0.1, 0), "oktas": (10 / 8, 0)}),
    Variable("opaque_sky_cover_tenths", "Opaque sky cover", "tenths", 0, 10, "Opaque cloud cover in tenths of sky.",
             ("opaqueskycover", "opaquecloudcover"),
             {"tenths": (1, 0), "%": (0.1, 0), "oktas": (10 / 8, 0)}),
    Variable("extraterrestrial_horizontal_radiation_wh_m2", "Extraterrestrial horizontal radiation", "Wh/m²", 0, 1500,
             "Top-of-atmosphere radiation on a horizontal surface in the hour. Derived from solar geometry if absent.",
             ("extraterrestrialhorizontalradiation", "etr", "etrh"), {"Wh/m²": (1, 0)}, derivable=True),
    Variable("extraterrestrial_direct_normal_radiation_wh_m2", "Extraterrestrial direct-normal radiation", "Wh/m²", 0, 1500,
             "Top-of-atmosphere direct-normal radiation in the hour. Derived from solar geometry if absent.",
             ("extraterrestrialdirectnormalradiation", "etrn", "etrdn"), {"Wh/m²": (1, 0)}, derivable=True),
    Variable("load_mw", "Actual load", "MW", 0, 200_000,
             "Observed hourly average electricity load. Optional; provides history context and ground truth.",
             ("load", "mw", "loadmw", "hourlyloadmw", "demand", "demandmw", "consumption", "consumptionmw", "hourlyload"),
             {"MW": (1, 0), "MWh per hour": (1, 0), "kW": (0.001, 0), "GW": (1000, 0)}),
]}

WEATHER_VARS = [v for v in VARIABLES if v != "load_mw"]
TIMESTAMP_ALIASES = ("timestamp", "datetime", "datetimelocal", "time", "date_time", "datetimebeginning", "localtime", "date")


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def match_column(column: str) -> tuple[str | None, str]:
    """Map a CSV header to a canonical variable. Returns (variable, confidence) with confidence
    'exact' (canonical name or registered alias) or 'none'. Unknown names are never guessed."""
    n = norm(column)
    for v in VARIABLES.values():
        if n == norm(v.name) or n in v.aliases:
            return v.name, "exact"
    return None, "none"


# --------------------------------------------------------------------------- data contracts

@dataclass
class LogEntry:
    step: str
    action: str
    detail: str
    rows_affected: int = 0


@dataclass
class Issue:
    severity: str  # "fatal" | "recoverable" | "warning"
    code: str
    message: str
    column: str | None = None
    rows: int = 0


@dataclass
class PredictionRequest:
    location_id: str
    start: pd.Timestamp  # first predicted hour, local standard time (hour-beginning)
    end: pd.Timestamp    # last predicted hour, inclusive
    input_mode: str      # "manual" | "csv"
    weather: pd.DataFrame  # canonical hourly frame (may include load_mw and rows before `start` as context)
    context_source: str = "provided"  # "provided" | "repository" | "none"
    prediction_mode: str = "forecast"  # "forecast" (recursive) | "one_step" (needs actual load every hour)
    aggregation: str = "hourly"        # "hourly" | "daily"
    units: dict[str, str] = field(default_factory=dict)
    scenario: bool = False             # True when one manual observation is repeated over the timeframe
    processing_log: list[LogEntry] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    compute_alternatives: bool = True
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class ComponentScore:
    name: str
    score: float | None        # 0-100, None = unavailable
    weight: float              # configured weight
    effective_weight: float = 0.0
    detail: str = ""
    confidence: str = "normal"  # "normal" | "low" | "unavailable"


@dataclass
class ModelScore:
    model_id: str
    display_name: str
    family: str
    version: str
    eligibility: str            # "eligible" | "conditional" | "ineligible"
    reasons: list[str]
    components: list[ComponentScore]
    fitness: float | None
    rank: int | None = None
    selected: bool = False
    near_tie: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    training_domain: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResponse:
    request_id: str
    location_id: str
    status: str                 # "ok" | "no_eligible_model" | "error"
    message: str
    selected_model: str | None
    selected_fitness: float | None
    scores: list[ModelScore]
    predictions: pd.DataFrame | None   # hourly, local standard time
    alternatives: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    selection_explanation: list[str]
    processing_log: list[LogEntry]
    issues: list[Issue]
    climate: dict[str, Any] = field(default_factory=dict)
    analog: dict[str, Any] = field(default_factory=dict)  # set when the location is not a trained city (service.run_analog)
    units: dict[str, str] = field(default_factory=lambda: {"power": "MW", "energy": "MWh"})

    def to_dict(self) -> dict[str, Any]:
        def conv(o):
            if isinstance(o, pd.DataFrame):
                return o.reset_index().astype({c: str for c in o.reset_index().select_dtypes("datetime").columns}).to_dict("records")
            if isinstance(o, pd.Timestamp):
                return str(o)
            if isinstance(o, dict):
                return {k: conv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [conv(x) for x in o]
            if hasattr(o, "__dataclass_fields__"):
                return conv(asdict(o))
            return o
        return conv({k: getattr(self, k) for k in self.__dataclass_fields__})
