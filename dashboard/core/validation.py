"""CSV ingestion, column mapping, unit conversion and validation with a processing log.

Conditions are reported in three classes:
  fatal        inference cannot continue (e.g. no parseable timestamps, coarser than hourly)
  recoverable  inference can continue after a logged correction the user must confirm
  warning      inference is possible but reliability may be reduced
Statistically unusual values are flagged, never removed: a heatwave is a valid condition.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import CITIES, clean_like_training, config, repository_record, solar_etr
from .schema import TIMESTAMP_ALIASES, VARIABLES, Issue, LogEntry, match_column, norm

PRECIP_LIKE = {"hourly_precipitation_cm"}  # summed, not averaged, when resampling


@dataclass
class Ingested:
    raw: pd.DataFrame                  # the uploaded table exactly as read (strings), kept for audit
    issues: list[Issue] = field(default_factory=list)
    log: list[LogEntry] = field(default_factory=list)


@dataclass
class Processed:
    frame: pd.DataFrame | None         # canonical hourly frame (local standard time)
    issues: list[Issue]
    log: list[LogEntry]
    quality: dict
    mapping: dict[str, str]

    @property
    def fatal(self) -> bool:
        return any(i.severity == "fatal" for i in self.issues)

    @property
    def needs_confirmation(self) -> bool:
        return any(i.severity == "recoverable" for i in self.issues)


def read_csv(data: bytes, filename: str) -> Ingested:
    """Safe CSV read: size/extension/encoding checks, everything kept as text (nothing is executed)."""
    cfg = config()["validation"]
    issues: list[Issue] = []
    if not filename.lower().endswith((".csv", ".txt")):
        issues.append(Issue("fatal", "file_type", f"'{filename}' is not a .csv file."))
    if len(data) > cfg["max_upload_mb"] * 1024 * 1024:
        issues.append(Issue("fatal", "file_size", f"File is {len(data) / 1e6:.1f} MB; the limit is {cfg['max_upload_mb']} MB."))
    if b"\x00" in data[:4096]:
        issues.append(Issue("fatal", "binary", "File contains binary data; upload a plain-text CSV."))
    if issues:
        return Ingested(pd.DataFrame(), issues)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        issues.append(Issue("warning", "encoding", "File is not UTF-8; it was decoded as Latin-1."))
    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, skipinitialspace=True, nrows=cfg["max_rows"] + 1)
    except Exception as exc:  # noqa: BLE001
        return Ingested(pd.DataFrame(), [Issue("fatal", "parse", f"Could not parse CSV: {exc}")])
    raw.columns = [str(c).strip() for c in raw.columns]
    if len(raw) > cfg["max_rows"]:
        issues.append(Issue("fatal", "row_limit", f"More than {cfg['max_rows']:,} rows; split the file."))
    if raw.empty or raw.shape[1] < 2:
        issues.append(Issue("fatal", "empty", "The CSV needs a timestamp column and at least one weather column."))
    if raw.columns.duplicated().any():
        issues.append(Issue("fatal", "duplicate_columns", f"Duplicate column names: {sorted(set(raw.columns[raw.columns.duplicated()]))}"))
    return Ingested(raw, issues, [LogEntry("upload", "read", f"{filename}: {len(raw):,} rows × {raw.shape[1]} columns", len(raw))])


def propose_mapping(columns: list[str]) -> dict[str, dict]:
    """Suggest a mapping. Only canonical names and registered aliases map automatically; two columns
    claiming one variable are both marked ambiguous; anything else stays unmapped."""
    out = {}
    ts = [c for c in columns if norm(c) in {norm(a) for a in TIMESTAMP_ALIASES}]
    preferred = [c for c in ts if norm(c) in ("timestamp", "datetime")]
    ts_col = (preferred or ts)[0] if len(preferred or ts) == 1 else None
    for c in columns:
        if c in ts:
            out[c] = {"variable": "timestamp" if c == ts_col else None, "status": "timestamp" if c == ts_col else "ambiguous_timestamp"}
            continue
        v, how = match_column(c)
        out[c] = {"variable": v, "status": "exact" if v else "unmapped"}
    claimed: dict[str, list[str]] = {}
    for c, m in out.items():
        if m["variable"] and m["variable"] != "timestamp":
            claimed.setdefault(m["variable"], []).append(c)
    for v, cols in claimed.items():
        if len(cols) > 1:
            for c in cols:
                out[c] = {"variable": None, "status": "ambiguous", "candidates": [v]}
    return out


def suggest_unit(variable: str, values: pd.Series) -> tuple[str, str | None]:
    """Default canonical unit, or a suggestion from the value range (always shown for confirmation)."""
    v = VARIABLES[variable]
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return v.unit, None
    med, mx = float(x.median()), float(x.max())
    rules = {
        "temperature_c": [(med > 45 or mx > 60, "°F")], "dew_point_c": [(med > 45 or mx > 55, "°F")],
        "relative_humidity_percent": [(mx <= 1.0, "fraction (0-1)")],
        "station_pressure_kpa": [(med > 800, "hPa / mbar"), (20 < med < 35, "inHg")],
        "wind_speed_ms": [(med > 25, "km/h")], "hourly_precipitation_cm": [(mx > 30, "mm")],
        "load_mw": [(med > 100_000, "kW")],
    }.get(variable, [])
    for cond, unit in rules:
        if cond:
            return unit, f"{v.label}: values (median {med:.4g}) look like {unit}; confirm the unit."
    return v.unit, None


def process(raw: pd.DataFrame, mapping: dict[str, str], units: dict[str, str], city: str,
            timestamp_col: str, timestamps_are: str = "local_standard") -> Processed:
    """Apply a confirmed mapping: parse, convert, deduplicate, regularise to hourly, flag issues."""
    cfg = config()["validation"]
    issues: list[Issue] = []
    log: list[LogEntry] = []
    q: dict = {"rows_uploaded": int(len(raw))}
    mapped = {c: v for c, v in mapping.items() if v and v != "timestamp"}
    ignored = [c for c in raw.columns if c not in mapped and c != timestamp_col]
    if ignored:
        log.append(LogEntry("mapping", "ignored columns", ", ".join(ignored)))
        issues.append(Issue("warning", "unmapped_columns", f"{len(ignored)} column(s) not mapped and ignored: {', '.join(ignored[:8])}"))
    if not mapped:
        return Processed(None, [Issue("fatal", "no_variables", "No columns are mapped to weather variables.")], log, q, mapping)

    # timestamps
    s = raw[timestamp_col].astype(str).str.strip()
    off = CITIES[city].utc_offset if city in CITIES else None
    if s.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).any():
        ts = pd.to_datetime(s, errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
        timestamps_are = "utc"
        log.append(LogEntry("timestamps", "timezone", "timestamps carry UTC offsets; converted via UTC"))
    else:
        ts = pd.to_datetime(s, errors="coerce", format="mixed")
    bad = int(ts.isna().sum())
    q["invalid_timestamps"] = bad
    if bad == len(raw):
        return Processed(None, [Issue("fatal", "timestamps", f"No values in '{timestamp_col}' could be parsed as timestamps.")], log, q, mapping)
    if bad:
        issues.append(Issue("recoverable", "invalid_timestamps", f"{bad} row(s) with unparseable timestamps will be dropped.", timestamp_col, bad))
        log.append(LogEntry("timestamps", "drop unparseable", f"{bad} rows", bad))
    if timestamps_are == "utc":
        if off is None:
            return Processed(None, [Issue("fatal", "timezone", "UTC timestamps need a supported location to convert to local standard time.")], log, q, mapping)
        ts = ts - pd.Timedelta(hours=off)
        log.append(LogEntry("timestamps", "UTC -> local standard", f"shifted by -{off} h"))

    df = pd.DataFrame(index=ts)
    for col, var in mapped.items():
        x = pd.to_numeric(raw[col].astype(str).str.strip().replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "null": np.nan}), errors="coerce")
        nonnum = int(x.isna().sum() - raw[col].astype(str).str.strip().isin(["", "NA", "NaN", "null"]).sum())
        if nonnum > 0:
            issues.append(Issue("recoverable", "non_numeric", f"{col}: {nonnum} non-numeric value(s) treated as missing.", var, nonnum))
        unit = units.get(var, VARIABLES[var].unit)
        scale, offset = VARIABLES[var].units.get(unit, (None, None))
        if scale is None:
            return Processed(None, issues + [Issue("fatal", "unit", f"Unsupported unit '{unit}' for {VARIABLES[var].label}.", var)], log, q, mapping)
        if (scale, offset) != (1, 0):
            x = x * scale + offset
            log.append(LogEntry("units", f"{col}: {unit} -> {VARIABLES[var].unit}", f"x·{scale:.6g} + {offset:.6g}", int(x.notna().sum())))
        df[var] = x.to_numpy()
    df = df[df.index.notna()]
    if not df.index.is_monotonic_increasing:
        log.append(LogEntry("timestamps", "sort", "rows were not in chronological order; sorted"))
        df = df.sort_index()
    dup = int(df.index.duplicated().sum())
    q["duplicate_timestamps"] = dup
    if dup:
        issues.append(Issue("recoverable", "duplicates", f"{dup} duplicate timestamp(s); the first occurrence is kept.", rows=dup))
        log.append(LogEntry("timestamps", "drop duplicates (keep first)", f"{dup} rows", dup))
        df = df[~df.index.duplicated(keep="first")]

    # frequency
    step = df.index.to_series().diff().median() if len(df) > 1 else pd.Timedelta(hours=1)
    q["median_interval_minutes"] = float(step / pd.Timedelta(minutes=1)) if pd.notna(step) else None
    if pd.notna(step) and step > pd.Timedelta(hours=1):
        return Processed(None, issues + [Issue("fatal", "frequency", f"Observations are every {step}; the models need hourly data. Coarser data is not interpolated.")], log, q, mapping)
    if pd.notna(step) and step < pd.Timedelta(hours=1):
        agg = {c: ("sum" if c in PRECIP_LIKE else "mean") for c in df.columns}
        n0 = len(df)
        df = df.resample("h").agg(agg)
        issues.append(Issue("recoverable", "resample", f"Sub-hourly data ({step}) aggregated to hourly (mean; precipitation summed)."))
        log.append(LogEntry("timestamps", "resample to hourly", f"{n0} -> {len(df)} rows", n0))
    elif (df.index != df.index.floor("h")).any():
        n_off = int((df.index != df.index.floor("h")).sum())
        df.index = df.index.floor("h")
        df = df[~df.index.duplicated(keep="first")]
        issues.append(Issue("recoverable", "off_hour", f"{n_off} timestamp(s) not on the hour were floored to the hour."))
        log.append(LogEntry("timestamps", "floor to hour", f"{n_off} rows", n_off))
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    gaps = len(full) - len(df)
    q["missing_hours"] = int(gaps)
    if gaps:
        issues.append(Issue("warning", "gaps", f"{gaps} hour(s) missing between {full[0]} and {full[-1]}.", rows=gaps))
    df = df.reindex(full)
    df.index.name = "timestamp"
    q["rows_hourly"] = int(len(df))

    # plausibility (same ranges the training pipelines enforced) and statistical flags
    q["columns"] = {}
    rec = repository_record(city) if city in CITIES else None
    for var in df.columns:
        v = VARIABLES[var]
        missing0 = int(df[var].isna().sum())
        invalid = ~df[var].between(v.lo, v.hi) & df[var].notna()
        if var == "load_mw":
            invalid |= df[var] <= 0
        n_inv = int(invalid.sum())
        if n_inv:
            issues.append(Issue("recoverable", "implausible", f"{v.label}: {n_inv} value(s) outside {v.lo}–{v.hi} {v.unit} treated as missing.", var, n_inv))
            log.append(LogEntry("validation", f"{var}: implausible -> missing", f"outside [{v.lo}, {v.hi}]", n_inv))
            df.loc[invalid, var] = np.nan
        flagged = 0
        if rec is not None and var in rec and rec[var].notna().sum() > 1000:
            lo, hi = rec[var].quantile([0.001, 0.999])
            flagged = int((~df[var].between(lo, hi) & df[var].notna()).sum())
            if flagged:
                issues.append(Issue("warning", "unusual", f"{v.label}: {flagged} value(s) outside the 0.1–99.9 % range recorded for {CITIES[city].name} ({lo:.4g}–{hi:.4g} {v.unit}); kept.", var, flagged))
        q["columns"][var] = {"missing": missing0, "invalid": n_inv, "unusual_flagged": flagged, "filled": 0}

    # weather gaps: past-only forward fill, limited (the training pipelines forward-filled weather)
    lim = cfg["max_forward_fill_hours"]
    for var in [c for c in df.columns if c != "load_mw"]:
        before = df[var].isna()
        df[var] = df[var].ffill(limit=lim)
        filled = int((before & df[var].notna()).sum())
        if filled:
            q["columns"][var]["filled"] = filled
            issues.append(Issue("recoverable", "forward_fill", f"{VARIABLES[var].label}: {filled} missing hour(s) forward-filled (gaps ≤ {lim} h, past values only).", var, filled))
            log.append(LogEntry("imputation", f"{var}: forward fill ≤ {lim} h", "past-only, as in the training pipelines", filled))
        left = int(df[var].isna().sum())
        if left:
            issues.append(Issue("warning", "missing_left", f"{VARIABLES[var].label}: {left} hour(s) remain missing; hours whose model window needs them cannot be predicted.", var, left))
    if "load_mw" in df:
        log.append(LogEntry("validation", "load_mw", "observed load is never imputed"))
    df = derive(df, city, log)
    q["rows_used"] = int(len(df))
    q["span"] = [str(df.index[0]), str(df.index[-1])]
    return Processed(df, issues, log, q, mapping)


def derive(df: pd.DataFrame, city: str, log: list[LogEntry]) -> pd.DataFrame:
    """Add solar-geometry radiation when it is not supplied (exact function of time and place)."""
    etr = ["extraterrestrial_horizontal_radiation_wh_m2", "extraterrestrial_direct_normal_radiation_wh_m2"]
    if city in CITIES and any(c not in df or df[c].isna().all() for c in etr):
        sol = solar_etr(df.index, CITIES[city])
        for c in etr:
            if c not in df or df[c].isna().all():
                df[c] = sol[c]
                log.append(LogEntry("feature engineering", f"{c} derived", "NOAA solar position at the city's coordinates (MAE ≈ 4 Wh/m² vs NSRDB)", len(df)))
    return df


def manual_frame(city: str, start: pd.Timestamp, end: pd.Timestamp, values: dict[str, float],
                 context_hours: int) -> tuple[pd.DataFrame, list[Issue], list[LogEntry]]:
    """One weather observation applied to every hour of the timeframe (and the model context)."""
    issues, log = [], []
    for var, x in values.items():
        v = VARIABLES[var]
        if x is None or not np.isfinite(x) or not (v.lo <= x <= v.hi):
            issues.append(Issue("fatal", "manual_range", f"{v.label} must be between {v.lo} and {v.hi} {v.unit}.", var))
    idx = pd.date_range(start - pd.Timedelta(hours=context_hours), end + pd.Timedelta(hours=12), freq="h", name="timestamp")
    df = pd.DataFrame({k: float(x) for k, x in values.items()}, index=idx)
    issues.append(Issue("warning", "scenario", "Scenario simulation: the same weather is applied to every hour, including the model's look-back window. This is not a weather forecast."))
    log.append(LogEntry("manual input", "constant scenario", f"{len(values)} variables repeated over {len(idx)} hours", len(idx)))
    return derive(df, city, log), issues, log


def repository_context(city: str, start: pd.Timestamp, hours: int, end: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """Recorded weather + load for `hours` before `start` (and load through `end` for ground truth),
    cleaned exactly as in training. None if the record does not cover the period."""
    rec = clean_like_training(repository_record(city))
    lo = start - pd.Timedelta(hours=hours)
    if lo < rec.index[0] or start > rec.index[-1]:
        return None
    return rec.loc[lo:(end or start - pd.Timedelta(hours=1))]
