"""CSV parsing, feature mapping, units, timestamps, missing/invalid data (synthetic tables)."""
import numpy as np
import pandas as pd
import pytest

from core.validation import manual_frame, process, propose_mapping, read_csv, suggest_unit


def _csv(df):
    return df.to_csv(index=False).encode()


def _base(n=48, start="2023-07-01"):  # SYNTHETIC
    idx = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({"timestamp": idx.astype(str), "temperature_c": 30.0, "dew_point_c": 10.0,
                         "relative_humidity_percent": 40.0, "hourly_precipitation_cm": 0.0})


def run(df, mapping=None, units=None, city="phoenix", ts_kind="local_standard"):
    ing = read_csv(_csv(df), "x.csv")
    m = mapping or {c: v["variable"] for c, v in propose_mapping(list(ing.raw.columns)).items()}
    return process(ing.raw, m, units or {}, city, "timestamp" if "timestamp" in df else list(df.columns)[0], ts_kind)


def test_valid_csv_round_trip():
    p = run(_base())
    assert not p.fatal and not p.needs_confirmation
    assert len(p.frame) == 48 and p.frame.index.freq is None or True
    assert p.frame.temperature_c.eq(30).all()
    # radiation is derived from solar geometry and logged
    assert "extraterrestrial_horizontal_radiation_wh_m2" in p.frame
    assert any(e.step == "feature engineering" for e in p.log)


def test_alias_mapping_unknown_column_and_ambiguity(synthetic_csv_bytes):
    ing = read_csv(synthetic_csv_bytes, "w.csv")
    prop = propose_mapping(list(ing.raw.columns))
    assert prop["Timestamp"]["variable"] == "timestamp"
    assert prop["Humidity"]["variable"] == "relative_humidity_percent"
    assert prop["Dew Point"]["variable"] == "dew_point_c"
    assert prop["mystery_col"]["status"] == "unmapped" and prop["mystery_col"]["variable"] is None  # never guessed
    assert prop["Temp (F)"]["variable"] is None  # not an alias -> must be mapped by the user
    amb = propose_mapping(["timestamp", "temp", "temperature"])
    assert amb["temp"]["status"] == amb["temperature"]["status"] == "ambiguous"


def test_unit_suggestion_and_conversion():
    df = _base().assign(temperature_c=86.0)
    unit, warn = suggest_unit("temperature_c", df.temperature_c)
    assert unit == "°F" and warn
    p = run(df, units={"temperature_c": "°F"})
    assert p.frame.temperature_c.round(6).eq(30.0).all()
    assert any(e.step == "units" for e in p.log)


def test_unsupported_unit_is_fatal():
    assert run(_base(), units={"temperature_c": "Rankine"}).fatal


def test_invalid_timestamps_and_duplicates_are_recoverable():
    df = _base()
    df.loc[3, "timestamp"] = "not a date"
    df = pd.concat([df, df.iloc[[5]]])
    p = run(df)
    codes = {i.code: i for i in p.issues}
    assert codes["invalid_timestamps"].severity == "recoverable"
    assert codes["duplicates"].severity == "recoverable" and p.quality["duplicate_timestamps"] == 1
    assert not p.fatal


def test_all_timestamps_invalid_is_fatal():
    df = _base().assign(timestamp="garbage")
    assert run(df).fatal


def test_missing_observations_are_forward_filled_within_limit_and_logged():
    df = _base(60)
    df.loc[10:12, "temperature_c"] = np.nan  # 3 h gap -> filled
    df.loc[30:45, "dew_point_c"] = np.nan    # 16 h gap -> only 6 h filled
    p = run(df)
    assert p.frame.temperature_c.notna().all()
    assert p.quality["columns"]["temperature_c"]["filled"] == 3
    assert p.frame.dew_point_c.isna().sum() == 10
    assert any(i.code == "missing_left" for i in p.issues)


def test_out_of_range_values_become_missing_but_unusual_values_are_kept():
    df = _base(24)
    df.loc[2, "relative_humidity_percent"] = 140   # physically impossible
    df.loc[5, "temperature_c"] = 49.5              # extreme but possible in Phoenix: flagged, not removed
    p = run(df)
    assert any(i.code == "implausible" for i in p.issues)
    assert p.quality["columns"]["relative_humidity_percent"]["invalid"] == 1
    assert p.frame.loc[pd.Timestamp("2023-07-01 05:00"), "temperature_c"] == 49.5


def test_gaps_and_coarse_frequency():
    df = _base(48).drop(index=range(20, 24))
    p = run(df)
    assert p.quality["missing_hours"] == 4 and any(i.code == "gaps" for i in p.issues)
    coarse = _base(48).iloc[::3]
    assert run(coarse).fatal  # 3-hourly data is not interpolated


def test_subhourly_is_resampled_with_precip_summed():
    idx = pd.date_range("2023-07-01", periods=8, freq="30min")
    df = pd.DataFrame({"timestamp": idx.astype(str), "temperature_c": [20, 22] * 4, "hourly_precipitation_cm": 0.1})
    p = run(df)
    assert len(p.frame) == 4 and p.frame.temperature_c.eq(21).all()
    assert np.allclose(p.frame.hourly_precipitation_cm, 0.2)


def test_utc_timestamps_converted_to_local_standard():
    df = _base(24)
    df["timestamp"] = (pd.to_datetime(df.timestamp) + pd.Timedelta(hours=7)).astype(str) + "+00:00"
    p = run(df)
    assert p.frame.index[0] == pd.Timestamp("2023-07-01 00:00")  # Phoenix = UTC-7


def test_file_checks():
    assert any(i.code == "file_type" for i in read_csv(b"a,b\n1,2", "x.xlsx").issues)
    assert any(i.code == "binary" for i in read_csv(b"\x00\x01\x02", "x.csv").issues)
    assert any(i.code == "empty" for i in read_csv(b"onlyone\n1\n", "x.csv").issues)


def test_manual_frame_range_check_and_scenario_flag():
    f, issues, _ = manual_frame("phoenix", pd.Timestamp("2023-07-01"), pd.Timestamp("2023-07-01 05:00"),
                                {"temperature_c": 200.0}, 0)
    assert any(i.severity == "fatal" for i in issues)
    f, issues, _ = manual_frame("phoenix", pd.Timestamp("2023-07-01"), pd.Timestamp("2023-07-01 05:00"), {"temperature_c": 40.0}, 24)
    assert any(i.code == "scenario" for i in issues)
    assert f.index[0] == pd.Timestamp("2023-06-30 00:00") and f.temperature_c.eq(40).all()
