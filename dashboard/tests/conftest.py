"""Shared fixtures. Tests that need real model artifacts use the repository files; synthetic
frames are used only for parsing/scoring logic and are labelled SYNTHETIC in their names."""
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="session")
def reg():
    from core.models import registry
    r = registry()
    for a in r.values():
        a.ensure_loaded()
    return r


@pytest.fixture
def synthetic_csv_bytes():
    """SYNTHETIC weather table (not real observations) for CSV-parsing tests."""
    idx = pd.date_range("2023-07-01", periods=48, freq="h")
    df = pd.DataFrame({"Timestamp": idx.astype(str), "Temp (F)": 95.0, "Humidity": 30.0, "Dew Point": 40.0, "precip": 0.0,
                       "mystery_col": 1.0})
    return df.to_csv(index=False).encode()
