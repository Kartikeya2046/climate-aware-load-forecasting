"""UI workflows driven headlessly with Streamlit's AppTest (real models, recorded data)."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_md5

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["overview", "model-selection", "analytics", "climate", "data-quality", "report"]


def go(at, page):
    at._page_hash = calc_md5(page)  # st.Page hash = md5(url_path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def button(at, label):
    return next(b for b in at.button if b.label == label)


@pytest.fixture
def at():
    a = AppTest.from_file(APP, default_timeout=300)
    a.run()
    return go(a, "workspace")


def test_manual_workflow_and_every_page(at):
    at.selectbox(key="city").set_value("phoenix").run()
    button(at, "Validate inputs").click()
    at.run()
    assert not at.exception
    if any(c.key == "confirm" for c in at.checkbox):
        next(c for c in at.checkbox if c.key == "confirm").check().run()
    button(at, "Run prediction").click()
    at.run()
    assert not at.exception and any("Prediction complete" in s.value for s in at.success)
    assert at.session_state.result["response"].selected_model == "tcn_las_vegas"
    for p in PAGES:
        go(at, p)
    go(at, "overview")
    assert any("Total predicted energy" in m.value for m in at.markdown)


class Upload:
    def __init__(self, name, data):
        self.name, self._d = name, data

    def getvalue(self):
        return self._d


def test_csv_workflow_with_confirmation_and_download(at):
    from core.data import repository_record
    rec = repository_record("birmingham").loc["2023-02-12":"2023-03-03 06:00"].copy()
    rec.iloc[400, 0] = 300.0  # implausible temperature -> recoverable correction requiring confirmation
    at.session_state["_injected_upload"] = Upload("bhm.csv", rec.to_csv().encode())
    at.selectbox(key="city").set_value("birmingham").run()
    at.session_state["_injected_upload"] = Upload("bhm.csv", rec.to_csv().encode())
    at.date_input(key="start_date").set_value(pd.Timestamp("2023-03-01").date())
    at.date_input(key="end_date").set_value(pd.Timestamp("2023-03-02").date())
    at.radio(key="mode").set_value("one_step")
    at.radio(key="input_mode").set_value("csv").run()
    assert not at.exception
    button(at, "Validate & preview").click()
    at.run()
    assert not at.exception
    assert any("Correction" in w.value for w in at.warning)
    assert button(at, "Run prediction").disabled  # blocked until corrections are confirmed
    next(c for c in at.checkbox if c.key == "confirm").check().run()
    button(at, "Run prediction").click()
    at.run()
    resp = at.session_state.result["response"]
    assert resp.status == "ok" and resp.selected_model == "tcn_atlanta" and resp.summary["hours_predicted"] == 48
    go(at, "data-quality")
    go(at, "report")


def test_any_city_csv_uses_climate_analog(at):
    data = (Path(APP).parent / "samples" / "kirkland_test_weather.csv").read_bytes()
    at.session_state["_injected_upload"] = Upload("kirkland.csv", data)
    at.text_input(key="custom_name").set_value("Kirk land")
    at.radio(key="input_mode").set_value("csv").run()
    button(at, "Validate & preview").click()
    at.run()
    next(c for c in at.checkbox if c.key == "confirm").check().run()  # visibility corrections
    button(at, "Run prediction").click()
    at.run()
    assert not at.exception
    resp = at.session_state.result["response"]
    assert resp.status == "ok" and resp.analog["city"] == "phoenix" and resp.selected_model == "tcn_las_vegas"
    for p in PAGES:
        go(at, p)


def test_any_city_with_load_uses_lstm_analog(at):
    data = (Path(APP).parent / "samples" / "maple_falls_test_weather_load.csv").read_bytes()
    at.session_state["_injected_upload"] = Upload("maple.csv", data)
    at.date_input(key="start_date").set_value(pd.Timestamp("2015-07-08").date())
    at.date_input(key="end_date").set_value(pd.Timestamp("2015-07-10").date())
    at.radio(key="mode").set_value("one_step")
    at.radio(key="input_mode").set_value("csv").run()
    button(at, "Validate & preview").click()
    at.run()
    button(at, "Run prediction").click()
    at.run()
    assert not at.exception
    resp = at.session_state.result["response"]  # temperature in deg F: the suggested unit is pre-selected and converted
    assert resp.analog["city"] == "pittsburgh" and resp.selected_model == "lstm_dayton" and resp.summary["observed_MAPE_percent"] < 10


def test_validation_error_display_for_fatal_csv(at):
    at.session_state["_injected_upload"] = Upload("bad.csv", b"timestamp,temperature_c\nnope,1\nnope,2\n")
    at.radio(key="input_mode").set_value("csv").run()
    button(at, "Validate & preview").click()
    at.run()
    assert any("could be parsed" in e.value for e in at.error)
    assert not any(b.label == "Run prediction" for b in at.button)


def test_reset(at):
    at.session_state["result"] = None
    button(at, "Reset workspace").click()
    at.run()
    assert "result" not in at.session_state
