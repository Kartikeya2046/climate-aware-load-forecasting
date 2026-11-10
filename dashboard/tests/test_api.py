"""The web frontend's API: one real request end to end (recorded Atlanta data, real models)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import api


def test_recorded_run_end_to_end():
    c = TestClient(api.app)
    meta = c.get("/api/meta").json()
    assert len(meta["cities"]) == 6 and all(m["loaded"] for m in meta["models"])
    v = c.post("/api/validate", json={"city": "atlanta", "start": "2023-07-10 00:00", "end": "2023-07-10 23:00",
                                      "mode": "one_step", "source": "recorded"}).json()
    assert v["pending_id"] and not v["fatal"]
    r = c.post("/api/run", json={"pending_id": v["pending_id"], "confirmed": True}).json()
    assert r["status"] == "ok" and r["selected_model"] == "tcn_atlanta"
    assert len(r["predictions"]["t"]) == 24 and all(x is not None for x in r["predictions"]["pred_mw"])
    assert r["summary"]["observed_MAPE_percent"] < 3
    assert c.get(f"/api/result/{r['id']}/export/csv").status_code == 200
    assert c.get(f"/api/result/{r['id']}/climate/tcn_atlanta").json()["hist"]
    assert c.post("/api/validate", json={"city": "atlanta", "start": "2023-07-10 00:00", "end": "2023-07-09 00:00",
                                         "source": "recorded"}).status_code == 400
