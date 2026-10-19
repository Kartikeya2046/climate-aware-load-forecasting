"""Score normalisation, weights, missing components, ranking, ties, fallback (pure logic)."""
import pandas as pd
import pytest

from core.schema import ComponentScore, ModelScore
from core.scoring import (COMPONENTS, calibration_score, evidence_key, feature_score, fitness, quality_score, rank,
                          skill_to_score, temporal_score, validation_score)


def comps(**scores):
    w = dict(zip(COMPONENTS, [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]))
    return [ComponentScore(n, scores.get(n), w[n]) for n in COMPONENTS]


def ms(mid, fit, elig="eligible", val=50.0):
    return ModelScore(mid, mid, "f", "v", elig, [], comps(validation_performance=val), fit)


def test_skill_transform_is_monotone_and_anchored():
    assert skill_to_score(1.0) == 100 and skill_to_score(0.0) == 50 and skill_to_score(-1.0) == 25
    assert skill_to_score(-0.15) > skill_to_score(-8.0) > 0  # worse-than-naive models stay distinguishable
    assert skill_to_score(1.5) == 100


def test_all_components_bounded():
    ev = {"skill_mae": 0.8, "skill_rmse": 0.7, "MAPE_percent": 1, "MAE_MW": 10, "R2": .99, "n": 100, "naive": {"MAPE_percent": 5},
          "calibration": {"coverage": 0.9, "nominal": 0.9}, "halves": {"H1": {"MAE_MW": 10}, "H2": {"MAE_MW": 12}}}
    for s, _ in [validation_score(ev), calibration_score(ev), feature_score(["a", "b"], {"a"}, {"b"}),
                 temporal_score("LSTM", "forecast", 400, 0.3, pd.Timestamp("2030-01-01"), pd.Timestamp("2030-01-20"), 0.5)]:
        assert 0 <= s <= 100
    assert calibration_score(ev)[0] == pytest.approx((100 + 100 * 10 / 12) / 2)


def test_weights_renormalise_over_available_components():
    c = comps(validation_performance=80, climate_similarity=60, feature_compatibility=100, input_data_quality=100,
              temporal_compatibility=100, calibration_reliability=None)
    f = fitness(c)
    expected = (0.35 * 80 + 0.25 * 60 + 0.15 * 100 + 0.10 * 100 + 0.10 * 100) / 0.95
    assert f == pytest.approx(expected)
    assert sum(x.effective_weight for x in c) == pytest.approx(1.0)
    assert next(x for x in c if x.name == "calibration_reliability").effective_weight == 0


def test_missing_evidence_is_unavailable_not_perfect():
    assert validation_score(None)[0] is None and calibration_score({})[0] is None
    c = comps(validation_performance=None, climate_similarity=40, feature_compatibility=40, input_data_quality=40,
              temporal_compatibility=40, calibration_reliability=None)
    assert fitness(c) == pytest.approx(40)  # not lifted by the missing components
    assert fitness(comps()) is None


def test_ranking_is_deterministic_excludes_ineligible_and_flags_ties():
    scores = [ms("b", 70.0), ms("a", 70.0), ms("c", 95.0, "ineligible"), ms("d", 69.5, "conditional")]
    out = rank(scores, near_tie=1.0)
    assert [s.model_id for s in out if s.rank] == ["a", "b", "d"]  # tie broken by validation then id
    assert out[0].selected and not any(s.selected for s in out[1:])
    assert next(s for s in out if s.model_id == "c").rank is None
    assert next(s for s in out if s.model_id == "b").near_tie and next(s for s in out if s.model_id == "d").near_tie
    assert [s.model_id for s in rank(list(reversed(scores)))] == [s.model_id for s in out]


def test_tie_break_uses_validation_before_id():
    out = rank([ms("a", 70.0, val=40), ms("b", 70.0, val=60)])
    assert out[0].model_id == "b"


def test_no_eligible_model():
    out = rank([ms("x", None, "ineligible"), ms("y", 50, "ineligible")])
    assert not any(s.selected for s in out)


def test_evidence_key():
    assert evidence_key("one_step", "observed") == "one_step"
    assert evidence_key("forecast", "observed") == "forecast"
    assert evidence_key("forecast", "baseline_assumption") == "forecast_no_history"


def test_quality_score_penalises_issues():
    f = pd.DataFrame({"a": [1.0, None, 3.0, 4.0]})
    s, _ = quality_score(f, ["a"], {"rows_hourly": 4, "rows_uploaded": 4, "columns": {"a": {"filled": 0, "invalid": 0}}})
    assert s == pytest.approx(75)
