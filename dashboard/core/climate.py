"""Climate similarity, reusing the project's methodology
(experiments/04_.../Atlanta_Birmingham_Climate_Similarity_Analysis.ipynb, same in 05):

  per variable   SMD -> 100·e^-SMD, normalised Wasserstein (W / pooled sd) -> 100·e^-nW,
                 KS statistic -> 100·(1-KS); variable score = mean of the three
  correlation    100·(1 - mean |r_a - r_b| over the upper triangle)
  multivariate   100·e^-d, d = distance between mean vectors after a StandardScaler fitted on the pooled data
  overall        0.60·distribution + 0.20·correlation + 0.20·multivariate

Dashboard additions (documented choices):
* Input weather is compared with the model's *training* weather restricted to the same calendar
  months (±1 month), so a summer week is not judged against a whole-year climatology.
* Fewer than `min_rows_distribution_similarity` rows, or a constant manual scenario, cannot
  support distribution statistics. A point-based proximity is reported instead and labelled low
  confidence: per variable 100·(1 - 2·|F(x) - 0.5|) with F the training ECDF.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data import CITIES, clean_like_training, config, repository_record

COMMON = ["temperature_c", "dew_point_c", "relative_humidity_percent", "station_pressure_kpa",
          "wind_speed_ms", "wind_direction_deg", "horizontal_visibility_km", "hourly_precipitation_cm"]


def variable_scores(a: np.ndarray, b: np.ndarray) -> dict:
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt((sa ** 2 + sb ** 2) / 2)
    smd = abs(a.mean() - b.mean()) / ps if ps > 0 else (0.0 if a.mean() == b.mean() else np.inf)
    wd = wasserstein_distance(a, b)
    nwd = wd / ps if ps > 0 else (0.0 if wd == 0 else np.inf)
    ks = ks_2samp(a, b).statistic
    s = {"SMD": float(smd), "Wasserstein": float(wd), "normalized_Wasserstein": float(nwd), "KS": float(ks),
         "SMD_similarity": float(100 * np.exp(-smd)), "Wasserstein_similarity": float(100 * np.exp(-nwd)),
         "KS_similarity": float(100 * (1 - ks)), "input_mean": float(a.mean()), "reference_mean": float(b.mean())}
    s["score"] = (s["SMD_similarity"] + s["Wasserstein_similarity"] + s["KS_similarity"]) / 3
    return s


def compare(a: pd.DataFrame, b: pd.DataFrame, variables: list[str]) -> dict:
    """The project formula for two weather samples."""
    a, b = a[variables].dropna(), b[variables].dropna()
    per = {v: variable_scores(a[v].to_numpy(), b[v].to_numpy()) for v in variables}
    dist = float(np.mean([p["score"] for p in per.values()]))
    varying = [v for v in variables if a[v].std() > 0 and b[v].std() > 0]
    corr = None
    if len(varying) >= 2:
        diff = (a[varying].corr() - b[varying].corr()).to_numpy()
        corr = float(np.clip(1 - np.mean(np.abs(diff[np.triu_indices_from(diff, k=1)])), 0, 1) * 100)
    pooled = pd.concat([a, b])
    sc = StandardScaler().fit(pooled)
    d = float(np.linalg.norm(sc.transform(a).mean(0) - sc.transform(b).mean(0)))
    multi = float(100 * np.exp(-d))
    parts = {"distribution": (dist, 0.60), "correlation": (corr, 0.20), "multivariate": (multi, 0.20)}
    w = sum(wt for s, wt in parts.values() if s is not None)
    overall = sum(s * wt for s, wt in parts.values() if s is not None) / w
    return {"method": "distribution", "variables": per, "distribution_score": dist, "correlation_score": corr,
            "multivariate_score": multi, "multivariate_distance": d, "overall": float(overall),
            "classification": classify(overall), "n_input": int(len(a)), "n_reference": int(len(b))}


def classify(s: float) -> str:
    return "Highly similar" if s >= 80 else "Similar" if s >= 65 else "Moderately similar" if s >= 50 else "Dissimilar"


def seasonal(ref: pd.DataFrame, months: set[int]) -> pd.DataFrame:
    near = set()
    for m in months:
        near |= {(m - 2) % 12 + 1, m, m % 12 + 1}
    return ref[ref.index.month.isin(sorted(near))]


def point_proximity(values: dict[str, float], ref: pd.DataFrame, variables: list[str]) -> dict:
    per = {}
    for v in variables:
        x = ref[v].dropna().to_numpy()
        F = float(np.mean(x <= values[v]))
        per[v] = {"value": float(values[v]), "training_percentile": F * 100, "score": 100 * (1 - 2 * abs(F - 0.5)),
                  "reference_median": float(np.median(x))}
    overall = float(np.mean([p["score"] for p in per.values()]))
    return {"method": "point_proximity", "variables": per, "overall": overall, "classification": classify(overall),
            "n_input": 1, "n_reference": int(len(ref)),
            "note": "One observation cannot support distribution similarity; percentile-centrality proximity (low confidence)."}


@lru_cache(maxsize=None)
def reference_weather(city: str, start: str, end: str) -> pd.DataFrame:
    rec = clean_like_training(repository_record(city))
    return rec.loc[start:end].drop(columns="load_mw")


def similarity_to_training(inp: pd.DataFrame, ref_city: str, ref_start: str, ref_end: str,
                           variables: list[str], constant: bool) -> dict:
    ref = reference_weather(ref_city, ref_start, ref_end)
    variables = [v for v in variables if v in inp and v in ref and inp[v].notna().any()]
    if not variables:
        return {"method": "unavailable", "overall": None, "note": "no shared weather variables"}
    months = set(inp.index.month)
    sref = seasonal(ref, months)
    rows = inp[variables].dropna()
    base = {"reference": f"{CITIES[ref_city].name} training weather {ref_start[:10]} → {ref_end[:10]}, months {sorted(set(sref.index.month))}"}
    if constant or len(rows) < config()["validation"]["min_rows_distribution_similarity"]:
        vals = rows.iloc[-1].to_dict() if len(rows) else {}
        if not vals:
            return {"method": "unavailable", "overall": None, "note": "no complete weather rows"}
        return {**point_proximity(vals, sref, variables), **base, "confidence": "low"}
    return {**compare(rows, sref, variables), **base, "confidence": "normal" if len(rows) >= 720 else "low"}


@lru_cache(maxsize=1)
def city_matrix() -> dict:
    """Pairwise project climate-similarity score between all supported cities (full records, 8 variables)."""
    recs = {c: clean_like_training(repository_record(c))[COMMON].dropna() for c in CITIES}
    out = {}
    ids = list(CITIES)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            r = compare(recs[a].sample(20000, random_state=0), recs[b].sample(20000, random_state=0), COMMON)
            out[f"{a}|{b}"] = {"overall": r["overall"], "distribution": r["distribution_score"],
                               "correlation": r["correlation_score"], "multivariate": r["multivariate_score"],
                               "per_variable": {v: s["score"] for v, s in r["variables"].items()}}
    return out


@lru_cache(maxsize=1)
def pca_space():
    """PCA(2) fitted once on a fixed sample of all cities' recorded weather (StandardScaler first)."""
    sample = pd.concat([clean_like_training(repository_record(c))[COMMON].dropna().sample(3000, random_state=1).assign(city=c)
                        for c in CITIES])
    sc = StandardScaler().fit(sample[COMMON])
    pca = PCA(n_components=2, random_state=42).fit(sc.transform(sample[COMMON]))
    coords = pca.transform(sc.transform(sample[COMMON]))
    pts = pd.DataFrame(coords, columns=["PC1", "PC2"]).assign(city=sample.city.values)
    return sc, pca, pts


def project(inp: pd.DataFrame) -> pd.DataFrame | None:
    sc, pca, _ = pca_space()
    rows = inp.reindex(columns=COMMON).dropna()
    if rows.empty:
        return None
    return pd.DataFrame(pca.transform(sc.transform(rows[COMMON])), columns=["PC1", "PC2"], index=rows.index)
