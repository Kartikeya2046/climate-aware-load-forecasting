# Energy Intelligence Dashboard

An interactive Streamlit application that predicts hourly electricity consumption from weather, using
the project's **real trained models**. For each request it checks every model's eligibility,
calculates a transparent **Model Fitness Score**, selects the highest-scoring eligible model, and
shows the prediction with uncertainty, analytics, climate similarity and data-quality information.

```bash
# from the repository root, in the Anaconda base environment (Python 3.13, TensorFlow 2.21 / Keras 3)
pip install -r requirements.txt          # streamlit, plotly, pytest on top of the research stack
cd dashboard
streamlit run app.py                     # http://localhost:8501
python -m pytest tests -q                # 51 tests, ~40 s, real artifacts
python -m core.evidence                  # optional: regenerate held-out evidence (~3 min, CPU)
python -m core.evidence --weights-study  # optional: re-evaluate the fitness weights
```

Run `streamlit` from inside `dashboard/` so that `dashboard/.streamlit/config.toml` (light theme,
25 MB upload limit) is picked up. No GPU or retraining is needed.

## Architecture

```
dashboard/
├── app.py            Streamlit UI: 7 pages, session state, widgets. No model or scoring logic.
├── charts.py         Plotly figure builders (validated palette; every figure draws returned values only)
├── config.toml       fitness weights, thresholds, model manifest (artifact paths)
├── core/             UI-independent, testable
│   ├── schema.py       canonical variables, units, aliases; request/score/response data contracts
│   ├── data.py         supported cities, repository datasets, timestamp alignment, solar geometry
│   ├── validation.py   CSV ingestion, column mapping, unit conversion, validation + processing log
│   ├── models.py       model registry, artifact checks, loaders, TCN / LSTM inference adapters
│   ├── climate.py      project climate-similarity method, PCA space, city matrix
│   ├── scoring.py      eligibility helpers, six fitness components, weighting, ranking
│   ├── service.py      prepare_inputs + run_prediction: the end-to-end pipeline
│   ├── evidence.py     offline held-out backtests -> evidence/evidence.json
│   ├── weights_study.py evaluation of weight sets on held-out windows
│   └── report.py       CSV / JSON / HTML exports
├── evidence/         evidence.json, weights_study.json (generated, committed)
└── tests/            pytest + Streamlit AppTest workflows
```

Pipeline: raw input → schema mapping → validation → context assembly (history, alignment) →
eligibility → fitness scoring → selection → inference → inverse transformation → output checks →
empirical intervals → energy aggregation → analytics/exports.

## Supported models (verified against the research outputs)

| id | Family | Artifacts | Inputs | Target |
|---|---|---|---|---|
| `tcn_atlanta` | Two-branch TCN | `experiments/04_atlanta_birmingham/neural_model/atlanta_birmingham_two_branch_results/`: city baseline MLPs (Atlanta, Birmingham) + Atlanta weather-residual TCN + scalers | 168 h × 13: temperature, dew point, RH, precipitation (+ CDD, HDD, T×RH) + 6 residual-history features | next-hour log residual → MW |
| `tcn_las_vegas` | Two-branch TCN | `experiments/05_las_vegas_phoenix/neural_models/two_branch_tcn_results/`: baselines (Las Vegas, Phoenix) + Las Vegas weather-residual TCN + scalers | 168 h × 23: 12 weather variables incl. pressure, wind, visibility, precipitable water, sky cover, extraterrestrial radiation (+ derived) + 6 residual-history features | next-hour log residual → MW |
| `lstm_dayton` | LSTM | `experiments/03_dayton_duquesne/models/day_lstm_model.keras` + `day_scaler.pkl` | 24 h × 12: load, 8 weather variables, hour, day of week, month (MinMax) | next-hour load, MW |

Two-branch combination (as trained): `log E = log E_baseline + r`, i.e.
`E = E_baseline + ΔE_weather` with `ΔE = E_baseline·(eʳ − 1)`. The baseline uses only calendar and trend
features. `r` is the TCN's next-hour residual, which comes from weather plus residual history.

Reproduction checks (tests): the adapters reproduce the research notebooks' saved 2023 predictions.
Birmingham differs by 0.08 MW mean (MAE 12.59 vs 12.60 MW) and Phoenix by 0.39 MW mean
(MAE 119.50 vs 119.50 MW); the remaining difference is float16 mixed precision in training vs
float32 here. The Dayton LSTM reproduces `DUQ_predictions.csv` exactly.

Not registered: `best_dayton_lstm.keras` (its scalers were never saved), the 01 campus models (their
Lambda layer cannot be reloaded), and the 04/05 LSTM variants (not part of the agreed model set; listed
under "All model artifacts discovered" on the Model Selection page). No XGBoost model was ever saved.

The 04/05 files were saved by TensorFlow 2.10 as HDF5 under `mixed_float16`. `models.load_keras`
loads a temporary copy whose config uses the float32 policy, so the repository files are never
modified. Outputs agree with TF 2.10 to about 1e-3.

### Locations

Dayton, Pittsburgh (PJM DUQ), Atlanta, Birmingham, Las Vegas and Phoenix.

- **Two-branch models** are applicable only where their experiment trained a city baseline.
- **The Dayton LSTM** is technically applicable everywhere, but it is *conditional* (cross-domain)
  outside Dayton and Pittsburgh.
- **Any other city** (Location → "Any city — match by climate", the default): upload its weather CSV.
  The weather is scored against every trained city's recorded climate (same months ±1, project
  formula), and cities are tried from most to least similar. The first city with an eligible model is
  the *analog*, and the usual fitness ranking picks the model there (`service.run_analog`). The result
  is "this weather on the analog city's grid": its MW scale, infrastructure and population are
  assumed. Radiation is derived from the analog city's coordinates. Without a load column only
  forecast mode works, and the LSTM (which needs 24 h of observed load) is skipped, so Dayton or
  Pittsburgh is used only when the CSV includes load.

### Held-out evidence (`evidence/evidence.json`, MAPE %)

| Model → location | one hour ahead | 168 h free-running | naive (24 h / 168 h) |
|---|---|---|---|
| TCN Atlanta → Atlanta | 1.07 | 14.47 | 5.75 / 9.40 |
| TCN Atlanta → Birmingham | 1.01 | 13.77 | 5.69 / 9.34 |
| TCN Las Vegas → Las Vegas | 1.29 | 7.01 | 3.67 / 6.94 |
| TCN Las Vegas → Phoenix | 1.90 | 9.79 | 4.22 / 8.42 |
| LSTM Dayton → Dayton | 1.72 | 9.07 | 8.07 / 9.82 |
| LSTM Dayton → Pittsburgh | 3.98 | 15.51 | 6.66 / 9.18 |
| LSTM Dayton → Atlanta / Birmingham / Las Vegas / Phoenix | 6.6 / 17.3 / 7.4 / 17.4 | 22–98 | — |

Holdouts: 2023 for the TCN cities (training 2016–2021, validation 2022), and the Dayton LSTM's
chronological test split (2014-06 → 2015-12) for Dayton and Pittsburgh. **All models are
one-hour-ahead models.** Fed their own predictions for a week, they are worse than the naive
benchmarks; at Atlanta and Birmingham they are even worse than the calendar baseline alone
(10.7 % and 7.7 %). The dashboard shows this instead of hiding it.

## Model Fitness Score

`Fitness = Σ wᵢ·sᵢ / Σ wᵢ` over the *available* components. Each sᵢ is on a 0–100 scale. Unavailable
components are dropped and the weights renormalised; they are never scored as perfect. This is a
ranking rubric, **not a probability**.

| Component | Weight | Definition |
|---|---:|---|
| Validation performance | 0.35 | `100·2^(skill−1)`. Skill = ½·(1 − MAE/MAE_naive) + ½·(1 − RMSE/RMSE_naive) on held-out hours for the same location and mode. 100 = perfect, 50 = naive-equivalent |
| Climate similarity | 0.25 | Project formula (0.6 distribution [SMD, normalised Wasserstein, KS] + 0.2 correlation + 0.2 multivariate) comparing the input weather with the model's training weather for the same months ±1. A single observation gets a point proximity instead, labelled low confidence |
| Feature compatibility | 0.15 | Mean over required inputs: 1.0 supplied, 0.95 unit-converted, 0.90 derived. A missing mandatory input makes the model ineligible |
| Input data quality | 0.10 | `100·(1 − min(1, missing + ½·filled + implausible + duplicate + gap fractions))` |
| Temporal & horizon | 0.10 | `100·(½ + ½·observed-history share)·min(1, 168/H)·date-domain factor·coverage` |
| Calibration & reliability | 0.05 | Mean of interval-coverage accuracy (90 % interval fitted on the first half of the holdout, tested on the second) and half-to-half MAE stability |

Eligibility is checked before scoring:
- artifacts exist and load
- schema and shape checks pass
- the health check passes
- the location pathway exists
- the required inputs are present
- the first model window has weather
- observed load history exists (LSTM; one-hour-ahead mode)

A two-branch model without load history is *conditional*: unknown residual history is set to 0,
meaning load is assumed equal to the calendar baseline. Ineligible models never rank. Ranking is
deterministic: fitness, then validation score, then id. Candidates within 1 point of the winner are
flagged as near ties.

**Weight study** (`evidence/weights_study.json`): 208 second-half weekly windows, with validation taken
from first-half windows only. The configured weights select the best model in 98.1 % of windows, with
a mean regret of 0.04 MAPE points. Validation-only, climate-only and equal weights perform the same,
because each location has at most two scorable candidates and one family dominates. The study
supports the selections, but it **cannot discriminate between weight sets**; the weights remain
hypotheses.

## Data handling

- **Canonical schema:** an hourly index in **local standard time** (no DST), with the variables, units
  and aliases in `core/schema.py`.
  - Exact names and registered aliases map automatically.
  - Ambiguous or unknown columns are never guessed.
  - Units are suggested from the value range and must be confirmed.
- **Three severities:**
  - fatal: blocks the run
  - recoverable: a logged correction the user must confirm before *Run prediction* is enabled
  - warning: shown, does not block
- **Corrections, all logged:**
  - Implausible values (the training ranges) become missing.
  - Weather gaps of 6 h or less are forward-filled from past values, as in training.
  - Observed load is never imputed.
  - Duplicate timestamps keep the first occurrence.
  - Sub-hourly data is averaged to hourly (precipitation is summed); coarser data is fatal.
  - Statistically unusual values are flagged, never removed.
- **The fitted training scalers are always used;** nothing is fitted on uploaded data. The original
  upload is kept unchanged and can be downloaded next to the processed data.
- **Timestamp alignment (research data issue, verified for all six cities):** in the research data,
  load is labelled in UTC and weather in local standard time, and the training pipelines joined them
  on the raw label.
  - The adapters reproduce that pairing exactly (`data.to_model_frame`).
  - A consequence is that a prediction for local hour L reads weather up to L + (UTC offset − 1) h.
  - Missing lead hours hold the last observation, as a confirmed correction.
- **Extraterrestrial radiation** (needed by the Las Vegas TCN) is derived from NOAA solar geometry
  when absent. It matches the NSRDB values to about 4 Wh/m² MAE.

## Units

- The models predict the hourly average load in MW.
- Energy per 1-hour interval = MW × 1 h = MWh. Totals are sums over the predicted hours.
- Daily aggregation sums MWh and averages MW.
- **Intervals:**
  - Hourly 90 % intervals use the empirical 5–95 % quantiles of held-out `log(actual/pred)`, for the
    same model, location and mode (per horizon bucket in forecast mode).
  - Total-energy intervals use held-out 24 h or 168 h window totals. They are omitted for longer
    horizons or incomplete coverage.

## Dashboard pages

| Page | Contents |
|---|---|
| **Overview** | Headline total energy (or average power) with its interval, average per interval, peak and time, selected model, fitness, eligible count, data-quality status and mode |
| **Prediction Workspace** | Location and coverage, timeframe, mode, primary unit, aggregation, the manual form (fields come from the applicable models and are prefilled with recorded monthly medians), and CSV upload with preview, mapping, units and timestamp convention. Also offers a sample-CSV download, validation results, a confirmation gate and Run |
| **Model Selection** | Sortable candidate table with score bars, eligibility filter, fitness chart, the rationale, per-model expanders (components, calculations, held-out evidence by mode, research-reported metrics, artifacts), benchmarks, the weight study and the discovered artifacts |
| **Energy Analytics** | Date-range filter, hourly/daily and MW/MWh toggles, and computed alternatives. Charts: time series with interval, cumulative energy, hour and hour×weekday profiles, weather series, weather correlations, temperature/RH/dew point vs load, baseline vs adjustment, observed vs predicted, and the model comparison |
| **Climate Similarity** | Per-candidate similarity, the variable table, distribution overlays, correlation-structure comparison, a PCA projection (fixed fitted scaler + PCA), the city×city matrix and the custom-location analog |
| **Data Quality** | File info, row counts, per-variable missing/invalid/filled/unusual counts, mapping, conditions, processing log, a processed-data inspector and downloads |
| **Prediction Report** | Downloads (CSV, HTML, JSON) and the full report |

## Tests (`python -m pytest tests -q`)

- **51 tests**, covering:
  - CSV parsing, mapping and aliases, units, timestamps, duplicates, gaps, missing and invalid values,
    frequency and UTC conversion
  - registry and discovery, missing artifacts and scalers, corrupt files, schema mismatch
  - reproduction of the research predictions; sequence and residual features; recursive = one-step
    at the first step; solar derivation; alignment round trip
  - score normalisation, renormalisation, missing components, deterministic ranking, ties,
    no-eligible fallback
  - service integration per city, energy aggregation and intervals, the manual scenario,
    unsupported locations, exports
  - AppTest UI workflows: the manual flow across all pages, CSV upload with confirmation, fatal-error
    display, reset
- Synthetic tables are used only for parsing and scoring logic, and are labelled as such.
- AppTest cannot drive `st.file_uploader`, so the CSV UI test uses a session-state seam
  (`_injected_upload`).

## Limitations and remaining work

- **Accuracy is good only one hour ahead with observed load.** Free-running forecasts and manual
  scenarios are far less accurate, and they are flagged as such. Production multi-hour forecasting
  needs models trained for that horizon.
- **The research data's UTC/local timestamp mismatch should be fixed and the models retrained.**
  The dashboard currently reproduces it faithfully.
- **Dayton and Pittsburgh have only one model** (no two-branch baseline), so there is no real
  selection there. Expanding the candidate pool (for example, the 04/05 LSTM variants, which have
  saved metadata) would make the selection and the weight study more informative.
- **Calibration evidence is limited.** The intervals assume future errors resemble those on the
  2023 / test-split holdout. The scalers were pickled with scikit-learn 1.7.2 and load in 1.6.1;
  the reproduction tests guard this.
- **Not production-hardened:** single user, no authentication, no persistence, CPU inference (about
  5–8 s per request), and no static image export of the charts.
