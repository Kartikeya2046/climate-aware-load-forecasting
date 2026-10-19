# Experiments, workflows and known issues

> Detailed notes for every experiment folder, moved from the original top-level README.
> Paths are relative to the repository root. Back to the [project README](../README.md).

Hourly electricity-load forecasting from weather data, and testing whether a model trained on one
region transfers to a climatically similar one. The work is a sequence of five self-contained
experiments, each with its own data, notebooks, models and outputs:

| # | Folder | Region / data | Models |
|---|--------|---------------|--------|
| 01 | `experiments/01_campus_substation` | Campus SUB-S substation meter (+ BH3/BH4 building meters), campus weather station, 2025–26 | MLR, polynomial MLR, LSTM (v1, v2), Prophet |
| 02 | `experiments/02_indiana_kentucky` | AEP Indiana load + Indiana weather 2005–18; Kentucky weather for transfer | LSTM (Indiana → Kentucky) |
| 03 | `experiments/03_dayton_duquesne` | PJM Dayton (DAY) and Duquesne (DUQ) zone loads 2006–16 + SAMSON weather | LSTM (Dayton → Duquesne) |
| 04 | `experiments/04_atlanta_birmingham` | County load 2016–23 (HDF5) + SAMSON weather re-dated to 2016–23 | LSTM variants, two-branch TCN (Atlanta → Birmingham) |
| 05 | `experiments/05_las_vegas_phoenix` | Same sources for Las Vegas and Phoenix | LSTM variants, TCN, two-branch TCN (Las Vegas ↔ Phoenix) |

## Repository layout

```
model_training/
├── README.md, requirements.txt, .gitignore
├── dashboard/                     interactive prediction & model-selection app (see dashboard/README.md)
├── experiments/
│   ├── 01_campus_substation/      ┐
│   ├── 02_indiana_kentucky/       │ standard layout:
│   ├── 03_dayton_duquesne/        ┘   data/{raw,interim,processed}  notebooks/  models/  outputs/
│   ├── 04_atlanta_birmingham/     ┐ stage-based layout kept from the original project
│   └── 05_las_vegas_phoenix/      ┘   (one folder per pipeline stage; see below)
└── archive/
    ├── migration_log.csv          every move / delete / path edit made by the reorganisation
    ├── original_tree.txt          directory listing before the reorganisation
    └── ipynb_checkpoints/         old Jupyter autosaves that differed from the live files
```

Experiments 01–03 were flat folders and were given the standard layout. Experiments 04 and 05 were
already split into stage folders whose notebooks write into each other's directories (and whose
prepared arrays are several GB), so their internal layout is unchanged; only their paths were made
relative.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate     # or use an existing conda env
pip install -r requirements.txt
```

The original work used two Jupyter kernels: `btp_lstm_gpu` (TensorFlow, GPU) for training and
`experiments/04_atlanta_birmingham/cleaning/.venv` (pandas/h5py, no TensorFlow) for the 04 cleaning
notebooks. That `.venv` was moved with its folder: `Scripts\python.exe` and `Activate.ps1` work, the
`activate`/`activate.bat` paths were updated, but its `pip.exe`-style launchers still embed the old
path — use `python -m pip`, or recreate the venv.

## Energy Intelligence Dashboard

`dashboard/` is a Streamlit application. It predicts hourly consumption from manual or uploaded
weather with the trained two-branch TCN models (Atlanta → Birmingham, Las Vegas → Phoenix) and the
Dayton LSTM (Dayton, Pittsburgh), and it selects a model with an evidence-based Model Fitness Score.

```bash
cd dashboard
streamlit run app.py
python -m pytest tests -q
```

`dashboard/README.md` documents the architecture, the scoring formula, the held-out evidence and the
limitations. One of those limitations applies to the research data itself: load timestamps are UTC
while weather timestamps are local standard time.

## Running things

**Run every notebook with its own folder as the working directory** (the Jupyter / VS Code default).
All paths are relative to the notebook, e.g. `../data/processed/...` in experiments 01–03 and
`Path.cwd()` / `Path.cwd().parent` in 04–05. No absolute paths remain.

Scripts:

```bash
python experiments/01_campus_substation/scripts/lstm_energy_forecasting.py   # any cwd; trains, ~minutes
python experiments/05_las_vegas_phoenix/create_lstm_training_notebook.py     # regenerates a training notebook
```

The weather/energy extraction scripts in 04 and 05 (`extract_*_weather.py`, `date_change.py`,
`mastercsv_creation.py`, `cleaning/extract_city_energy.py`) are run from their own folder.

## Workflows

### 01 — campus substation (`data/` → `notebooks/` → `models/`, `outputs/`)

1. `clean_weather.ipynb`: `raw/weather.csv` → `interim/cleaned_weather.csv`
2. `raw/energy_data_SUB-S.csv` → `interim/sub_s_cleanedtime.csv`, `SUB_S_clean_withoutgrouping.csv`
   (done outside the notebooks; no code for this step is in the repo)
3. `weather_cleaning pt2.ipynb`: → `interim/hourly_energy.csv`, `interim/cleaned_data.csv`, `interim/weather_hourly.csv`
4. `fiktering.ipynb`: → `interim/hourly_energy_filtered.csv`, `processed/final_training_dataset.csv`
5. Analysis / baselines on the final dataset: `correlation_analysis.ipynb`, `correlation_analysis_final.ipynb`,
   `energy_mlr_baseline.ipynb`, `mlr_energy_model_filtered_energy.ipynb`, `polynomial_mlr_energy.ipynb`
   → `outputs/figures`, `outputs/tables`
6. `energy_feature_engineering.ipynb`: → `processed/final_feature_engineered_dataset.csv`
7. Models:
   - `lstm_energy_forecasting.ipynb` / `scripts/lstm_energy_forecasting.py` (same pipeline) → `models/best_lstm_model.keras`
   - `lstm_energy_forecasting_new.ipynb` (v1) and `lstm_energy_v2.ipynb` (v2) → `models/`, `models/checkpoints/`, `outputs/logs/`
   - `prophet_energy_forecasting.ipynb` → `outputs/prophet/`

`energy_data_BH3.csv` / `energy_data_BH4.csv` are raw building meters not used by any notebook.

### 02 — Indiana → Kentucky

1. `INDIANA_CLEANING.ipynb`: `raw/kentucky_weather_reconstructed.csv` → `interim/kentucky_weather_1962_1980.csv`
   → `processed/kentucky_weather_2005_2018.csv` (re-dated to the Indiana period)
2. `Indiana_LSTM_Energy_Forecast.ipynb`: Indiana load + weather → `models/`, `outputs/`
3. `Kentucky_Feature_Engineering.ipynb`, `Kentucky_LSTM_Evaluation.ipynb`: apply the model to Kentucky weather

`models/` is empty: the Indiana training run never finished (MemoryError in the saved outputs). The
Kentucky notebooks currently load the **campus** scaler/model from `01_campus_substation/models/`,
exactly as they did before the move — see *Known issues*.

### 03 — Dayton → Duquesne

1. `weather_parser_notebook.ipynb`: SAMSON `w93815.*` files in `data/raw/` → `interim/cleaned_weather_DAY.csv`
2. `energy cleaning and merging.ipynb`: yearly PJM `*_LOAD_HOURLY_<year>.csv` in `data/raw/` → `interim/cleaned_data_<zone>/` → `processed/FINAL_MERGED_<zone>_LOAD_2006_2016.csv`
3. `Date_change.ipynb`: → `processed/weather_<zone>_2006-2016.csv`
4. `Dayton_LSTM_Energy_Forecast.ipynb`: Dayton LSTM with EDA/correlation figures
5. `Dayton_to_Duquesne_LSTM_Transfer.ipynb` (was `Untitled.ipynb`): trains on Dayton, evaluates on Duquesne,
   draws the summary metric tables → `models/`, `outputs/`

`data/raw/` is empty: the raw yearly PJM files and SAMSON files are not in the repo.

### 04 — Atlanta ↔ Birmingham (stage folders)

| Stage | Folder | Contents |
|-------|--------|----------|
| Weather | `Atlanta/`, `Birmingham/` | `*_weather_raw/` SAMSON files → `extract_*_weather.py` → `date_change.py` (1983–90 → 2016–23) → `mastercsv_creation.py` → `*_master.csv` |
| Energy | `cleaning/` | `historic_load_hourly_2016_2023_county.h5` → `extract_city_energy.py` (also produced 05's energy CSVs) |
| EDA / prep | `cleaning/` | `Atlanta_Birmingham_Weather_Energy_EDA_Feature_Engineering.ipynb`, `LSTM_Data_Preparation_Atlanta_Birmingham.ipynb` |
| Similarity | `atlanta_birmingham_climate_similarity/` | analysis notebook + `tables/`, `plots/`, `report/` |
| LSTM | `LSTM_GPU_TRAINING/` | `ATLANTA/`, `BIRMINGHAM/` arrays + scalers; `LSTM_DATA_PREP/` (`data/` → `Prepare_LSTM_Data…` → `prepared_lstm/` → `feature_groups_and_weather_anomalies` → `LSTM_GPU_TRAINING_ALL_MODELS` → `training_results/`) |
| TCN | `neural_model/` | two-branch TCN training + visualisation → `atlanta_birmingham_two_branch_results/` |

### 05 — Las Vegas ↔ Phoenix (stage folders)

| Stage | Folder / notebook | Output |
|-------|-------------------|--------|
| Weather | `Las_Vegas_weather/`, `Phoenix_weather/` (extract + date_change) | `data/*_hourly_weather_2016_2023.csv` |
| Inputs | `data/` | hourly energy + weather CSVs used by everything below |
| Similarity | `climate_similarity_analysis/` | `tables/`, `plots/`, `report/` |
| EDA | `LV_Phoenix_Full_Weather_Energy_Analysis.ipynb` | `LV_Phoenix_Weather_Energy_Analysis/` |
| Preprocessing | `LSTM_LasVegas_Phoenix_Data_Preprocessing_BASELINE_DEVIATION.ipynb` (the executed one; plain/REBUILT are earlier variants) | `processed_lstm_baseline_deviation/` |
| Features | `Feature_Engineering_Cross_City_Weather_Energy.ipynb` | `processed_feature_candidates_cross_city_v1/` |
| LSTM | `training notebooks/Train_LSTM_Energy_Deviation…`, `Train_LSTM_Cross_City…` | `LSTM_GPU_TRAINING_RESULTS/`, `lstm_training_results_cross_city_v1/` |
| TCN | `tcn_models/TCN_data_preparation`, `TCN_training_gpu` | `data/tcn_prepared/`, `data/tcn_training/` |
| Two-branch TCN | `neural_models/two_branch_cross_city_tcn_training.ipynb` | `neural_models/two_branch_tcn_results/` |
| Generators | `create_*_notebook.py` | rewrite the feature-engineering / training notebooks |

## Known issues (pre-existing, not introduced by the reorganisation)

- **02**: Kentucky notebooks load `01_campus_substation/models/feature_scaler.pkl` (45 campus features)
  and fail against the 29 Kentucky features; the intended Indiana artifacts were never produced.
- **02**: the second code cell of `INDIANA_CLEANING.ipynb` reads `Indiana_consumption_2005_18.csv`, which
  never existed (the real file is `Indiana_energy_consumption_2005-18.csv`).
- **01**: `fiktering.ipynb` rewrites `interim/hourly_energy.csv` in place (columns are now
  `timestamp, energy_kwh.1`), so the older `correlation_analysis.ipynb` / `energy_mlr_baseline.ipynb`, which
  expect a `cleaned bucket` column, no longer match it (`correlation_analysis.ipynb` fails with `KeyError`). `correlation_analysis_final.ipynb` is the working version.
- **01**: the LSTM `.keras` files contain a `Lambda` attention layer saved without `output_shape`; Keras 3
  cannot rebuild it from the file alone — rebuild the model in the notebook and `load_weights` instead.
- **Two TensorFlow environments**: 01–03 models were saved with Keras 3 (zip `.keras`); 04–05 models were
  saved by the `btp_lstm_gpu` env (TF 2.10) and are really HDF5 despite the `.keras` name. Load each with
  the environment that wrote it.
- **03**: raw PJM yearly CSVs and SAMSON `w93815.*` files are missing, so steps 1–2 cannot be rerun.
- **04/05**: several notebooks were never executed or point at outputs that were later moved by hand
  (e.g. `LSTM_GPU_TRAINING/Birmingham_External_Test_Preparation_Atlanta_Scalers.ipynb` expects scalers in
  `cleaning/`; `LSTM_GPU_TRAINING/ATLANTA/LSTM_Training_Atlanta_to_Birmingham.ipynb` loads its `.npy` arrays from
  `LSTM_GPU_TRAINING/`, but they now sit in `ATLANTA/` and `BIRMINGHAM/`). Their relative structure was preserved as found.
- Generated reports/metadata (`*_climate_similarity_report.md/html`, `data/tcn_prepared/*/metadata.json`)
  still print the absolute paths that were current when they were generated; rerunning regenerates them.
- Deliberate duplicate inputs kept: `*_master.csv` copies in 04 (`Atlanta/`, `cleaning/`,
  `LSTM_GPU_TRAINING/ATLANTA/`, `LSTM_DATA_PREP/data/`) and the weather CSVs in 05 (`*_weather/` and `data/`)
  — each copy is the input folder of a different notebook or portable GPU bundle.
