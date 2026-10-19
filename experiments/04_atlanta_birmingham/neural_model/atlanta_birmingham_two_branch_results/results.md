# Atlanta to Birmingham two-branch TCN: results report

## Executive result

An Atlanta-trained weather-residual TCN was transferred to Birmingham and evaluated on the completely held-out 2023 period. Birmingham's own structural baseline was modeled separately, then combined with the transferred residual prediction.

| Metric | Birmingham 2023 result |
|---|---:|
| MAE | **12.60 MW** |
| RMSE | **17.18 MW** |
| MAPE | **1.01%** |
| R² | **0.9955** |
| Forecasts | **8,760 hourly observations** |

## Objective

Atlanta and Birmingham have different normal demand levels and structural patterns. Predicting raw MW directly across cities would mix those differences with the effect of weather. The model therefore separates demand into a city-specific normal level and a transferable short-term adjustment:

```text
log(predicted demand at t+1)
  = log(city-specific baseline at t+1)
  + predicted weather/residual effect at t+1
```

The combined log prediction is exponentiated to produce final MW demand.

## Experiment design

| Stage | Period | Role |
|---|---|---|
| Training | 1 Jan 2016 – 31 Dec 2021 | Fit Atlanta residual TCN and the two city baseline MLPs |
| Validation | Calendar year 2022 | Select TCN epoch and tune learning rate through validation loss |
| Test | Calendar year 2023 | Final Atlanta-to-Birmingham cross-city evaluation |
| Forecast horizon | 1 hour ahead | Predict the next hourly load |
| Lookback | 168 hours | Previous 7 days of inputs for each TCN prediction |

No Birmingham 2023 target values were used to fit a baseline model, a scaler, or the Atlanta TCN. The saved notebook also verifies this chronology before reporting metrics.

### More weight for recent years

The eight-year dataset may contain demand drift from infrastructure, population, and consumption changes. Training examples are weighted with an exponential two-year half-life:

```text
weight = exp(-ln(2) × age in years / 2)
```

An observation two years older than the latest training point therefore receives half the training weight. This weighting is applied to both the baseline MLP and the Atlanta TCN training data.

## Climate-informed weather selection

The Atlanta–Birmingham similarity analysis classified the pair as moderately similar (overall score: 62.49/100). The TCN deliberately uses the variables with the strongest cross-city similarity:

| Retained core weather variable | Similarity score |
|---|---:|
| Temperature | 98.40 / 100 |
| Precipitation | 94.70 / 100 |
| Dew point | 93.18 / 100 |
| Relative humidity | 85.10 / 100 |

Pressure and visibility were excluded because their similarity was weak (8.73 and 46.35 respectively). The model additionally derives cooling-degree temperature, heating-degree temperature, and a temperature × humidity interaction.

## Data quality and leakage controls

- The two master CSVs were restored to a continuous hourly timeline.
- Invalid weather readings were changed to missing values using physical bounds.
- Weather gaps use forward-fill only, never a future observation.
- Missing load targets are never fabricated; any sequence affected by an unavailable target/input is omitted.
- Calendar splits are chronological; no random train/test split is used.
- TCN scalers are fitted on Atlanta training data only and then applied unchanged to Birmingham.
- Every 2023 Birmingham target is after the 2022 validation cutoff.

## Layer 1: city-specific structural baseline MLP

Atlanta and Birmingham each have an independently trained baseline model. The baseline learns normal city demand without weather, using calendar and long-run trend features only.

### Baseline inputs

- Hour-of-day sine and cosine
- Day-of-week sine and cosine
- Month sine and cosine
- Weekend flag
- Continuous years-since-start trend

### Architecture

```text
8 calendar/trend features
  → Dense(64, ReLU)
  → Dropout(0.10)
  → Dense(32, ReLU)
  → Dense(1): scaled log baseline demand
```

| Model | Parameters |
|---|---:|
| Atlanta baseline MLP | 2,689 |
| Birmingham baseline MLP | 2,689 |

For every hour, the baseline produces expected log demand. The model residual is then:

```text
log residual = log(actual load) − log(predicted city baseline)
```

## Layer 2: Atlanta weather-residual TCN

The TCN is trained only on Atlanta data. Its target is the next-hour log residual rather than raw MW, which makes the learned response more transferable to Birmingham.

### TCN input

Each sample has shape:

```text
(batch, 168 hours, 13 features)
```

The 13 features comprise seven weather-derived variables and six causal residual-history variables.

**Weather-derived features (7)**

- Temperature, dew point, relative humidity, hourly precipitation
- Cooling-degree and heating-degree temperature
- Temperature × humidity interaction

**Causal residual history (6)**

- Current observed residual and 1-hour, 24-hour, and 168-hour lags
- Previous 24-hour residual mean and standard deviation

The prior-residual inputs are available at prediction time because this is a one-hour-ahead forecast. Consequently, the result represents **weather plus autoregressive residual transfer**, not weather alone.

### TCN architecture and operation

```text
Input: 168 × 13
  → Residual TCN block, dilation 1
  → Residual TCN block, dilation 2
  → Residual TCN block, dilation 4
  → Residual TCN block, dilation 8
  → Residual TCN block, dilation 16
  → Residual TCN block, dilation 32
  → Residual TCN block, dilation 64
  → GlobalAveragePooling1D
  → Dense(64, ReLU)
  → Dropout(0.20)
  → Dense(1): next-hour scaled log residual
```

Every TCN block contains two causal `Conv1D` layers with 64 filters, kernel size 3, and `SpatialDropout1D(0.10)`, followed by a residual skip connection and ReLU activation. Causal convolution prevents the model from seeing later hours, while increasing dilation lets it capture both immediate and multi-day weather effects.

| TCN property | Value |
|---|---:|
| Residual blocks | 7 |
| Dilation rates | 1, 2, 4, 8, 16, 32, 64 |
| Filters per convolution | 64 |
| Kernel size | 3 |
| Input shape | 168 × 13 |
| TCN trainable parameters | **168,257** |

## GPU training and model selection

The notebook checks TensorFlow, CUDA/GPU visibility, and executes a GPU matrix-multiplication probe before training. The model is explicitly constructed on GPU:0; CPU is used only for normal file/data-pipeline operations.

The Atlanta TCN used Adam optimization with initial learning rate 0.001, mean-squared-error loss on scaled residuals, validation-based learning-rate reduction, early stopping, and best-validation checkpointing.

Training stopped after 41 epochs. The best saved validation checkpoint was epoch 29:

| Best validation checkpoint measure | Value |
|---|---:|
| Training loss | 0.01397 |
| Training MAE, scaled residual | 0.13639 |
| Validation loss | 0.02346 |
| Validation MAE, scaled residual | 0.11065 |
| Learning rate | 0.00050 |

## Cross-city reconstruction

For each Birmingham hour in 2023:

1. The Birmingham baseline MLP estimates Birmingham's normal log demand from calendar and trend inputs.
2. The Atlanta-trained TCN receives Birmingham's selected weather history and observed prior residual history.
3. The TCN predicts Birmingham's next-hour log residual using the weather/residual relationship learned in Atlanta.
4. The predicted residual is added to the Birmingham baseline in log space.
5. The result is exponentiated to obtain predicted Birmingham MW demand.

## Final interpretation

The final 1.01% MAPE and 0.9955 R² demonstrate very accurate one-hour-ahead transfer under this experiment design. The defensible conclusion is that the Atlanta-trained weather/residual relationship transferred successfully when paired with Birmingham's separately calibrated structural baseline.

This should not be described as weather-only prediction: Birmingham's own calendar/trend baseline and previously observed load residuals provide essential city-specific context. The cross-city component is the TCN relationship learned from Atlanta only.

## Saved reproducibility artifacts

- `atlanta/baseline_mlp.keras` and `birmingham/baseline_mlp.keras`
- Baseline feature and target scalers for both cities
- `atlanta_to_birmingham/best_atlanta_weather_tcn.keras`
- `atlanta_to_birmingham/final_atlanta_weather_tcn.keras`
- Atlanta TCN feature/residual scalers
- `atlanta_weather_tcn_history.csv`
- `experiment_config.json`
- `birmingham_2023_cross_city_predictions.csv`
- `birmingham_2023_cross_city_metrics.json`
