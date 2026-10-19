# Cross-city electricity-demand forecasting results

## Experiment at a glance

| Item | Value |
|---|---|
| Source city for learned weather relationship | Las Vegas |
| Target city for transfer evaluation | Phoenix |
| Training period | 1 Jan 2016 – 31 Dec 2021 |
| Validation period | Calendar year 2022 |
| Final held-out test period | Calendar year 2023 |
| Forecast horizon | 1 hour ahead |
| Sequence lookback | 168 hours (7 days) |
| Test observations | 8,760 hourly forecasts |
| Recency weighting | Exponential, two-year half-life |
| Training objective | Mean squared error on scaled log residuals |

## Aim

Las Vegas and Phoenix have different absolute demand levels. Directly transferring a model trained to predict raw MW would therefore entangle city scale with weather response. The experiment separates the problem into:

```text
log(predicted load at t+1)
    = city-specific baseline at t+1
    + transferred weather/residual effect at t+1
```

The final load in MW is recovered as:

```text
predicted load = exp(predicted log load)
```

This lets Phoenix retain its own normal demand level, calendar profile, and long-term trend, while the Las Vegas-trained TCN supplies the learned short-term weather/residual adjustment.

## Data preparation and leakage controls

The raw hourly weather and electricity-demand CSVs for both cities cover 2016–2023. Weather timestamps were aligned to an hourly grid. Invalid physical readings were converted to missing values; weather was forward-filled only, so no future weather observation was used to fill an earlier hour.

The time split is chronological. No 2022 or 2023 target values were used to fit the baseline or TCN scalers/models. The final Phoenix test is strictly after the validation cutoff.

Recent data is weighted more heavily during training:

```text
weight = exp(-ln(2) × age_in_years / 2)
```

Thus, an observation two years older than the latest training observation has half its training weight. This accommodates gradual changes such as infrastructure expansion and demand-pattern drift without allowing future data into training.

## Layer 1 — city-specific baseline neural network

Two independent baseline MLPs were trained: one for Las Vegas and one for Phoenix. They use only calendar and trend variables—no weather variables—so they estimate each city's expected normal load.

### Baseline inputs (8)

- Hour-of-day sine and cosine
- Day-of-week sine and cosine
- Month sine and cosine
- Weekend flag
- Continuous years-since-start trend

### Baseline architecture

```text
8 inputs
  → Dense(64, ReLU)
  → Dropout(0.10)
  → Dense(32, ReLU)
  → Dense(1, float32): scaled log baseline
```

| Baseline model | Trainable parameters |
|---|---:|
| Las Vegas baseline MLP | 2,689 |
| Phoenix baseline MLP | 2,689 |

For every hourly record, the baseline output is converted back to log-load and used to calculate:

```text
residual_log = log(actual load) − log(city baseline load)
```

## Layer 2 — Las Vegas weather-residual TCN

The TCN is trained only with Las Vegas data from 2016–2021. Its target is the next-hour log residual, not the raw electricity demand. The model therefore learns the transferable short-term adjustment around the normal city-specific baseline.

### TCN input

Each forecast receives 168 previous hourly observations and 23 features per hour, giving an input shape of:

```text
(batch size, 168 hours, 23 features)
```

The 23 features are 17 weather-derived features and 6 causal residual-history features.

**Weather-derived features**

- Temperature, dew point, relative humidity, pressure
- Wind speed, wind-direction sine/cosine, visibility
- Precipitable water and hourly precipitation
- Total/opaque cloud cover
- Extraterrestrial horizontal/direct-normal radiation
- Cooling-degree and heating-degree variables
- Temperature × humidity interaction

**Residual-history features**

- Residual at the current observed hour and lags of 1, 24, and 168 hours
- 24-hour rolling residual mean and standard deviation

The residual-history terms make this a realistic one-hour-ahead operational forecast: the previous hour's actual load is known when predicting the next hour. They also mean the TCN is a **weather-plus-autoregressive-residual** model, rather than a weather-only model.

### TCN architecture

```text
Input: 168 × 23
  → 7 residual TCN blocks with dilation rates 1, 2, 4, 8, 16, 32, 64
      each block: causal Conv1D(64, kernel=3) → SpatialDropout(0.10)
                  → causal Conv1D(64, kernel=3) → SpatialDropout(0.10)
                  → residual skip connection → ReLU
  → GlobalAveragePooling1D
  → Dense(64, ReLU)
  → Dropout(0.20)
  → Dense(1, float32): next-hour scaled log residual
```

The causal convolutions ensure that each temporal representation uses only the present and earlier input hours. Dilation lets the model model short and longer delayed effects without using recurrent layers.

| TCN property | Value |
|---|---:|
| Residual blocks | 7 |
| Convolution filters | 64 |
| Kernel size | 3 |
| Dilation rates | 1, 2, 4, 8, 16, 32, 64 |
| Dropout | 0.10 in TCN blocks; 0.20 before output |
| Total/trainable parameters | 170,817 / 170,817 |

## Training process

The notebook verifies TensorFlow GPU execution with an explicit GPU matrix-multiplication probe before any model is trained. Mixed precision is enabled when configured in the GPU environment.

The TCN used Adam optimization, an initial learning rate of 0.001, learning-rate reduction on validation-loss plateau, early stopping, and best-model checkpointing. Training stopped after 41 epochs. The best validation checkpoint was at epoch 29:

| Measure at best validation epoch | Value |
|---|---:|
| Training loss | 0.01622 |
| Training MAE, scaled residual | 0.14001 |
| Validation loss | 0.04566 |
| Validation MAE, scaled residual | 0.14665 |
| Learning rate | 0.00025 |

## Cross-city reconstruction and final evaluation

For every Phoenix hour in 2023:

1. The Phoenix baseline MLP predicts Phoenix's normal log-load using calendar and trend inputs.
2. The Las Vegas-trained TCN predicts the next-hour weather/residual adjustment from Phoenix weather and previously observed Phoenix residual history.
3. The two outputs are added in log space.
4. The result is exponentiated to obtain predicted Phoenix demand in MW.

No Phoenix 2023 target values were used to train either baseline or the Las Vegas TCN. The evaluation covers the complete 2023 year.

| Final Phoenix 2023 result | Value |
|---|---:|
| MAE | **119.50 MW** |
| RMSE | **168.63 MW** |
| MAPE | **1.90%** |
| R² | **0.9925** |
| Number of forecasts | **8,760** |

## Interpretation

The 0.9925 R² and 1.90% MAPE show that the two-branch framework transferred very effectively from Las Vegas to Phoenix for this one-hour-ahead forecasting setup. The result should be presented as evidence that a weather/residual relationship learned in Las Vegas can be combined with Phoenix's separately calibrated structural baseline to forecast Phoenix demand accurately.

The correct claim is not that Phoenix demand is predicted from weather alone. Its calendar/trend baseline and observed prior residuals provide city-specific context; the cross-city component is the weather-residual TCN trained only on Las Vegas.

## Reproducibility artifacts

The result folder contains:

- `las_vegas/baseline_mlp.keras` and `phoenix/baseline_mlp.keras`
- City-specific baseline scalers
- `las_vegas_to_phoenix/best_weather_tcn.keras` and `final_weather_tcn.keras`
- Las Vegas-only TCN feature and residual scalers
- `weather_tcn_history.csv`
- `experiment_config.json`
- `phoenix_2023_cross_city_predictions.csv`
- `phoenix_2023_cross_city_metrics.json`
