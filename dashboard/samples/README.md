# Test inputs

`kirkland_test_weather.csv` contains **synthetic** weather for a fictional city, "Kirk land". It is not
real observations. It was derived from the recorded Phoenix weather for 2023-07-01 → 2023-07-14
(`experiments/05_las_vegas_phoenix/data/phoenix_hourly_weather_2016_2023.csv`) as follows:

- **Temperature:** Phoenix − 3 °C
- **Relative humidity:** Phoenix + 10 %, capped at 100
- **Dew point:** recomputed from temperature and humidity with the Magnus formula
- **Wind speed:** Phoenix × 0.9
- **Precipitable water:** Phoenix × 1.1
- **Pressure, wind direction, visibility, precipitation and sky cover:** unchanged

The file has 336 hourly rows in local standard time and no load column. The headers use common
aliases (`temp`, `humidity`, `dew point`, …), and it includes an extra `station` column that the
dashboard should ignore.

It exercises the any-city (climate analog) workflow. Leave Location on "Any city — match by climate",
optionally name it "Kirk land", keep a timeframe inside 2023-07-08 → 2023-07-14 (the default
2023-07-10 → 07-12 works) in forecast mode, and upload the file. The dashboard should:

- auto-map 12 columns and ignore `station`
- flag 5 out-of-range visibility values from the recorded Phoenix data, which require confirmation
- rank Phoenix as the most similar climate (≈ 64/100; every other city ≈ 42–46)
- predict with TCN Las Vegas using the Phoenix baseline (≈ 8,000 MW average for 07-10 → 07-12)

## `maple_falls_test_weather_load.csv` (synthetic, weather + load)

A fictional "Maple Falls", derived from recorded **Pittsburgh** data for 2015-07-01 → 2015-07-14
(336 hourly rows, local standard time). It includes a load column, so the Dayton LSTM can run.

- **Temperature:** Pittsburgh − 1.5 °C, written in **°F**, which tests the unit suggestion and conversion
- **Relative humidity:** + 5 %, capped at 100
- **Dew point:** recomputed from temperature and humidity with the Magnus formula
- **Wind speed:** × 1.1
- **Load (`demand_mw`):** Pittsburgh load × 0.8, for a smaller city
- **Pressure, wind direction, visibility and precipitation:** unchanged

Set the timeframe to 2015-07-08 → 2015-07-10 (the LSTM needs 24 h of load before the start). Expected:

- Temperature unit pre-selected as °F
- Ranking: Pittsburgh 71.6, Dayton 59.6, Atlanta 55.0, Birmingham 46.8, Phoenix 39.6, Las Vegas 38.0
- Model: Dayton LSTM on the Pittsburgh analog
  - one hour ahead: observed MAPE ≈ 6.4 %
  - forecast: ≈ 15.9 %

## `cedar_ridge_test_weather.csv` (synthetic, weather only)

A fictional "Cedar Ridge", derived from recorded **Birmingham** weather for 2023-07-01 → 2023-07-14
(336 hourly rows, no load).

- **Temperature:** Birmingham + 1 °C
- **Relative humidity:** − 5 %
- **Dew point:** recomputed with the Magnus formula
- **Wind speed:** × 1.2
- **Precipitable water:** × 0.95
- **Other variables:** unchanged

It uses canonical headers, plus an extra `site` column that should be ignored.

Use the default timeframe 2023-07-10 → 2023-07-12 in forecast mode. Expected:

- Ranking: Birmingham 62.0, Atlanta 57.9, Dayton 49.9, Pittsburgh 49.1, Phoenix 42.0, Las Vegas 40.8
- Model: TCN Atlanta with the Birmingham baseline, averaging ≈ 1,500 MW
- In one-hour-ahead mode, no model is eligible because there is no load. This is expected: the dashboard
  explains why and reports Birmingham as the closest climate.
