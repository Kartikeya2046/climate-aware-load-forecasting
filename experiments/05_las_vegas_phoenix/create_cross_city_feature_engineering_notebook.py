"""Create an executable cross-city energy feature-engineering notebook.

The generated notebook reads the existing merged city feature tables and writes a
feature-candidate dataset plus an auditable feature manifest.  It deliberately
does not impute, scale, select features, or compute target-based baselines:
those operations belong inside each training split.
"""

from pathlib import Path
import nbformat as nbf


BASE = Path(__file__).resolve().parent
OUT = BASE / "Feature_Engineering_Cross_City_Weather_Energy.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


cells = [
    md(
        "# Cross-city weather → expected-energy feature engineering\n\n"
        "This notebook creates a broad, **leakage-safe candidate feature set** for Las Vegas and Phoenix. "
        "It is designed for subsequent feature selection in a training notebook, including a weather-transfer "
        "model and an operational model with historical-load features.\n\n"
        "**Important:** rows retain missing values produced by lags and rolling windows. Do not fill, scale, "
        "select, or target-encode features here; fit those operations on each training split only."
    ),
    code(
        "from pathlib import Path\n"
        "import json\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from pandas.tseries.holiday import USFederalHolidayCalendar\n\n"
        "BASE = Path.cwd()\n"
        "INPUTS = {\n"
        "    'las_vegas': BASE / 'processed_lstm_baseline_deviation' / 'las_vegas' / 'las_vegas_featured.csv',\n"
        "    'phoenix': BASE / 'processed_lstm_baseline_deviation' / 'phoenix' / 'phoenix_featured.csv',\n"
        "}\n"
        "OUTPUT_DIR = BASE / 'processed_feature_candidates_cross_city_v1'\n"
        "OUTPUT_DIR.mkdir(exist_ok=True)\n\n"
        "TARGET = 'hourly_load_mw'\n"
        "LAGS = (1, 3, 6, 12, 24, 48, 72)\n"
        "TARGET_LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)\n"
        "ROLL_WINDOWS = (3, 6, 12, 24, 72)\n"
        "print('Writing feature candidates to:', OUTPUT_DIR)"
    ),
    md(
        "## Feature policy\n\n"
        "- **Transfer core**: calendar, contemporaneous/forecast weather, deterministic solar proxies, and historical weather features.\n"
        "- **Operational-only**: historical-load lags and trailing summaries. They are valid only when past metered load is known at prediction time; exclude them from cold-start cross-city evaluation.\n"
        "- **Experimental/local**: visibility, cloud, aerosol, snow, and weather-code-related features. Keep them for later ablation, not as a presumed transferable core.\n"
        "- A 25-hour gap starts a new segment. No lag or rolling feature crosses a gap."
    ),
    code(
        "def add_feature(registry, frame, name, values, group, policy, description):\n"
        "    frame[name] = values\n"
        "    registry.append({'feature': name, 'group': group, 'policy': policy, 'description': description})\n\n"
        "def segmented_shift(frame, col, periods):\n"
        "    return frame.groupby('_segment_id', sort=False)[col].shift(periods)\n\n"
        "def segmented_rolling(frame, col, window, stat):\n"
        "    # shift first: the row at timestamp t never contributes to its own feature\n"
        "    prior = frame.groupby('_segment_id', sort=False)[col].shift(1)\n"
        "    grouped = prior.groupby(frame['_segment_id'], sort=False)\n"
        "    if stat == 'mean':\n"
        "        return grouped.transform(lambda s: s.rolling(window, min_periods=window).mean())\n"
        "    if stat == 'max':\n"
        "        return grouped.transform(lambda s: s.rolling(window, min_periods=window).max())\n"
        "    if stat == 'min':\n"
        "        return grouped.transform(lambda s: s.rolling(window, min_periods=window).min())\n"
        "    if stat == 'std':\n"
        "        return grouped.transform(lambda s: s.rolling(window, min_periods=window).std())\n"
        "    raise ValueError(stat)\n\n"
        "def build_features(city, path):\n"
        "    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)\n"
        "    df.insert(1, 'city', city)\n"
        "    gap = df['timestamp'].diff().ne(pd.Timedelta(hours=1))\n"
        "    gap.iloc[0] = True\n"
        "    df['_segment_id'] = gap.cumsum().astype('int16')\n"
        "    registry = []\n\n"
        "    # Preserve original columns as candidates and document their provenance.\n"
        "    for col in df.columns:\n"
        "        if col not in {'timestamp', 'city', TARGET, '_segment_id'}:\n"
        "            registry.append({'feature': col, 'group': 'existing_raw_or_engineered', 'policy': 'candidate',\n"
        "                             'description': 'Present in the existing merged feature table.'})\n\n"
        "    ts = df['timestamp']\n"
        "    hour = ts.dt.hour\n"
        "    dow = ts.dt.dayofweek\n"
        "    doy = ts.dt.dayofyear\n"
        "    add_feature(registry, df, 'year_candidate', ts.dt.year, 'calendar_optional', 'candidate', 'Year: useful for drift analysis; normally exclude for cross-city transfer.')\n"
        "    add_feature(registry, df, 'is_holiday_us_federal', ts.dt.normalize().isin(USFederalHolidayCalendar().holidays(ts.min(), ts.max())).astype('int8'), 'calendar', 'transfer_core', 'US federal holiday flag.')\n"
        "    add_feature(registry, df, 'is_business_hour', ((hour >= 8) & (hour < 18) & (dow < 5)).astype('int8'), 'calendar', 'transfer_core', 'Working-hour proxy.')\n"
        "    add_feature(registry, df, 'is_night', ((hour < 6) | (hour >= 22)).astype('int8'), 'calendar', 'transfer_core', 'Night-time proxy.')\n"
        "    add_feature(registry, df, 'week_of_year', ts.dt.isocalendar().week.astype('int16'), 'calendar_optional', 'candidate', 'Week-of-year; test against cyclic annual encoding.')\n\n"
        "    # Weather availability: only make features for columns that actually exist.\n"
        "    temp = df['temperature_c']\n"
        "    dew = df['dew_point_c']\n"
        "    rh = df['relative_humidity_percent']\n"
        "    wind = df['wind_speed_ms']\n"
        "    direction = np.deg2rad(df['wind_direction_deg'])\n"
        "    add_feature(registry, df, 'wind_direction_sin', np.sin(direction), 'wind', 'transfer_core', 'Circular wind-direction encoding.')\n"
        "    add_feature(registry, df, 'wind_direction_cos', np.cos(direction), 'wind', 'transfer_core', 'Circular wind-direction encoding.')\n"
        "    add_feature(registry, df, 'wind_u_component', -wind * np.sin(direction), 'wind', 'candidate', 'East-west wind component.')\n"
        "    add_feature(registry, df, 'wind_v_component', -wind * np.cos(direction), 'wind', 'candidate', 'North-south wind component.')\n"
        "    add_feature(registry, df, 'temperature_dewpoint_spread_c', temp - dew, 'thermal_moisture', 'transfer_core', 'Dryness / dew-point depression.')\n"
        "    add_feature(registry, df, 'temperature_humidity_interaction_v2', temp * rh, 'thermal_moisture', 'candidate', 'Temperature × relative humidity.')\n"
        "    add_feature(registry, df, 'temperature_dewpoint_interaction', temp * dew, 'thermal_moisture', 'candidate', 'Temperature × dew point.')\n"
        "    vapor_pressure = 6.11 * np.exp(5417.7530 * (1 / 273.16 - 1 / (273.15 + dew)))\n"
        "    add_feature(registry, df, 'humidex_c', temp + (5 / 9) * (vapor_pressure - 10), 'thermal_moisture', 'candidate', 'Humidity-adjusted apparent temperature (humidex).')\n"
        "    for base in (15, 18, 20, 22, 24):\n"
        "        add_feature(registry, df, f'cooling_degree_base_{base}c', (temp - base).clip(lower=0), 'thermal_nonlinear', 'transfer_core', f'Cooling degree hours at {base}°C.')\n"
        "        add_feature(registry, df, f'heating_degree_base_{base}c', (base - temp).clip(lower=0), 'thermal_nonlinear', 'transfer_core', f'Heating degree hours at {base}°C.')\n"
        "    add_feature(registry, df, 'temperature_cubed', temp ** 3, 'thermal_nonlinear', 'candidate', 'Cubic temperature term for linear models only.')\n"
        "    for col in ('hourly_precipitation_cm', 'ceiling_height_m', 'horizontal_visibility_km', 'broadband_aerosol_optical_depth', 'precipitable_water_mm'):\n"
        "        if col in df:\n"
        "            add_feature(registry, df, f'log1p_{col}', np.log1p(df[col].clip(lower=0)), 'transforms', 'candidate', f'Log transform of {col}.')\n"
        "    if 'hourly_precipitation_cm' in df:\n"
        "        add_feature(registry, df, 'is_precipitating', (df['hourly_precipitation_cm'] > 0).astype('int8'), 'weather_event', 'candidate', 'Nonzero precipitation event.')\n"
        "    if 'total_sky_cover_tenths' in df and 'opaque_sky_cover_tenths' in df:\n"
        "        add_feature(registry, df, 'cloud_opaque_fraction', df['opaque_sky_cover_tenths'] / 10, 'cloud_local', 'candidate', 'Opaque sky fraction.')\n"
        "        add_feature(registry, df, 'cloud_nonopaque_tenths', df['total_sky_cover_tenths'] - df['opaque_sky_cover_tenths'], 'cloud_local', 'candidate', 'Total minus opaque cloud cover.')\n\n"
        "    # Explicit calendar × weather interactions, justified by the prior hour/season diagnostics.\n"
        "    for name, base_values in {'temperature': temp, 'dewpoint': dew, 'humidex': df['humidex_c']}.items():\n"
        "        add_feature(registry, df, f'{name}_x_sin_hour', base_values * np.sin(2 * np.pi * hour / 24), 'time_weather_interaction', 'transfer_core', f'{name} × cyclic hour sine.')\n"
        "        add_feature(registry, df, f'{name}_x_cos_hour', base_values * np.cos(2 * np.pi * hour / 24), 'time_weather_interaction', 'transfer_core', f'{name} × cyclic hour cosine.')\n"
        "        add_feature(registry, df, f'{name}_x_weekend', base_values * (dow >= 5), 'time_weather_interaction', 'candidate', f'{name} × weekend.')\n"
        "    for base in (18, 22):\n"
        "        cooling = df[f'cooling_degree_base_{base}c']\n"
        "        add_feature(registry, df, f'cooling_degree_{base}c_x_sin_hour', cooling * np.sin(2 * np.pi * hour / 24), 'time_weather_interaction', 'candidate', f'Cooling degree {base}°C × hour sine.')\n"
        "        add_feature(registry, df, f'cooling_degree_{base}c_x_cos_hour', cooling * np.cos(2 * np.pi * hour / 24), 'time_weather_interaction', 'candidate', f'Cooling degree {base}°C × hour cosine.')\n\n"
        "    lag_weather = [c for c in ('temperature_c', 'dew_point_c', 'relative_humidity_percent', 'wind_speed_ms', 'station_pressure_kpa', 'precipitable_water_mm', 'hourly_precipitation_cm', 'total_sky_cover_tenths', 'opaque_sky_cover_tenths') if c in df]\n"
        "    for col in lag_weather:\n"
        "        for lag in LAGS:\n"
        "            add_feature(registry, df, f'{col}_lag_{lag}h', segmented_shift(df, col, lag), 'weather_lag', 'transfer_core', f'{col} observed {lag} hours earlier; never crosses a time gap.')\n"
        "        for window in ROLL_WINDOWS:\n"
        "            add_feature(registry, df, f'{col}_prior_{window}h_mean', segmented_rolling(df, col, window, 'mean'), 'weather_rolling', 'transfer_core', f'Prior {window}-hour mean of {col}.')\n"
        "        if col in {'temperature_c', 'dew_point_c', 'wind_speed_ms', 'hourly_precipitation_cm'}:\n"
        "            for window in (6, 24):\n"
        "                add_feature(registry, df, f'{col}_prior_{window}h_max', segmented_rolling(df, col, window, 'max'), 'weather_rolling', 'candidate', f'Prior {window}-hour maximum of {col}.')\n"
        "                add_feature(registry, df, f'{col}_prior_{window}h_min', segmented_rolling(df, col, window, 'min'), 'weather_rolling', 'candidate', f'Prior {window}-hour minimum of {col}.')\n"
        "    for lag in (1, 3, 6, 24):\n"
        "        add_feature(registry, df, f'temperature_change_{lag}h', temp - segmented_shift(df, 'temperature_c', lag), 'weather_change', 'transfer_core', f'Current temperature minus temperature {lag} hours earlier.')\n"
        "        add_feature(registry, df, f'dewpoint_change_{lag}h', dew - segmented_shift(df, 'dew_point_c', lag), 'weather_change', 'candidate', f'Current dew point minus dew point {lag} hours earlier.')\n\n"
        "    # Historical load is intentionally separate: valid operationally, excluded for cold-start transfer.\n"
        "    add_feature(registry, df, 'target_log1p', np.log1p(df[TARGET]), 'target', 'target_only', 'Optional target transform; do not include as an input.')\n"
        "    for lag in TARGET_LAGS:\n"
        "        add_feature(registry, df, f'load_lag_{lag}h', segmented_shift(df, TARGET, lag), 'load_history', 'operational_only', f'Load observed {lag} hours earlier; exclude from zero-history cross-city tests.')\n"
        "    for window in (3, 6, 24, 72, 168):\n"
        "        add_feature(registry, df, f'load_prior_{window}h_mean', segmented_rolling(df, TARGET, window, 'mean'), 'load_history', 'operational_only', f'Prior {window}-hour load mean.')\n"
        "    for window in (24, 168):\n"
        "        add_feature(registry, df, f'load_prior_{window}h_std', segmented_rolling(df, TARGET, window, 'std'), 'load_history', 'operational_only', f'Prior {window}-hour load variability.')\n\n"
        "    # Keep technical columns for audit but remove the segmentation key from modelling candidates.\n"
        "    manifest = pd.DataFrame(registry).drop_duplicates('feature', keep='last').sort_values(['policy', 'group', 'feature'])\n"
        "    return df, manifest\n"
    ),
    code(
        "all_manifests = []\n"
        "summary = []\n"
        "for city, path in INPUTS.items():\n"
        "    city_df, city_manifest = build_features(city, path)\n"
        "    # CSV only: portable across the existing GPU environment without optional engines.\n"
        "    output = OUTPUT_DIR / f'{city}_feature_candidates.csv'\n"
        "    city_df.to_csv(output, index=False)\n"
        "    storage_format = 'csv'\n"
        "    city_manifest['city'] = city\n"
        "    all_manifests.append(city_manifest)\n"
        "    summary.append({'city': city, 'rows': len(city_df), 'columns_including_targets_and_audit': len(city_df.columns), 'format': storage_format, 'output': str(output)})\n"
        "    print(city, city_df.shape, '->', output.name)\n\n"
        "manifest = pd.concat(all_manifests, ignore_index=True).drop_duplicates(['feature', 'policy', 'group']).sort_values(['policy', 'group', 'feature'])\n"
        "manifest.to_csv(OUTPUT_DIR / 'feature_manifest.csv', index=False)\n"
        "pd.DataFrame(summary).to_csv(OUTPUT_DIR / 'generation_summary.csv', index=False)\n"
        "display(pd.DataFrame(summary))\n"
        "display(manifest.groupby(['policy', 'group']).size().rename('feature_count').reset_index())"
    ),
    md(
        "## How to use this output in the training notebook\n\n"
        "1. Read a city parquet file and filter the manifest by `policy`.\n"
        "2. For cross-city transfer, start with `transfer_core`; exclude `load_history`, `target`, `year_candidate`, and local-only weather groups unless an ablation proves they help on held-out cities.\n"
        "3. Split by time/city first. Fit imputation, scaling, spline thresholds, PCA, target encoding, and feature selection only on the training partition.\n"
        "4. Compare calendar-only, transfer-core, transfer-core + candidate groups, and operational-only load-history additions.\n"
        "5. For LSTM/GRU, use raw sequential weather/calendar features; do not feed every hand-built lag as well unless an ablation shows value."
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3"}
nbf.write(nb, OUT)
print(OUT)
