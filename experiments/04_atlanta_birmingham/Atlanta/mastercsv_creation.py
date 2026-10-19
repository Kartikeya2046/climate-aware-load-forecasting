import pandas as pd

# ==============================
# FILE PATHS
# ==============================

energy_file = "atlanta_hourly_energy.csv"
weather_file = "atlanta_hourly_weather_2016_2023.csv"

output_file = "atlanta_master.csv"


# ==============================
# LOAD DATA
# ==============================

energy = pd.read_csv(energy_file)
weather = pd.read_csv(weather_file)

print("Energy columns:")
print(energy.columns.tolist())

print("\nWeather columns:")
print(weather.columns.tolist())


# ==============================
# CONVERT TIMESTAMPS
# ==============================

energy["timestamp"] = pd.to_datetime(
    energy["timestamp"],
    errors="coerce"
)

weather["timestamp"] = pd.to_datetime(
    weather["timestamp"],
    errors="coerce"
)


# ==============================
# REMOVE INVALID TIMESTAMPS
# ==============================

energy = energy.dropna(subset=["timestamp"])
weather = weather.dropna(subset=["timestamp"])


# ==============================
# REMOVE DUPLICATE TIMESTAMPS
# ==============================

energy = energy.drop_duplicates(
    subset=["timestamp"],
    keep="first"
)

weather = weather.drop_duplicates(
    subset=["timestamp"],
    keep="first"
)


# ==============================
# MERGE ON TIMESTAMP
# ==============================

master = pd.merge(
    energy,
    weather,
    on="timestamp",
    how="inner"
)


# ==============================
# SORT CHRONOLOGICALLY
# ==============================

master = master.sort_values("timestamp").reset_index(drop=True)


# ==============================
# SAVE MASTER CSV
# ==============================

master.to_csv(output_file, index=False)


# ==============================
# CHECK RESULT
# ==============================

print("\nMaster dataset shape:", master.shape)

print("\nFirst 5 rows:")
print(master.head())

print("\nTimestamp range:")
print(master["timestamp"].min(), "to", master["timestamp"].max())

print("\nDuplicate timestamps:",
      master["timestamp"].duplicated().sum())

print("\nMissing values:")
print(master.isna().sum())