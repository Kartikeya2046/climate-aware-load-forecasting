import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "birmingham_hourly_weather_1983_1990.csv"

OUTPUT_FILE = "birmingham_hourly_weather_2016_2023.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])
df["timestamp"] = pd.to_datetime(df["timestamp"])


# ============================================================
# DELETE ALL FEBRUARY 29 ROWS
# ============================================================

df = df[
    ~(
        (df["date"].dt.month == 2) &
        (df["date"].dt.day == 29)
    )
].copy()


# ============================================================
# CHANGE YEARS
#
# 1983 -> 2016
# 1984 -> 2017
# ...
# 1990 -> 2023
# ============================================================

year_mapping = {
    1983: 2016,
    1984: 2017,
    1985: 2018,
    1986: 2019,
    1987: 2020,
    1988: 2021,
    1989: 2022,
    1990: 2023
}


# Change the year in the date column
df["date"] = df["date"].apply(
    lambda x: x.replace(
        year=year_mapping[x.year]
    )
)


# Change the year in the timestamp column
df["timestamp"] = df["timestamp"].apply(
    lambda x: x.replace(
        year=year_mapping[x.year]
    )
)


# ============================================================
# SORT BY TIMESTAMP
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CHECK
# ============================================================

print("Conversion complete.")
print(f"Rows: {len(df):,}")
print(f"Start: {df['timestamp'].min()}")
print(f"End:   {df['timestamp'].max()}")
print(f"Saved: {OUTPUT_FILE}")