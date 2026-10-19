import pandas as pd

INPUT_FILE = "atlanta_hourly_energy.csv"
OUTPUT_FILE = "atlanta_hourly_energy_no_feb29.csv"

df = pd.read_csv(INPUT_FILE)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Remove February 29 from every year
df = df[
    ~(
        (df["timestamp"].dt.month == 2) &
        (df["timestamp"].dt.day == 29)
    )
].copy()

# Save cleaned CSV
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Saved: {OUTPUT_FILE}")
print(f"Rows remaining: {len(df):,}")