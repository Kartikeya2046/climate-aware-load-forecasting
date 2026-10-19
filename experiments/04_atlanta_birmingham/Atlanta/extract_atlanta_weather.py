from pathlib import Path
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDER = Path("atlanta_weather_raw")

WBAN = "13874"

START_YEAR = 1983
END_YEAR = 1990

OUTPUT_FILE = "atlanta_hourly_weather_1983_1990_corrected.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_numeric(value):
    """
    Convert a fixed-width numeric value to float.

    Missing numeric values are represented by '---' in the
    source files and are converted to NaN.
    """
    value = value.strip()

    if value == "" or value == "---":
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_field(line, start_col, end_col):
    """
    Extract a field using 1-based inclusive column numbers.

    The source format includes a one-character quality flag
    immediately after most numeric values, so this function
    extracts only the numeric portion.
    """
    return line[start_col - 1:end_col].strip()


# ============================================================
# PARSE ONE HOURLY FILE
# ============================================================

def parse_hourly_file(filepath):

    records = []

    print(f"Reading {filepath.name}")

    with open(filepath, "r", encoding="latin-1") as file:

        # First line is the station header
        header = file.readline()

        for line_number, line in enumerate(file, start=2):

            if not line.strip():
                continue

            # ------------------------------------------------
            # DATE
            # Columns 002-011
            # ------------------------------------------------

            date_text = line[1:11].strip()

            try:
                date = pd.to_datetime(
                    date_text,
                    format="%Y-%m-%d"
                )
            except ValueError:
                print(
                    f"Invalid date in {filepath.name}, "
                    f"line {line_number}: {date_text}"
                )
                continue

            # ------------------------------------------------
            # HOUR
            # Columns 012-014
            #
            # 01-24 = hourly observations
            # 25     = daily values
            # ------------------------------------------------

            hour_text = line[11:14].strip()

            try:
                hour = int(hour_text)
            except ValueError:
                continue

            if hour == 25:
                continue

            if hour < 1 or hour > 24:
                continue

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Every numeric field below has a one-character
            # flag immediately after the value.
            #
            # Example:
            #
            #   Columns 065-070 = temperature value + flag
            #   Actual numeric value = Columns 065-069
            #
            # The previous script included the flag character
            # in the numeric slice, causing float conversion
            # to fail and producing NaN for essentially every
            # weather variable.
            # ------------------------------------------------

            record = {

                "date": date,
                "hour": hour,

                # Radiation
                "extraterrestrial_horizontal_radiation_wh_m2":
                    parse_numeric(parse_field(line, 16, 20)),

                "extraterrestrial_direct_normal_radiation_wh_m2":
                    parse_numeric(parse_field(line, 23, 27)),

                "global_horizontal_radiation_wh_m2":
                    parse_numeric(parse_field(line, 30, 36)),

                "direct_normal_radiation_wh_m2":
                    parse_numeric(parse_field(line, 39, 45)),

                "diffuse_horizontal_radiation_wh_m2":
                    parse_numeric(parse_field(line, 48, 54)),

                # Sky cover
                "total_sky_cover_tenths":
                    parse_numeric(parse_field(line, 57, 58)),

                "opaque_sky_cover_tenths":
                    parse_numeric(parse_field(line, 61, 62)),

                # Temperature
                "temperature_c":
                    parse_numeric(parse_field(line, 65, 69)),

                "dew_point_c":
                    parse_numeric(parse_field(line, 72, 76)),

                # Humidity
                "relative_humidity_percent":
                    parse_numeric(parse_field(line, 79, 81)),

                # Pressure
                "station_pressure_kpa":
                    parse_numeric(parse_field(line, 84, 88)),

                # Wind
                "wind_direction_deg":
                    parse_numeric(parse_field(line, 91, 93)),

                "wind_speed_ms":
                    parse_numeric(parse_field(line, 96, 100)),

                # Visibility
                "horizontal_visibility_km":
                    parse_numeric(parse_field(line, 103, 108)),

                # Ceiling
                "ceiling_height_m":
                    parse_numeric(parse_field(line, 111, 116)),

                # Observation indicator
                # Columns 119-120 = indicator + flag
                "observation_indicator":
                    line[118:119].strip(),

                # Present weather
                # Columns 122-131 = weather code + flag
                "present_weather":
                    line[121:130].strip(),

                # Atmospheric water
                "precipitable_water_mm":
                    parse_numeric(parse_field(line, 133, 135)),

                # Aerosol
                "broadband_aerosol_optical_depth":
                    parse_numeric(parse_field(line, 138, 143)),

                # Snow
                "snow_depth_cm":
                    parse_numeric(parse_field(line, 146, 149)),

                "days_since_last_snowfall":
                    parse_numeric(parse_field(line, 152, 154)),

                # Hourly precipitation
                "hourly_precipitation_cm":
                    parse_numeric(parse_field(line, 157, 162)),

                # Daily values are present only on Hour 25,
                # so they are intentionally NOT extracted here.
            }

            records.append(record)

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():

    all_data = []

    print("=" * 70)
    print("ATLANTA WBAN 13874 WEATHER EXTRACTION")
    print("=" * 70)

    print(f"Years: {START_YEAR}-{END_YEAR}")
    print("Station: Atlanta, GA")
    print(f"WBAN: {WBAN}")
    print()

    # --------------------------------------------------------
    # Process exactly 1983-1990
    # --------------------------------------------------------

    for year in range(START_YEAR, END_YEAR + 1):

        year_suffix = str(year)[-2:]
        filename = f"w{WBAN}.h{year_suffix}"
        filepath = DATA_FOLDER / filename

        if not filepath.exists():
            print(f"WARNING: Missing file: {filename}")
            continue

        df_year = parse_hourly_file(filepath)

        print(f"  Rows extracted: {len(df_year):,}")

        all_data.append(df_year)

    if not all_data:
        raise FileNotFoundError(
            "No Atlanta hourly files were found."
        )

    # --------------------------------------------------------
    # Combine all years
    # --------------------------------------------------------

    weather = pd.concat(
        all_data,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Create timestamp
    #
    # Hour 1 = 00:00
    # Hour 2 = 01:00
    # ...
    # Hour 24 = 23:00
    # --------------------------------------------------------

    weather["timestamp"] = (
        weather["date"]
        + pd.to_timedelta(
            weather["hour"] - 1,
            unit="h"
        )
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    weather = weather.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    duplicate_count = weather["timestamp"].duplicated().sum()

    print()
    print(f"Duplicate timestamps: {duplicate_count}")

    weather = weather.drop_duplicates(
        subset="timestamp",
        keep="first"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Verify hourly records
    # --------------------------------------------------------

    print(
        f"Hour values present: "
        f"{sorted(weather['hour'].unique().tolist())}"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print()
    print("Missing values:")
    print(weather.isna().sum())

    # --------------------------------------------------------
    # Final dataset information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print(f"Total rows: {len(weather):,}")
    print(f"Start: {weather['timestamp'].min()}")
    print(f"End: {weather['timestamp'].max()}")

    # --------------------------------------------------------
    # Show usable weather-variable coverage
    # --------------------------------------------------------

    weather_columns = [
        "temperature_c",
        "dew_point_c",
        "relative_humidity_percent",
        "station_pressure_kpa",
        "wind_direction_deg",
        "wind_speed_ms",
        "horizontal_visibility_km",
        "ceiling_height_m",
        "hourly_precipitation_cm",
        "precipitable_water_mm",
        "total_sky_cover_tenths",
        "opaque_sky_cover_tenths",
        "global_horizontal_radiation_wh_m2",
        "direct_normal_radiation_wh_m2",
        "diffuse_horizontal_radiation_wh_m2",
        "extraterrestrial_horizontal_radiation_wh_m2",
        "extraterrestrial_direct_normal_radiation_wh_m2",
        "broadband_aerosol_optical_depth",
        "snow_depth_cm",
        "days_since_last_snowfall",
    ]

    print()
    print("Weather variable coverage:")
    print(
        (weather[weather_columns].notna().mean() * 100)
        .round(2)
        .sort_values(ascending=False)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    weather.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()