import h5py
import pandas as pd

file_path = "historic_load_hourly_2016_2023_county.h5"

counties = {
    "Las_Vegas": "p32003",
    "Phoenix": "p04013"
}

with h5py.File(file_path, "r") as f:

    # Read county IDs
    county_ids = [
        x.decode("utf-8") if isinstance(x, bytes) else x
        for x in f["data/axis0"][:]
    ]

    # Read timestamps
    timestamps = pd.to_datetime(
        f["data/axis1"][:],
        unit="ns"
    )

    values = f["data/block0_values"]

    for city, fips in counties.items():

        if fips not in county_ids:
            raise ValueError(
                f"{fips} for {city} was not found in the HDF5 dataset."
            )

        column_index = county_ids.index(fips)

        print(f"{city}:")
        print(f"  FIPS: {fips}")
        print(f"  Column: {column_index}")

        # Extract hourly electricity load
        load = values[:, column_index]

        # Create separate DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "hourly_load_mw": load
        })

        # Save separate CSV
        output_file = f"{city.lower()}_hourly_energy.csv"
        df.to_csv(output_file, index=False)

        print(f"  Rows: {len(df)}")
        print(f"  Start: {df['timestamp'].iloc[0]}")
        print(f"  End: {df['timestamp'].iloc[-1]}")
        print(f"  Missing load values: {df['hourly_load_mw'].isna().sum()}")
        print(f"  Saved: {output_file}")
        print()