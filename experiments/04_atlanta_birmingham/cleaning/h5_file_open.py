import h5py

file_path = "historic_load_hourly_2016_2023_county.h5"

with h5py.File(file_path, "r") as f:
    print("=== DATASET INFORMATION ===")

    for name in [
        "data/axis0",
        "data/axis1",
        "data/block0_items",
        "data/block0_values"
    ]:
        dset = f[name]

        print(f"\n{name}")
        print("Shape:", dset.shape)
        print("Dtype:", dset.dtype)

        try:
            print("First 10 values:")
            print(dset[:10])
        except Exception as e:
            print("Could not display values:", e)
            