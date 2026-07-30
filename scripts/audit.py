import os
import pandas as pd

dataset_path = "data_petrobras"

sensor_columns = [
    "P-PDG",
    "P-TPT",
    "T-TPT",
    "P-MON-CKP",
    "T-JUS-CKP",
    "P-JUS-CKGL",
    "T-JUS-CKGL",
    "QGL"
]

summary = {}

for sensor in sensor_columns:
    summary[sensor] = {
        "all_zero": 0,
        "all_nan": 0,
        "total_files": 0
    }

for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):

        if not file.endswith(".csv"):
            continue

        filepath = os.path.join(folder_path, file)

        df = pd.read_csv(filepath)

        for sensor in sensor_columns:

            summary[sensor]["total_files"] += 1

            if df[sensor].isna().all():
                summary[sensor]["all_nan"] += 1

            elif (df[sensor] == 0).all():
                summary[sensor]["all_zero"] += 1

print("\n========== SENSOR AUDIT ==========\n")

for sensor, stats in summary.items():
    print(f"{sensor}")
    print(f"  Total Files : {stats['total_files']}")
    print(f"  All Zero    : {stats['all_zero']}")
    print(f"  All NaN     : {stats['all_nan']}")
    print()