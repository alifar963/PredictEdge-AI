import os
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "data_petrobras"

total_recordings = 0

print("=" * 50)
print("PETROBRAS 3W DATASET SUMMARY")
print("=" * 50)

for folder in sorted(os.listdir(dataset_path)):
    
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        csv_files = [f for f in os.listdir(folder_path)
                     if f.endswith(".csv")]

        print(f"Folder {folder}: {len(csv_files)} recordings")

        total_recordings += len(csv_files)

print("\nTotal recordings:", total_recordings)
sample_file = os.path.join(dataset_path, "0", os.listdir(os.path.join(dataset_path, "0"))[0])

df = pd.read_csv(sample_file)

#print(df.head())
#print("\nShape:")
#print(df.shape)

#print("\nColumns:")
#print(df.columns)

#print("\nData Types:")
#print(df.dtypes)

#print("\nMissing Values:")
#print(df.isnull().sum())

#print("\nClass Distribution:")
#print(df["class"].value_counts())
#print(df[["P-PDG"]].describe())


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

for sensor in sensor_columns:

    plt.figure(figsize=(15,4))

    plt.plot(df[sensor])

    plt.title(sensor)

    plt.xlabel("Time")

    plt.ylabel(sensor)

    plt.grid(True)

    plt.show()

plt.figure(figsize=(15,3))

plt.plot(df["class"])

plt.title("Class Labels")

plt.xlabel("Time")

plt.ylabel("Class")

plt.grid(True)

plt.show()
print("\n========== SENSOR SUMMARY ==========\n")

for sensor in sensor_columns:
    print(f"\n{sensor}")
    print(df[sensor].describe())