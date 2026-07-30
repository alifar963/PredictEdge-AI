import os
import pandas as pd
dataset_path = "data_petrobras"
sensor_columns = [
    "P-TPT",
    "T-TPT",
    "P-MON-CKP",
    "T-JUS-CKP"
]
dataset = []
for folder in sorted(os.listdir(dataset_path)):

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    label = int(folder)

    for file in os.listdir(folder_path):

        if not file.endswith(".csv"):
            continue

        filepath = os.path.join(folder_path, file)

        df = pd.read_csv(filepath)
        features = {}

        for sensor in sensor_columns:

            features[f"{sensor}_mean"] = df[sensor].mean()
            features[f"{sensor}_std"] = df[sensor].std()
            features[f"{sensor}_min"] = df[sensor].min()
            features[f"{sensor}_max"] = df[sensor].max()
            features[f"{sensor}_median"] = df[sensor].median()
        features["label"] = label
        dataset.append(features)

feature_df = pd.DataFrame(dataset)

print(feature_df.head())

print()

print(feature_df.shape)
feature_df.to_csv("3w_features.csv", index=False)

print("\nFeature dataset saved successfully!")