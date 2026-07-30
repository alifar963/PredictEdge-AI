import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data2/features.csv", sep=";")

# Add ratio/engineered features BEFORE splitting
df["ratio_peak1_peak2"] = df["peak1x"] / (df["peak2x"] + 1e-6)
df["ratio_rms_median98"] = df["rms(98,102)"] / (df["median(98,102)"] + 1e-6)
df["ratio_median813_median98102"] = df["median(8,13)"] / (df["median(98,102)"] + 1e-6)

groups = df["esp_id"]
df_model = df.drop(columns=["id", "esp_id"])

x = df_model.drop(columns=["label"])
y = df_model["label"]

print("Features used:", x.columns.tolist())

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nLabel Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"{i} --> {label}")

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(x, y_encoded, groups=groups))

x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("\n=== RESULTS WITH RATIO FEATURES (group split) ===")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

print("\nFeature Importances:")
for name, importance in sorted(zip(x.columns, model.feature_importances_), key=lambda t: -t[1]):
    print(f"{name}: {importance:.4f}")