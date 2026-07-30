import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score

df = pd.read_csv("data2/features.csv", sep=";")

# Keep esp_id for grouping, drop only "id"
groups = df["esp_id"]
df_model = df.drop(columns=["id", "esp_id"])

x = df_model.drop(columns=["label"])
y = df_model["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Label Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"{i} --> {label}")

# Group-based split: no esp_id appears in both train and test
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(x, y_encoded, groups=groups))

x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print(f"\nTrain size: {len(x_train)}, Test size: {len(x_test)}")
print(f"Unique ESPs in train: {groups.iloc[train_idx].nunique()}")
print(f"Unique ESPs in test: {groups.iloc[test_idx].nunique()}")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("\n=== GROUP-SPLIT RESULTS (honest, unseen-ESP evaluation) ===")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, x, y_encoded, groups=groups, cv=gkf, scoring='f1_weighted')
print("Group K-Fold F1 scores:", scores)
print("Mean:", scores.mean(), "Std:", scores.std())