import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

df = pd.read_csv("data2/features.csv", sep=";")
df = df.drop(columns=["id", "esp_id"])

x = df.drop(columns=["label"])
y = df["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Label Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"{i} --> {label}")

x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\nBefore SMOTE:", Counter(y_train))

smote = SMOTE(random_state=42, k_neighbors=3)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

print("After SMOTE:", Counter(y_train_res))

model_smote = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
model_smote.fit(x_train_res, y_train_res)

predictions = model_smote.predict(x_test)

print("\n=== RESULTS WITH SMOTE ===")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))