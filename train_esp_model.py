import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
    )

df = pd.read_csv("data2/features.csv", sep=";")
#remove unnecessary columns
df = df.drop(columns=["id", "esp_id"])

#Features/Targets
x = df.drop(columns = ["label"]) 
y = df["label"]  

#encoding targets ( giving them numerical value)
encoder = LabelEncoder()
y_encoder = encoder.fit_transform(y)

print("Label Mapping:")

for i, label in enumerate(encoder.classes_):
    print(f"{i} --> {label}")

#train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y_encoder,
    test_size=0.2,
    random_state=42,
    stratify=y_encoder
)
#training on 500 trees
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="weighted"
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)


print(f"Accuracy : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print(classification_report(y_test, predictions))

joblib.dump(model, "models/esp_model.pkl")
joblib.dump(encoder, "models/esp_label_encoder.pkl")

