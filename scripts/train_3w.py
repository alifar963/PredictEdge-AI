import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("3w_features.csv")

print(df.head())
X = df.drop("label", axis=1)

y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
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

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix\n")
print(cm)

 
joblib.dump(model, "models/3w_model.pkl")
print("\nModel saved successfully!")