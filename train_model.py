import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
    )


df = pd.read_csv("data/AI4I 2020 Predictive Maintenance Dataset.csv")
# drop unwanted colomns
df = df.drop(columns=[
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

df = pd.get_dummies(df, columns=["Type"], dtype=int)

# Seprate inputs and targets
x = df.drop( "Machine failure" , axis=1)
print(x.columns)
y = df["Machine failure"]

# Split Data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create Model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(x_train, y_train)

# predict
predictions = model.predict(x_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


print(f"Accuracy : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print(classification_report(y_test, predictions))

joblib.dump(model, "models//random_forest_v1.pkl")
print("Model saved successfully")
