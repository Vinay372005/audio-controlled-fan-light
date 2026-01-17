import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

df = pd.read_csv("data/features.csv")

# REMOVE filename column
X = df.drop(["label", "filename"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/clap_detector.pkl")

print("Model accuracy:", accuracy)
print("Model saved to models/clap_detector.pkl")
