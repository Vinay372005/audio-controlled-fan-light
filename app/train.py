import os
import pandas as pd
from features import extract_features

DATASET_PATH = "data"
OUTPUT_CSV = "features.csv"

rows = []

for label, folder in enumerate(["noise", "clap"]):
    folder_path = os.path.join(DATASET_PATH, folder)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path)
            rows.append(list(features) + [label])

columns = [f"mfcc{i}" for i in range(1,14)] + [
    "zcr", "rms", "centroid", "bandwidth", "rolloff", "label"
]

df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("✅ features.csv created")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

df = pd.read_csv("features.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "models/clap_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model saved")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Noise", "Clap"],
            yticklabels=["Noise", "Clap"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("plots/confusion_matrix.png")
plt.show()

