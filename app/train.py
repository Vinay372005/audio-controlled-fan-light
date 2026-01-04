import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ==============================
# Paths
# ==============================
BASE_DIR = os.getcwd()                  # /content/audio-controlled-fan-light
DATA_CSV = os.path.join(BASE_DIR, "data", "features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "clap_detector.pkl")

# ==============================
# Load CSV
# ==============================
df = pd.read_csv(DATA_CSV)

# Remove non-feature columns (like filename)
feature_cols = [col for col in df.columns if col not in ["label", "filename"]]
X = df[feature_cols]
y = df["label"]

print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Train model
# ==============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# Evaluate
# ==============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy*100:.2f}%")

# ==============================
# Save model
# ==============================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")
