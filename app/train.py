import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from features import load_dataset

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

def train_model():
    # Load data
    X, y = load_dataset("data")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Model accuracy:", acc)

    # Save model
    joblib.dump(model, "models/clap_detector.pkl")
    print("Model saved at models/clap_detector.pkl")


if __name__ == "__main__":
    train_model()
