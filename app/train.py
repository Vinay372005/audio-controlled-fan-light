import os
from features import extract_features

CLAP_DIR = "data/clap"
NOISE_DIR = "data/noise"

X = []
y = []

def load_data():
    print("Preparing dataset...")
    print("Clap folder:", CLAP_DIR)
    print("Noise folder:", NOISE_DIR)

def train_model():
    print("Training module ready (model will be added later)")

if __name__ == "__main__":
    load_data()
    train_model()
