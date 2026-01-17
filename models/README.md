# Trained Model

This folder contains the trained machine learning model used for
clap vs noise classification.

## Model Details
- Algorithm: Random Forest Classifier
- Number of trees: 300
- Input features: 15 audio features
  - Zero Crossing Rate
  - RMS Energy
  - Spectral Centroid
  - Spectral Bandwidth
  - Spectral Roll-off
  - MFCC 1–10

## Output Labels
- 1 → Clap
- 0 → Noise

## Usage
The model is loaded using `joblib` inside `predict.py` and used
by the FSM logic to detect clap events.
