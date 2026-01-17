AUDIO CONTROLLED FAN AND LIGHT SYSTEM
===================================

This project implements an audio-based control system that detects
clap sounds and distinguishes them from background noise using
signal processing and machine learning techniques.

The system simulates fan and light control using a finite state
machine (FSM) and a trained Random Forest classifier.


PROJECT STRUCTURE
-----------------
app/
  app.py        - Main application entry point
  features.py   - Audio feature extraction
  train.py      - Model training script
  predict.py    - Prediction using trained model
  fsm.py        - Finite State Machine logic

data/
  clap/         - Clap audio samples (WAV)
  noise/        - Noise audio samples (WAV)
  features.csv  - Extracted audio features

models/
  clap_detector.pkl  - Trained Random Forest model
  README.md          - Model description

plots/
  feature_importance.png  - Feature importance visualization
  README.md               - Plot descriptions

docs/
  project_report.md  - Detailed project documentation


FEATURE EXTRACTION
------------------
The following audio features are extracted using Librosa:

1. Zero Crossing Rate
2. RMS Energy
3. Spectral Centroid
4. Spectral Bandwidth
5. Spectral Roll-off
6. MFCC 1 to MFCC 10

These features capture both temporal and spectral
characteristics of audio signals.


MACHINE LEARNING MODEL
----------------------
Algorithm      : Random Forest Classifier
Training Split : 80% Training, 20% Testing
Classes        :
  0 -> Noise
  1 -> Clap


FINITE STATE MACHINE (FSM)
--------------------------
States:
- WAITING
- CLAP_DETECTED

The system transitions to CLAP_DETECTED when a clap sound
is detected and then resets for the next detection.


HOW TO RUN
----------
1. Extract features:
   python app/features.py

2. Train the model:
   python app/train.py

3. Run clap detection:
   python app/app.py


RESULT
------
The system successfully detects clap sounds and ignores
background noise, demonstrating an effective audio-controlled
automation approach.


AUTHOR
------
Vinay Ayappagari
IIT Tirupati
B.Tech – Electrical Engineering
