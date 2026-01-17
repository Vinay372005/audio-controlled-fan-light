# Audio Controlled Fan and Light System

## 1. Introduction
This project implements an audio-based control system that
detects clap sounds and distinguishes them from environmental noise.
The system simulates fan and light control using a finite state machine (FSM).

## 2. Dataset
- 200 audio samples
  - 100 clap sounds
  - 100 noise sounds
- Audio format: WAV
- Duration: ~2 seconds per sample

## 3. Feature Extraction
The following features are extracted using Librosa:
- Zero Crossing Rate
- RMS Energy
- Spectral Centroid
- Spectral Bandwidth
- Spectral Roll-off
- MFCC (1–10)

These features effectively represent both temporal and spectral
characteristics of sound.

## 4. Machine Learning Model
- Algorithm: Random Forest Classifier
- Training/Test split: 80/20
- Output classes:
  - 0 → Noise
  - 1 → Clap

## 5. Finite State Machine (FSM)
The FSM has two states:
- WAITING
- CLAP_DETECTED

On detecting a clap, the system transitions to CLAP_DETECTED
and simulates a control action.

## 6. Results
The trained model achieves high accuracy in distinguishing
clap sounds from noise, validating the effectiveness of the
selected features and model.

## 7. Conclusion
The project successfully demonstrates an audio-controlled
system using signal processing and machine learning techniques.
This approach can be extended to real-time embedded systems.
