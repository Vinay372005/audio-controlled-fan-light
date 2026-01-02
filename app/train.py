# Training script (to be implemented)
import os

CLAP_DIR = "data/clap"
NOISE_DIR = "data/noise"

def check_data():
    clap_files = os.listdir(CLAP_DIR)
    noise_files = os.listdir(NOISE_DIR)

    print("Clap files:", len(clap_files))
    print("Noise files:", len(noise_files))

if __name__ == "__main__":
    check_data()
