import os
from app.predict import predict

WAITING = 0
CLAP_DETECTED = 1

class ClapFSM:
    def __init__(self):
        self.state = WAITING

    def run(self):
        for folder in ["data/clap", "data/noise"]:
            for file in sorted(os.listdir(folder)):
                if file.endswith(".wav"):
                    self.handle_file(os.path.join(folder, file))

    def handle_file(self, file_path):
        prediction = predict(file_path)

        if self.state == WAITING and prediction == 1:
            print(f"CLAP detected → {file_path}")
            self.state = CLAP_DETECTED
        else:
            self.state = WAITING
