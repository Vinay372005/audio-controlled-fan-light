import os
from predict import predict

# FSM States
WAITING = 0
CLAP_DETECTED = 1

class ClapFSM:
    def __init__(self, clap_folder="data/clap", noise_folder="data/noise"):
        self.state = WAITING
        self.clap_folder = clap_folder
        self.noise_folder = noise_folder

    def run(self):
        # Process clap files
        for file_name in sorted(os.listdir(self.clap_folder)):
            file_path = os.path.join(self.clap_folder, file_name)
            self.handle_file(file_path)

        # Process noise files
        for file_name in sorted(os.listdir(self.noise_folder)):
            file_path = os.path.join(self.noise_folder, file_name)
            self.handle_file(file_path)

    def handle_file(self, file_path):
        prediction = predict(file_path)

        if self.state == WAITING:
            if prediction == "clap":
                print(f"CLAP detected! ({file_path})")
                self.state = CLAP_DETECTED
        elif self.state == CLAP_DETECTED:
            # Reset state for next detection
            self.state = WAITING


if __name__ == "__main__":
    fsm = ClapFSM()
    fsm.run()
