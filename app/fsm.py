import os
from app.predict import predict

# FSM States
WAITING = 0
CLAP_DETECTED = 1

class ClapFSM:
    def __init__(self):
        # Lock base directory (project root)
        self.base_dir = os.getcwd()  # /content/audio-controlled-fan-light

        self.clap_folder = os.path.join(self.base_dir, "data", "clap")
        self.noise_folder = os.path.join(self.base_dir, "data", "noise")

        self.state = WAITING

    def run(self):
        print("FSM started...\n")

        # Process clap files
        print("Processing CLAP files...")
        for file_name in sorted(os.listdir(self.clap_folder)):
            if file_name.endswith(".wav"):
                file_path = os.path.join(self.clap_folder, file_name)
                self.handle_file(file_path)

        # Process noise files
        print("\nProcessing NOISE files...")
        for file_name in sorted(os.listdir(self.noise_folder)):
            if file_name.endswith(".wav"):
                file_path = os.path.join(self.noise_folder, file_name)
                self.handle_file(file_path)

        print("\nFSM finished processing all files.")

    def handle_file(self, file_path):
        prediction = predict(file_path)

        if self.state == WAITING:
            if prediction == "clap":
                print(f"👉 CLAP detected → FAN/LIGHT TOGGLED ({os.path.basename(file_path)})")
                self.state = CLAP_DETECTED

        elif self.state == CLAP_DETECTED:
            # Reset state after action
            self.state = WAITING


if __name__ == "__main__":
    fsm = ClapFSM()
    fsm.run()
