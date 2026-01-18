import time

class ClapFSM:
    def __init__(self):
        self.state = "OFF"
        self.last_clap_time = 0
        self.clap_count = 0

    def update(self, signal):
        if signal == "CLAP":
            now = time.time()
            if now - self.last_clap_time < 1:
                self.clap_count += 1
            else:
                self.clap_count = 1

            self.last_clap_time = now

            if self.clap_count == 2:
                self.toggle()
                self.clap_count = 0

        return self.state

    def toggle(self):
        self.state = "ON" if self.state == "OFF" else "OFF"
