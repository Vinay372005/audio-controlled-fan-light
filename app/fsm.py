class ClapFSM:
    def __init__(self):
        self.state = "WAITING"
        self.fan_status = "OFF"
        self.light_status = "OFF"
        self.clap_count = 0

    def process_clap(self):
        """Process detected clap and toggle devices."""
        if self.state == "WAITING":
            self.state = "CLAP_DETECTED"
            self.clap_count += 1
            if self.clap_count % 2 == 1:
                self.fan_status = "ON" if self.fan_status == "OFF" else "OFF"
            else:
                self.light_status = "ON" if self.light_status == "OFF" else "OFF"
            
            print(f"🔊 Clap #{self.clap_count}: Fan={self.fan_status}, Light={self.light_status}")
        # Auto-reset after short delay in app.py

    def reset(self):
        """Reset to waiting after noise or timeout."""
        self.state = "WAITING"
