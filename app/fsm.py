# FSM logic for ON/OFF control
state = "OFF"

def toggle():
    global state
    state = "ON" if state == "OFF" else "OFF"
    return state

if __name__ == "__main__":
    print("Initial State:", state)
    print("After Clap:", toggle())
