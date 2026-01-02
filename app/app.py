from fsm import ClapFSM

def main():
    print("Starting Clap Detection App...")
    fsm = ClapFSM()
    fsm.run()
    print("All files processed.")

if __name__ == "__main__":
    main()
