import serial.tools.list_ports

def main():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No COM ports found.")
        return
        
    for port, desc, hwid in sorted(ports):
        print(f"{port}: {desc} [{hwid}]")

if __name__ == '__main__':
    main()
