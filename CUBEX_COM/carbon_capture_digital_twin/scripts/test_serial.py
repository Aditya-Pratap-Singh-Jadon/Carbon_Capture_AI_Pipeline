import argparse
import serial
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Test Serial Receiver")
    parser.add_argument('--port', type=str, default='COM11', help='COM port to listen on')
    parser.add_argument('--baud', type=int, default=9600, help='Baud rate')
    
    args = parser.parse_args()
    
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
        print(f"Listening on {args.port} at {args.baud} baud...")
        print("Expected fields: timestamp,CO2_ppm,Temperature_C,Humidity_percent,Gas_Flow_L_min,Captured_CO2_g,Fan_Speed_RPM")
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    fields = line.split(',')
                    if len(fields) == 7:
                        print(f"Received valid telemetry: {fields}")
                    else:
                        print(f"Received invalid telemetry (len {len(fields)}): {line}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped listening.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
