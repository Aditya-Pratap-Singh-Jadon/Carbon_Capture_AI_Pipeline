import sys
import serial
import time
import os
from datetime import datetime

class SerialInterface:
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config['serial']
        self.ser = None
        self.csv_file = None
        
        if self.mode == 'serial':
            try:
                self.ser = serial.Serial(
                    port=self.config['default_port'],
                    baudrate=self.config['baud_rate'],
                    timeout=self.config['timeout']
                )
            except Exception as e:
                print(f"Error opening serial port {self.config['default_port']}: {e}")
                sys.exit(1)
        elif self.mode == 'csv':
            os.makedirs("logs", exist_ok=True)
            fname = datetime.now().strftime("logs/telemetry_%Y%m%d_%H%M%S.csv")
            self.csv_file = open(fname, "w")
            
    def send(self, data_str):
        line = data_str + "\n"
        if self.mode == 'stdout':
            sys.stdout.write(line)
            sys.stdout.flush()
        elif self.mode == 'serial' and self.ser:
            self.ser.write(line.encode('utf-8'))
        elif self.mode == 'csv' and self.csv_file:
            self.csv_file.write(line)
            self.csv_file.flush()
            
    def read_command(self):
        if self.mode == 'serial' and self.ser:
            if self.ser.in_waiting > 0:
                try:
                    return self.ser.readline().decode('utf-8').strip()
                except:
                    pass
        return None
        
    def close(self):
        if self.ser:
            self.ser.close()
        if self.csv_file:
            self.csv_file.close()
