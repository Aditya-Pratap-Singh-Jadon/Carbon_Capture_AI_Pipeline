import time
import yaml
from datetime import datetime

from .process_model import ProcessModel
from .sensor_model import SensorModel
from .scenarios import ScenarioEngine
from .telemetry import TelemetryFormatter
from .serial_interface import SerialInterface
from .controller import Controller
from .logger import setup_logger
import os

class DigitalTwin:
    def __init__(self, mode='stdout', scenario='mixed', config_path='config.yaml', samples=50, labeled=False):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        import random
        random.seed(self.config['simulation'].get('random_seed', 42))
        
        self.process = ProcessModel(self.config)
        self.sensor = SensorModel(self.config)
        
        self.scenarios = ScenarioEngine(self.process, self.sensor)
        
        self.scenario = scenario
        self.samples = samples
        self.labeled = labeled
        
        if self.scenario != 'mixed':
            self.scenarios.load_scenario(self.scenario)
        else:
            self.scenarios.load_scenario('startup')
            
        self.telemetry = TelemetryFormatter()
        self.serial = SerialInterface(mode, self.config)
        self.controller = Controller(self.process)
        self.logger = setup_logger(self.config)
        
        self.sampling_rate = self.config['simulation']['sampling_rate_hz']
        self.running = False
        self.current_sample = 0
        
    def _update_mixed_scenario(self):
        if self.scenario != 'mixed':
            return
            
        if self.current_sample == 10:
            self.scenarios.load_scenario('normal')
        elif self.current_sample == 20:
            self.scenarios.load_scenario('fan_failure')
        elif self.current_sample == 30:
            self.scenarios.load_scenario('normal')
        elif self.current_sample == 40:
            self.scenarios.load_scenario('sensor_spike')
            
    def run(self):
        self.running = True
        sleep_time = 1.0 / self.sampling_rate
        
        print(f"Starting Digital Twin in {self.serial.mode} mode, scenario: {self.scenario}, samples: {self.samples}")
        
        try:
            while self.running and self.current_sample < self.samples:
                self._update_mixed_scenario()
                
                # 1. Check for incoming commands
                cmd = self.serial.read_command()
                if cmd:
                    self.controller.apply_command(cmd)
                    
                # 2. Step process
                self.process.step()
                true_state = self.process.get_state()
                
                # 3. Apply sensor model
                observed_state = self.sensor.apply(true_state)
                
                # 4. Generate Telemetry
                now = datetime.now()
                scenario_label = self.scenarios.current_scenario if self.labeled else None
                telemetry_str = self.telemetry.format_csv(now, observed_state, scenario_label)
                
                # 5. Send Telemetry
                self.serial.send(telemetry_str)
                
                # 6. Log True State (never transmit)
                log_msg = f"Scenario: {self.scenarios.current_scenario} | True: {true_state} | Cmd RPM: {self.process.commanded_fan_rpm}"
                self.logger.info(log_msg)
                
                self.current_sample += 1
                
                # 7. Wait
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        finally:
            self.serial.close()
