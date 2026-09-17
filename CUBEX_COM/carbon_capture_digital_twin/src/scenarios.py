import yaml
import os

class ScenarioEngine:
    def __init__(self, process, sensor_model):
        self.process = process
        self.sensor_model = sensor_model
        self.current_scenario = None
        
    def load_scenario(self, name):
        # Allow passing the project root dir if running from different cwd
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "scenarios", f"{name}.yaml")
        
        if not os.path.exists(filepath):
            print(f"Warning: Scenario {name} not found at {filepath}")
            return
            
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            
        self.current_scenario = name
        self.apply_scenario_data(data)
        
    def apply_scenario_data(self, data):
        # Reset faults
        self.process.fault_fan_failure = False
        self.process.fault_low_flow = False
        self.process.fault_efficiency_drop = False
        
        for k in self.sensor_model.faults:
            self.sensor_model.faults[k] = {'drift': 0.0, 'spike': 0.0, 'dropout': False}
            
        if not data:
            return
            
        if 'process_faults' in data:
            pf = data['process_faults']
            self.process.fault_fan_failure = pf.get('fan_failure', False)
            self.process.fault_low_flow = pf.get('low_flow', False)
            self.process.fault_efficiency_drop = pf.get('efficiency_drop', False)
            
        if 'sensor_faults' in data:
            sf = data['sensor_faults']
            for sensor_key, faults in sf.items():
                if sensor_key in self.sensor_model.faults:
                    self.sensor_model.faults[sensor_key]['drift'] = faults.get('drift', 0.0)
                    self.sensor_model.faults[sensor_key]['spike'] = faults.get('spike', 0.0)
                    self.sensor_model.faults[sensor_key]['dropout'] = faults.get('dropout', False)
                    
        if 'initial_state' in data:
            istate = data['initial_state']
            if 'ambient_co2' in istate:
                self.process.ambient_co2 = istate['ambient_co2']
            if 'fan_command' in istate:
                self.process.commanded_fan_rpm = istate['fan_command']
                self.process.actual_fan_rpm = istate['fan_command']
