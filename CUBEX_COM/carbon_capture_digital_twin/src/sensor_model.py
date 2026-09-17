import random

class SensorModel:
    def __init__(self, config):
        self.config = config
        self.noise = config['sensor_noise']
        
        # Fault states (per sensor)
        self.faults = {
            'co2_ppm': {'drift': 0.0, 'spike': 0.0, 'dropout': False},
            'temperature_c': {'drift': 0.0, 'spike': 0.0, 'dropout': False},
            'humidity_percent': {'drift': 0.0, 'spike': 0.0, 'dropout': False},
            'gas_flow_l_min': {'drift': 0.0, 'spike': 0.0, 'dropout': False},
            'captured_co2_g': {'drift': 0.0, 'spike': 0.0, 'dropout': False},
            'fan_speed_rpm': {'drift': 0.0, 'spike': 0.0, 'dropout': False}
        }
        
    def apply(self, true_state):
        observed = {}
        
        std_map = {
            'co2_ppm': self.noise['co2_noise_std'],
            'temperature_c': self.noise['temperature_noise_std'],
            'humidity_percent': self.noise['humidity_noise_std'],
            'gas_flow_l_min': self.noise['flow_noise_std'],
            'captured_co2_g': self.noise['captured_co2_noise_std'],
            'fan_speed_rpm': self.noise['fan_speed_noise_std']
        }
        
        for key, true_val in true_state.items():
            fault = self.faults.get(key, {'drift': 0.0, 'spike': 0.0, 'dropout': False})
            
            if fault['dropout']:
                observed[key] = -1.0 # Or some other missing value indicator, but we'll use 0.0 for physical bounds later
                continue
                
            val = true_val + fault['drift']
            
            std = std_map.get(key, 0.0)
            if std > 0:
                val += random.gauss(0, std)
                
            if fault['spike'] != 0.0:
                val += fault['spike']
                
            # Physical bounds
            if key != 'temperature_c':
                val = max(0.0, val)
                
            observed[key] = round(val, 3)
            
        return observed
