import random

class ProcessModel:
    def __init__(self, config):
        self.config = config
        self.dt = 1.0 / config['simulation']['sampling_rate_hz']
        
        # True process states
        self.ambient_co2 = config['simulation']['ambient_co2_baseline_ppm']
        self.ambient_temp = config['simulation']['ambient_temperature_baseline_c']
        self.ambient_humidity = config['simulation']['ambient_humidity_baseline_percent']
        
        self.commanded_fan_rpm = config['process']['fan_min_rpm']
        self.actual_fan_rpm = config['process']['fan_min_rpm']
        
        self.gas_flow_l_min = 0.0
        self.total_captured_co2_g = 0.0
        
        # Config params
        self.fan_min = config['process']['fan_min_rpm']
        self.fan_max = config['process']['fan_max_rpm']
        self.fan_response_rate = config['process']['fan_response_rate']
        self.flow_per_rpm = config['process']['flow_per_rpm']
        self.capture_efficiency_base = config['process']['capture_efficiency_base']
        
        # Fault states
        self.fault_fan_failure = False
        self.fault_low_flow = False
        self.fault_efficiency_drop = False
        
    def step(self):
        # Evolve ambient CO2 smoothly (random walk with mean reversion)
        baseline_co2 = self.config['simulation']['ambient_co2_baseline_ppm']
        self.ambient_co2 += random.uniform(-0.5, 0.5)
        self.ambient_co2 += (baseline_co2 - self.ambient_co2) * 0.05
        
        # 1. Update fan RPM
        if self.fault_fan_failure:
            target_rpm = 0.0
        else:
            target_rpm = max(self.fan_min, min(self.commanded_fan_rpm, self.fan_max))
            
        self.actual_fan_rpm += (target_rpm - self.actual_fan_rpm) * self.fan_response_rate
        
        # 2. Update gas flow
        base_flow = self.actual_fan_rpm * self.flow_per_rpm
        if self.fault_low_flow:
            base_flow *= 0.3 # 70% reduction in flow
            
        self.gas_flow_l_min = base_flow
        
        # 3. Update captured CO2
        flow_rate_per_sec = self.gas_flow_l_min / 60.0
        co2_fraction = self.ambient_co2 / 1_000_000.0
        
        density_factor = 1.8 # approx g/L for CO2
        
        current_efficiency = self.capture_efficiency_base
        if self.fault_efficiency_drop:
            current_efficiency *= 0.5
            
        captured_this_step = flow_rate_per_sec * co2_fraction * density_factor * current_efficiency * self.dt
        self.total_captured_co2_g += captured_this_step

    def set_fan_command(self, rpm):
        self.commanded_fan_rpm = rpm
        
    def get_state(self):
        return {
            'co2_ppm': self.ambient_co2,
            'temperature_c': self.ambient_temp,
            'humidity_percent': self.ambient_humidity,
            'gas_flow_l_min': self.gas_flow_l_min,
            'captured_co2_g': self.total_captured_co2_g,
            'fan_speed_rpm': self.actual_fan_rpm
        }
