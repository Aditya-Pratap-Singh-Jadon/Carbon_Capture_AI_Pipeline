import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.process_model import ProcessModel
from src.sensor_model import SensorModel
from src.scenarios import ScenarioEngine
from src.telemetry import TelemetryFormatter
from datetime import datetime

def get_default_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_process_model_normal():
    config = get_default_config()
    process = ProcessModel(config)
    process.set_fan_command(1500)
    process.step()
    
    state = process.get_state()
    assert state['fan_speed_rpm'] > 1000 # should be responding
    assert state['gas_flow_l_min'] > 0

def test_sensor_model_bounds():
    config = get_default_config()
    sensor = SensorModel(config)
    
    # Test negative flow physical bound
    state = {'co2_ppm': 400.0, 'temperature_c': -5.0, 'humidity_percent': 0.0, 'gas_flow_l_min': -10.0, 'captured_co2_g': -1.0, 'fan_speed_rpm': -100.0}
    observed = sensor.apply(state)
    assert observed['gas_flow_l_min'] >= 0.0
    assert observed['temperature_c'] < 0.0 # temperature can be negative

def test_telemetry_formatter():
    formatter = TelemetryFormatter()
    now = datetime(2026, 9, 16, 12, 0, 0)
    state = {'co2_ppm': 420.0, 'temperature_c': 25.0, 'humidity_percent': 50.0, 'gas_flow_l_min': 500.0, 'captured_co2_g': 0.1, 'fan_speed_rpm': 1500.0}
    csv_str = formatter.format_csv(now, state)
    
    fields = csv_str.split(',')
    assert len(fields) == 7
    assert fields[0] == '2026-09-16T12:00:00.000'
    assert fields[1] == '420.0'
