class TelemetryFormatter:
    # Standard format: timestamp,CO2_ppm,Temperature_C,Humidity_percent,Gas_Flow_L_min,Captured_CO2_g,Fan_Speed_RPM
    def __init__(self):
        self.fields = [
            'co2_ppm', 
            'temperature_c', 
            'humidity_percent', 
            'gas_flow_l_min', 
            'captured_co2_g', 
            'fan_speed_rpm'
        ]
        
    def format_csv(self, timestamp, observed_state, scenario_label=None):
        vals = [timestamp.isoformat(timespec='milliseconds')]
        for f in self.fields:
            vals.append(str(observed_state.get(f, 0.0)))
        if scenario_label:
            vals.append(scenario_label)
        return ",".join(vals)
