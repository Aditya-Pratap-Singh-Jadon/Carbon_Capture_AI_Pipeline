class Controller:
    def __init__(self, process_model):
        self.process = process_model
        
    def apply_command(self, cmd_str):
        """Parse external command string, e.g. '+100 RPM', '0 RPM'"""
        if not cmd_str:
            return
            
        cmd_str = cmd_str.strip().upper()
        if cmd_str.endswith("RPM"):
            cmd_str = cmd_str.replace("RPM", "").strip()
            try:
                if cmd_str.startswith("+") or cmd_str.startswith("-"):
                    val = float(cmd_str)
                    self.process.set_fan_command(self.process.commanded_fan_rpm + val)
                else:
                    val = float(cmd_str)
                    self.process.set_fan_command(val)
            except ValueError:
                pass
