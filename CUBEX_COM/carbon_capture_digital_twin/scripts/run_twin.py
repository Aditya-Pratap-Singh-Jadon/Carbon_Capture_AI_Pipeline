import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.digital_twin import DigitalTwin

def main():
    parser = argparse.ArgumentParser(description="Digital Twin Simulator")
    parser.add_argument('--scenario', type=str, default='mixed', help='Scenario name (e.g. mixed, normal, sensor_drift)')
    parser.add_argument('--mode', type=str, default='stdout', choices=['stdout', 'serial', 'csv'], help='Output mode')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to generate before terminating')
    parser.add_argument('--labeled', action='store_true', help='Append scenario label to output (for training only)')
    
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config)
    
    twin = DigitalTwin(mode=args.mode, scenario=args.scenario, config_path=config_path, samples=args.samples, labeled=args.labeled)
    twin.run()

if __name__ == '__main__':
    main()
