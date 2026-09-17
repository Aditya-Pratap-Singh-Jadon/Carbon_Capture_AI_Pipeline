import logging
import os
from datetime import datetime

def setup_logger(config):
    os.makedirs("logs", exist_ok=True)
    fname = datetime.now().strftime("logs/true_state_%Y%m%d_%H%M%S.log")
    
    logger = logging.getLogger("DigitalTwin")
    level = getattr(logging, config['logging'].get('level', 'INFO').upper())
    logger.setLevel(level)
    
    fh = logging.FileHandler(fname)
    fh.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    return logger
