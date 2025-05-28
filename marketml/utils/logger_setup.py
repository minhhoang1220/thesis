# marketml/utils/logger_setup.py

import warnings
import logging
from pathlib import Path
import random
import numpy as np

# Attempt to import configs for LOG_OUTPUT_DIR
try:
    from marketml.configs import configs
    LOG_DIR_FROM_CONFIG = configs.LOG_OUTPUT_DIR
except ImportError:
    # Fallback: set log directory relative to project root if configs cannot be imported
    LOG_DIR_FROM_CONFIG = Path(__file__).resolve().parents[2] / "logs"
    print(f"Warning: Could not import configs for logger setup. Defaulting log directory to: {LOG_DIR_FROM_CONFIG}")

_project_logger_configured = False

def suppress_common_warnings():
    """
    Suppress common, non-critical warnings from libraries.
    """
    print("Suppressing common warnings...")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    try:
        from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
        warnings.filterwarnings("ignore", category=ValueWarning, module='statsmodels')
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module='statsmodels')
    except ImportError:
        pass

    try:
        from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
        warnings.filterwarnings("ignore", category=SklearnConvergenceWarning)
    except ImportError:
        pass

    print("Warnings suppressed.")

def setup_basic_logging(log_level=logging.INFO, log_file_name="project.log"):
    """
    Set up basic logging for the project.
    """
    global _project_logger_configured
    logger = logging.getLogger("marketml_project")

    if not _project_logger_configured:
        logs_dir = LOG_DIR_FROM_CONFIG
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = logs_dir / log_file_name

        # Remove all handlers to prevent duplicate logs
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        logger.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file_path, mode='a')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.propagate = False
        _project_logger_configured = True
        logger.info(f"Logging setup complete. Log level: {logging.getLevelName(logger.level)}. Log file: {log_file_path}")
    return logger

def set_random_seeds(seed_value=42):
    """
    Set random seeds for reproducibility.
    """
    import random
    import numpy as np
    try:
        import tensorflow as tf
    except ImportError:
        tf = None
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf:
        tf.random.set_seed(seed_value)
    logging.getLogger("marketml_project").info(f"Random seeds set to: {seed_value}")
