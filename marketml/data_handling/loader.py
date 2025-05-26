# marketml/data_handling/loader.py
import pandas as pd
from marketml.configs import configs
import logging

logger = logging.getLogger(__name__)

def load_price_data(nrows=None, logger_instance: logging.Logger = None) -> pd.DataFrame:
    current_logger = logger_instance if logger_instance else logger
    file_path = configs.RAW_GLOBAL_PRICE_FILE  # Get path from configs
    current_logger.info(f"Loading price data from: {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"], nrows=nrows)
        if df.empty:
            current_logger.warning(f"Price data file loaded from {file_path} is empty.")
        return df
    except FileNotFoundError:
        current_logger.error(f"Price data file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        current_logger.error(f"Error loading price data from {file_path}: {e}", exc_info=True)
        return pd.DataFrame()


def load_financial_data(nrows=None, logger_instance: logging.Logger = None) -> pd.DataFrame:
    current_logger = logger_instance if logger_instance else logger
    file_path = configs.RAW_GLOBAL_FINANCIAL_FILE  # Get path from configs
    current_logger.info(f"Loading financial data from: {file_path}")
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        if df.empty:
            current_logger.warning(f"Financial data file loaded from {file_path} is empty.")
        return df
    except FileNotFoundError:
        current_logger.error(f"Financial data file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        current_logger.error(f"Error loading financial data from {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

