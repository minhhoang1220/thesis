import pandas as pd
from marketml.configs import configs
from marketml.data_handling.loader import load_price_data
from marketml.features.feature_builder import build_enriched_features
from marketml.utils import logger_setup

logger = logger_setup.setup_basic_logging(log_file_name="build_features.log")

def main():
    logger.info("Starting: 01_build_features pipeline")
    raw_prices = load_price_data()
    if raw_prices.empty:
        logger.error("Raw price data is empty. Exiting feature building.")
        return

    enriched_df = build_enriched_features(raw_prices)

    if not enriched_df.empty:
        try:
            configs.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            enriched_df.to_csv(configs.ENRICHED_DATA_FILE, index=False)
            logger.info(f"Enriched data saved to: {configs.ENRICHED_DATA_FILE}")
        except Exception as e:
            logger.error(f"Error saving enriched data: {e}")
    else:
        logger.warning("Enriched dataframe is empty. Nothing saved.")
    logger.info("Finished: 01_build_features pipeline")

if __name__ == "__main__":
    main()
