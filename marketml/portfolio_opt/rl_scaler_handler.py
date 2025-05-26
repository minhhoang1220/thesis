# marketml/portfolio_opt/rl_scaler_handler.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__) # Sử dụng logger của module

SCALER_FILE_NAME = "financial_scalers.joblib" # Hằng số tên file

class FinancialFeatureScaler:
    def __init__(self, feature_names: list = None, means: pd.Series = None, stds: pd.Series = None):
        self.feature_names = feature_names if feature_names is not None else []
        self.means = means if means is not None else pd.Series(dtype=float)
        self.stds = stds if stds is not None else pd.Series(dtype=float)
        
        if not self.feature_names:
            logger.debug("FinancialFeatureScaler initialized without pre-defined feature_names.")
        if self.means.empty or self.stds.empty:
            logger.debug("FinancialFeatureScaler initialized with empty means or stds.")

    def fit(self, data: pd.DataFrame, feature_names: list):
        if not isinstance(data, pd.DataFrame) or data.empty:
            logger.warning("Cannot fit FinancialFeatureScaler: input data is not a DataFrame or is empty.")
            self.feature_names = list(feature_names) if feature_names else []
            self.means = pd.Series(dtype=float).reindex(self.feature_names, fill_value=0.0)
            self.stds = pd.Series(dtype=float).reindex(self.feature_names, fill_value=1.0)
            return

        self.feature_names = list(feature_names) # Store the intended order
        valid_features_to_fit = [f for f in self.feature_names if f in data.columns and pd.api.types.is_numeric_dtype(data[f])]
        
        if not valid_features_to_fit:
            logger.warning("No valid numeric features found in data to fit the scaler based on provided feature_names.")
            self.means = pd.Series(dtype=float).reindex(self.feature_names, fill_value=0.0)
            self.stds = pd.Series(dtype=float).reindex(self.feature_names, fill_value=1.0)
            return

        data_to_fit = data[valid_features_to_fit].replace([np.inf, -np.inf], np.nan)
        
        calculated_means = data_to_fit.mean()
        calculated_stds = data_to_fit.std()
        calculated_stds[calculated_stds < 1e-6] = 1.0 # Avoid division by zero or very small std

        # Ensure self.means and self.stds are Series with all self.feature_names as index
        self.means = pd.Series(0.0, index=self.feature_names, dtype=float)
        self.stds = pd.Series(1.0, index=self.feature_names, dtype=float)

        self.means.update(calculated_means) # Update with actual means for features present
        self.stds.update(calculated_stds)   # Update with actual stds for features present

        logger.info(f"FinancialFeatureScaler fitted on features: {valid_features_to_fit}")
        logger.debug(f"Calculated Means:\n{self.means.to_string()}")
        logger.debug(f"Calculated Stds:\n{self.stds.to_string()}")

    def transform(self, data_for_ticker: pd.Series) -> np.ndarray:
        if not self.feature_names:
            logger.warning("Attempting to transform with FinancialFeatureScaler that has not been fitted or has no feature_names. Returning empty array.")
            return np.array([], dtype=np.float32)
            
        scaled_features = np.zeros(len(self.feature_names), dtype=np.float32)
        for i, feature_name in enumerate(self.feature_names):
            raw_value = data_for_ticker.get(feature_name, np.nan)
            
            numeric_value = 0.0 # Default if conversion fails or NaN
            if pd.notna(raw_value):
                try:
                    numeric_value = float(raw_value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert feature '{feature_name}' value '{raw_value}' to float for ticker. Using 0.0.")
            else: # raw_value is NaN
                logger.debug(f"Feature '{feature_name}' for ticker is NaN. Using 0.0 before scaling.")

            # Get pre-calculated mean and std for this feature
            mean_val = self.means.get(feature_name) # Should exist due to reindex in fit
            std_val = self.stds.get(feature_name)   # Should exist due to reindex in fit

            if pd.notna(mean_val) and pd.notna(std_val) and std_val > 1e-6:
                scaled_features[i] = (numeric_value - mean_val) / std_val
            else:
                # If mean/std were not available (e.g., feature not in training) or std is too small,
                # use 0.0 if original was NaN or unconvertible, or the raw numeric_value if it was valid but not scaled.
                # For consistency, if scaling parameters are invalid, perhaps output 0 or a special flag.
                # Current behavior: if numeric_value is 0 (from NaN/error), it remains 0.
                # If numeric_value is non-zero but std_val is bad, it uses numeric_value (unscaled).
                # Let's default to 0 if scaling params are bad to avoid passing unscaled data mixed with scaled.
                scaled_features[i] = 0.0
                logger.debug(f"Invalid scaling parameters for feature '{feature_name}' (mean: {mean_val}, std: {std_val}). Using 0.0. Original numeric value was {numeric_value}.")
        return scaled_features

    def save(self, directory: Path):
        if not self.feature_names: # Only check feature_names, as means/stds are reindexed to match
            logger.warning("FinancialFeatureScaler has no feature_names defined. Nothing to save.")
            return
            
        filepath = directory / SCALER_FILE_NAME
        try:
            directory.mkdir(parents=True, exist_ok=True)
            joblib.dump({'feature_names': self.feature_names, 'means': self.means, 'stds': self.stds}, filepath)
            logger.info(f"FinancialFeatureScaler saved to {filepath.resolve()}")
        except Exception as e:
            logger.error(f"Error saving FinancialFeatureScaler to {filepath.resolve()}: {e}", exc_info=True)

    @classmethod
    def load(cls, directory: Path) -> 'FinancialFeatureScaler':
        filepath = directory / SCALER_FILE_NAME
        if filepath.exists():
            try:
                data = joblib.load(filepath)
                # Ensure all keys exist, provide defaults if not for robustness
                feature_names = data.get('feature_names', [])
                means = data.get('means', pd.Series(dtype=float))
                stds = data.get('stds', pd.Series(dtype=float))
                
                scaler = cls(feature_names=feature_names, means=means, stds=stds)
                logger.info(f"FinancialFeatureScaler loaded from {filepath.resolve()}")
                if not scaler.feature_names or scaler.means.empty or scaler.stds.empty:
                    logger.warning(f"Scaler loaded from {filepath.resolve()} has incomplete data "
                                   f"(Features: {bool(scaler.feature_names)}, Means: {not scaler.means.empty}, Stds: {not scaler.stds.empty}).")
                return scaler
            except Exception as e:
                logger.error(f"Error loading FinancialFeatureScaler from {filepath.resolve()}: {e}", exc_info=True)
                return cls() # Return an empty scaler on error
        else:
            logger.warning(f"Scaler file not found at {filepath.resolve()}. Returning a new, empty scaler.")
            return cls()