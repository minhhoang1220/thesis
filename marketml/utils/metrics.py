# /.ndmh/marketml/utils/metrics.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import logging

# Logger for this module, but function will prioritize logger passed in
module_logger = logging.getLogger(__name__)

def calculate_classification_metrics(y_true, y_pred, model_name="Model", logger: logging.Logger = None):
    """
    Calculates, logs, and returns classification metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels by the model.
        model_name (str): Name of the model for logging and keys in results.
        logger (logging.Logger, optional): Logger instance to use. If None, uses module_logger.

    Returns:
        dict: Dictionary containing calculated metrics. Returns dict with NaNs if calculation fails.
    """
    current_logger = logger if logger is not None else module_logger
    results = {}
    # Initialize with NaNs for robustness
    metric_keys = ["Accuracy", "F1_Macro", "F1_Weighted", "Precision_Macro", "Recall_Macro"]
    for key in metric_keys:
        results[f"{model_name}_{key}"] = np.nan

    current_logger.info(f"--- Evaluating {model_name} ---")
    try:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if len(y_true_arr) == 0 or len(y_pred_arr) == 0 :
            current_logger.warning(f"  {model_name}: Empty true labels or predictions. Cannot calculate metrics.")
            return results # Return NaNs

        if len(y_true_arr) != len(y_pred_arr):
            current_logger.warning(f"  {model_name}: Length mismatch between y_true ({len(y_true_arr)}) and y_pred ({len(y_pred_arr)}). Cannot calculate metrics.")
            return results # Return NaNs

        accuracy = accuracy_score(y_true_arr, y_pred_arr)
        
        # Ensure all unique labels in y_true and y_pred are present for target_names
        # Classes for -1, 0, 1 are 'Decrease (-1)', 'No Change (0)', 'Increase (1)'
        # Determine unique labels present in either y_true or y_pred that are in {-1, 0, 1}
        present_labels = sorted(list(set(y_true_arr) | set(y_pred_arr)))
        target_names_map = {-1: 'Decrease (-1)', 0: 'No Change (0)', 1: 'Increase (1)'}
        
        # Filter labels for report to only those actually present and in our map
        report_labels_numeric = [l for l in present_labels if l in target_names_map]
        report_target_names = [target_names_map[l] for l in report_labels_numeric]
        
        if not report_labels_numeric:
            report_labels_numeric = None
            report_target_names = None
            current_logger.warning(f"  {model_name}: No standard labels (-1,0,1) found in y_true/y_pred for detailed report. Report might be limited.")

        report_dict = classification_report(y_true_arr, y_pred_arr, labels=report_labels_numeric, output_dict=True, zero_division=0)
        report_str = classification_report(y_true_arr, y_pred_arr, labels=report_labels_numeric, target_names=report_target_names, zero_division=0)

        current_logger.info(f"  Accuracy: {accuracy:.4f}")
        current_logger.info(f"  Classification Report ({model_name}):\n{report_str}")

        results[f"{model_name}_Accuracy"] = accuracy
        results[f"{model_name}_F1_Macro"] = report_dict.get('macro avg', {}).get('f1-score', np.nan)
        results[f"{model_name}_F1_Weighted"] = report_dict.get('weighted avg', {}).get('f1-score', np.nan)
        results[f"{model_name}_Precision_Macro"] = report_dict.get('macro avg', {}).get('precision', np.nan)
        results[f"{model_name}_Recall_Macro"] = report_dict.get('macro avg', {}).get('recall', np.nan)

    except ValueError as ve:
        current_logger.error(f"  ValueError calculating metrics for {model_name} (possibly due to label issues): {ve}", exc_info=True)
    except Exception as e:
        current_logger.error(f"  General error calculating metrics for {model_name}: {e}", exc_info=True)
    
    return results
