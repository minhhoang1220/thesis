# File: marketml/models/rf_model.py
import pandas as pd
import numpy as np
import logging

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier, RandomizedSearchCV, randint, uniform = None, None, None, None

try:
    from marketml.utils import metrics
except ModuleNotFoundError:
    print("CRITICAL ERROR in rf_model.py: Could not import 'marketml.utils.metrics'.")
    raise

def run_rf_evaluation(X_train_scaled, y_train, X_test_scaled, y_test,
                      logger: logging.Logger,
                      n_iter_search=20, cv_folds_tuning=3, **kwargs):
    logger.info("--- Training and Evaluating RandomForest Model (with Tuning) ---")
    results = {}
    default_metrics = {
        "RandomForest_Accuracy": np.nan, "RandomForest_F1_Macro": np.nan,
        "RandomForest_F1_Weighted": np.nan, "RandomForest_Precision_Macro": np.nan,
        "RandomForest_Recall_Macro": np.nan, "RandomForest_BestParams": "Skipped or Error"
    }
    results.update(default_metrics)

    if not SKLEARN_AVAILABLE:
        logger.error("  Skipping RandomForest: scikit-learn not installed.")
        return results
    if not X_train_scaled.size or not y_train.size or not X_test_scaled.size or not y_test.size:
        logger.warning(f"  Skipping RandomForest: Empty data provided (X_train:{X_train_scaled.shape}, y_train:{y_train.shape}, X_test:{X_test_scaled.shape}, y_test:{y_test.shape}).")
        return results

    try:
        logger.info(f"  Performing RandomizedSearchCV for RandomForest (n_iter={n_iter_search}, cv={cv_folds_tuning})...")
        param_dist = {
            'n_estimators': randint(kwargs.get('rf_n_estimators_min', 200), kwargs.get('rf_n_estimators_max', 401)),
            'max_depth': randint(kwargs.get('rf_max_depth_min', 10), kwargs.get('rf_max_depth_max', 31)),
            'min_samples_split': randint(kwargs.get('rf_min_samples_split_min', 5), kwargs.get('rf_min_samples_split_max', 21)),
            'min_samples_leaf': randint(kwargs.get('rf_min_samples_leaf_min', 3), kwargs.get('rf_min_samples_leaf_max', 16)),
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }
        base_rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        rf_search = RandomizedSearchCV(
            estimator=base_rf, param_distributions=param_dist,
            n_iter=n_iter_search, cv=cv_folds_tuning, verbose=0,
            random_state=42, n_jobs=-1, scoring='f1_macro'
        )
        rf_search.fit(X_train_scaled, y_train)

        logger.info(f"  Best RandomForest params found: {rf_search.best_params_}")
        best_rf_model = rf_search.best_estimator_

        logger.info("  Predicting with best RandomForest...")
        rf_preds = best_rf_model.predict(X_test_scaled)

        rf_metrics_results = metrics.calculate_classification_metrics(y_test, rf_preds, model_name="RandomForest", logger=logger)
        results.update(rf_metrics_results)
        results["RandomForest_BestParams"] = str(rf_search.best_params_)

    except Exception as e:
         logger.error(f"Error during RandomForest execution: {e}", exc_info=True)
    return results
