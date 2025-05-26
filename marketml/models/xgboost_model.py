# File: marketml/models/xgboost_model.py
import pandas as pd
import numpy as np
import logging

try:
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb, RandomizedSearchCV, randint, uniform = None, None, None, None

try:
    from marketml.utils import metrics
except ModuleNotFoundError:
    print("CRITICAL ERROR in xgboost_model.py: Could not import 'marketml.utils.metrics'.")
    raise

def run_xgboost_evaluation(X_train_scaled, y_train, X_test_scaled, y_test,
                           logger: logging.Logger,
                           n_iter_search=20, cv_folds_tuning=3, **kwargs):
    logger.info("--- Training and Evaluating XGBoost Model (with Tuning) ---")
    results = {}
    default_metrics = {
        "XGBoost_Accuracy": np.nan, "XGBoost_F1_Macro": np.nan,
        "XGBoost_F1_Weighted": np.nan, "XGBoost_Precision_Macro": np.nan,
        "XGBoost_Recall_Macro": np.nan, "XGBoost_BestParams": "Skipped or Error"
    }
    results.update(default_metrics)

    if not XGB_AVAILABLE:
        logger.error("  Skipping XGBoost: xgboost library not installed.")
        return results
    if not X_train_scaled.size or not y_train.size or not X_test_scaled.size or not y_test.size:
        logger.warning(f"  Skipping XGBoost: Empty data provided (X_train:{X_train_scaled.shape}, y_train:{y_train.shape}, X_test:{X_test_scaled.shape}, y_test:{y_test.shape}).")
        return results

    try:
        # XGBoost expects target classes to be 0, 1, 2 instead of -1, 0, 1
        y_train_xgb = y_train + 1

        logger.info(f"  Performing RandomizedSearchCV for XGBoost (n_iter={n_iter_search}, cv={cv_folds_tuning})...")
        param_dist = {
            'n_estimators': randint(kwargs.get('xgb_n_estimators_min', 100), kwargs.get('xgb_n_estimators_max', 301)),
            'learning_rate': uniform(kwargs.get('xgb_lr_min', 0.01), kwargs.get('xgb_lr_range', 0.09)),
            'max_depth': randint(kwargs.get('xgb_max_depth_min', 3), kwargs.get('xgb_max_depth_max', 7)),
            'subsample': uniform(kwargs.get('xgb_subsample_min', 0.5), kwargs.get('xgb_subsample_range', 0.4)),
            'colsample_bytree': uniform(kwargs.get('xgb_colsample_min', 0.5), kwargs.get('xgb_colsample_range', 0.4)),
            'gamma': uniform(kwargs.get('xgb_gamma_min', 0), kwargs.get('xgb_gamma_range', 0.5))
        }
        base_xgb = xgb.XGBClassifier(
            objective='multi:softmax', num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42, n_jobs=-1
        )
        xgb_search = RandomizedSearchCV(
            estimator=base_xgb, param_distributions=param_dist,
            n_iter=n_iter_search, cv=cv_folds_tuning, verbose=0,
            random_state=42, n_jobs=-1, scoring='f1_macro'
        )
        xgb_search.fit(X_train_scaled, y_train_xgb)

        logger.info(f"  Best XGBoost params found: {xgb_search.best_params_}")
        best_xgb_model = xgb_search.best_estimator_

        logger.info("  Predicting with best XGBoost...")
        xgb_preds_mapped = best_xgb_model.predict(X_test_scaled)
        xgb_preds_trend = xgb_preds_mapped - 1 # Convert back to -1, 0, 1

        xgb_metrics_results = metrics.calculate_classification_metrics(y_test, xgb_preds_trend, model_name="XGBoost", logger=logger)
        results.update(xgb_metrics_results)
        results["XGBoost_BestParams"] = str(xgb_search.best_params_)

    except Exception as e:
         logger.error(f"Error during XGBoost execution: {e}", exc_info=True)
    return results
