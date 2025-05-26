# File: marketml/models/svm_model.py
import pandas as pd
import numpy as np
import logging

try:
    from sklearn.svm import SVC
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import expon, uniform, randint
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SVC, RandomizedSearchCV, expon, uniform, randint = None, None, None, None, None

try:
    from marketml.utils import metrics
except ModuleNotFoundError:
    print("CRITICAL ERROR in svm_model.py: Could not import 'marketml.utils.metrics'.")
    raise

def run_svm_evaluation(X_train_scaled, y_train, X_test_scaled, y_test,
                       logger: logging.Logger,
                       n_iter_search=20, cv_folds_tuning=3, **kwargs):
    logger.info("--- Training and Evaluating SVM Model (with Tuning) ---")
    results = {}
    default_metrics = {
        "SVM_Accuracy": np.nan, "SVM_F1_Macro": np.nan, "SVM_F1_Weighted": np.nan,
        "SVM_Precision_Macro": np.nan, "SVM_Recall_Macro": np.nan, "SVM_BestParams": "Skipped or Error"
    }
    results.update(default_metrics)

    if not SKLEARN_AVAILABLE:
        logger.error("  Skipping SVM: scikit-learn or scipy.stats components not installed/imported.")
        return results
    if not X_train_scaled.size or not y_train.size or not X_test_scaled.size or not y_test.size:
        logger.warning(f"  Skipping SVM: Empty data provided (X_train:{X_train_scaled.shape}, y_train:{y_train.shape}, X_test:{X_test_scaled.shape}, y_test:{y_test.shape}).")
        return results

    try:
        logger.info(f"  Performing RandomizedSearchCV for SVM (n_iter={n_iter_search}, cv={cv_folds_tuning})...")
        param_dist = {
            'C': expon(scale=kwargs.get('svm_c_scale', 1.0)),
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto'] + list(uniform(kwargs.get('svm_gamma_min', 1e-4), kwargs.get('svm_gamma_range', 1e-1 - 1e-4)).rvs(5)),
            'degree': randint(kwargs.get('svm_degree_min', 2), kwargs.get('svm_degree_max', 5))
        }
        base_svm = SVC(class_weight='balanced', probability=False, random_state=42)
        svm_search = RandomizedSearchCV(
            estimator=base_svm, param_distributions=param_dist,
            n_iter=n_iter_search, cv=cv_folds_tuning, verbose=0,
            random_state=42, n_jobs=-1, scoring='f1_macro'
        )
        svm_search.fit(X_train_scaled, y_train)

        logger.info(f"  Best SVM params found: {svm_search.best_params_}")
        best_svm_model = svm_search.best_estimator_

        logger.info("  Predicting with best SVM...")
        svm_preds = best_svm_model.predict(X_test_scaled)

        svm_metrics_results = metrics.calculate_classification_metrics(y_test, svm_preds, model_name="SVM", logger=logger)
        results.update(svm_metrics_results)
        results["SVM_BestParams"] = str(svm_search.best_params_)

    except Exception as e:
         logger.error(f"Error during SVM execution: {e}", exc_info=True)
    return results