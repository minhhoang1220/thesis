# File: marketml/models/rf_model.py
import pandas as pd
import numpy as np
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform 
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None # Dummy

# Đảm bảo có thể import từ thư mục cha hoặc utils đã trong sys.path
try:
    from marketml.utils import metrics
except ModuleNotFoundError:
    try:
        from ..utils import metrics
    except ImportError:
        print("Error: Cannot import metrics module in rf_model.py")
        metrics = None

def run_rf_evaluation(X_train_scaled, y_train, X_test_scaled, y_test, n_iter_search=20, cv_folds_tuning=3, **kwargs):
    """
    Huấn luyện, dự đoán và đánh giá mô hình RandomForest.

    Args:
        X_train_scaled (np.array or pd.DataFrame): Features huấn luyện đã scale.
        y_train (np.array or pd.Series): Target huấn luyện.
        X_test_scaled (np.array or pd.DataFrame): Features test đã scale.
        y_test (np.array or pd.Series): Target test.
        **kwargs: Các tham số khác cho RandomForestClassifier (ví dụ: n_estimators).

    Returns:
        dict: Dictionary chứa kết quả metrics của RandomForest.
    """
    print("\n--- Training and Evaluating RandomForest Model (with Tuning) ---")
    results = {}
    default_metrics = {"RandomForest_Accuracy": np.nan, "RandomForest_F1_Macro": np.nan, "RandomForest_F1_Weighted": np.nan,
                       "RandomForest_Precision_Macro": np.nan, "RandomForest_Recall_Macro": np.nan}
    results.update(default_metrics) # Khởi tạo

    if not SKLEARN_AVAILABLE or RandomizedSearchCV is None or randint is None:
        print("Error: scikit-learn not installed. Cannot run RandomForest.")
        return results
    if metrics is None:
        print("    Skipping RF evaluation: Metrics module not imported.")
        return results

    try:
        print(f"  Performing RandomizedSearchCV for RandomForest (n_iter={n_iter_search}, cv={cv_folds_tuning})...")

        # Chiến lược tuning của bạn:
        # max_depth: Giới hạn chiều sâu (10–30) để giảm overfitting.
        # min_samples_split, min_samples_leaf: Tăng nhẹ để giảm biến động giữa folds.
        # n_estimators: Tăng số cây lên 300–500 để tăng tính ổn định.
        param_dist = {
            'n_estimators': randint(300, 501), # Từ 300 đến 500
            'max_depth': randint(10, 31),    # Từ 10 đến 30
            'min_samples_split': randint(5, 21), # Tăng nhẹ, ví dụ từ 5 đến 20
            'min_samples_leaf': randint(3, 16),  # Tăng nhẹ, ví dụ từ 3 đến 15
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }

        base_rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

        rf_search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=cv_folds_tuning,
            verbose=0, # Đặt 1 hoặc 2 để xem chi tiết
            random_state=42,
            n_jobs=-1,
            scoring='f1_macro' # Ưu tiên F1 Macro
        )

        rf_search.fit(X_train_scaled, y_train)

        print(f"  Best RandomForest params found: {rf_search.best_params_}")
        best_rf_model = rf_search.best_estimator_

        print("  Predicting with best RandomForest...")
        rf_preds = best_rf_model.predict(X_test_scaled)

        rf_metrics = metrics.calculate_classification_metrics(y_test, rf_preds, model_name="RandomForest")
        results.update(rf_metrics)
        results["RandomForest_BestParams"] = str(rf_search.best_params_)

    except Exception as e:
         print(f"Error during RandomForest execution: {e}")
         # Giữ nguyên kết quả NaN

    return results