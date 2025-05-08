# File: marketml/models/rf_model.py
import pandas as pd
import numpy as np
try:
    from sklearn.ensemble import RandomForestClassifier
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

def run_rf_evaluation(X_train_scaled, y_train, X_test_scaled, y_test, **kwargs):
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
    print("\n--- Training and Evaluating RandomForest Model ---")
    results = {}
    default_metrics = {"RandomForest_Accuracy": np.nan, "RandomForest_F1_Macro": np.nan, "RandomForest_F1_Weighted": np.nan,
                       "RandomForest_Precision_Macro": np.nan, "RandomForest_Recall_Macro": np.nan}
    results.update(default_metrics) # Khởi tạo

    if not SKLEARN_AVAILABLE:
        print("Error: scikit-learn not installed. Cannot run RandomForest.")
        return results
    if metrics is None:
        print("    Skipping RF evaluation: Metrics module not imported.")
        return results

    try:
        print("  Training RandomForest...")
        # Lấy tham số từ kwargs hoặc dùng giá trị mặc định tốt
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 10)
        min_samples_leaf = kwargs.get('min_samples_leaf', 5)

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
        rf_model.fit(X_train_scaled, y_train) # Hoạt động với cả array và df

        print("  Predicting with RandomForest...")
        rf_preds = rf_model.predict(X_test_scaled)

        # Đánh giá
        rf_metrics = metrics.calculate_classification_metrics(y_test, rf_preds, model_name="RandomForest")
        results.update(rf_metrics) # Ghi đè NaN

    except Exception as e:
         print(f"Error during RandomForest execution: {e}")
         # Giữ nguyên kết quả NaN đã khởi tạo

    return results