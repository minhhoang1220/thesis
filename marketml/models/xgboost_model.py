# File: marketml/models/xgboost_model.py
import pandas as pd
import numpy as np
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None # Dummy

# Đảm bảo có thể import từ thư mục cha hoặc utils đã trong sys.path
try:
    from marketml.utils import metrics
except ModuleNotFoundError:
    try:
        from ..utils import metrics
    except ImportError:
        print("Error: Cannot import metrics module in xgboost_model.py")
        metrics = None

def run_xgboost_evaluation(X_train_scaled, y_train, X_test_scaled, y_test, **kwargs):
    """
    Huấn luyện, dự đoán và đánh giá mô hình XGBoost Classifier.

    Args:
        X_train_scaled (np.array or pd.DataFrame): Features huấn luyện đã scale.
        y_train (np.array or pd.Series): Target huấn luyện (giá trị gốc -1, 0, 1).
        X_test_scaled (np.array or pd.DataFrame): Features test đã scale.
        y_test (np.array or pd.Series): Target test (giá trị gốc -1, 0, 1).
        **kwargs: Các tham số khác cho XGBClassifier (ví dụ: n_estimators).

    Returns:
        dict: Dictionary chứa kết quả metrics của XGBoost.
    """
    print("\n--- Training and Evaluating XGBoost Model ---")
    results = {}
    default_metrics = {"XGBoost_Accuracy": np.nan, "XGBoost_F1_Macro": np.nan, "XGBoost_F1_Weighted": np.nan,
                       "XGBoost_Precision_Macro": np.nan, "XGBoost_Recall_Macro": np.nan}
    results.update(default_metrics) # Khởi tạo

    if not XGB_AVAILABLE:
        print("Error: xgboost not installed. Cannot run XGBoost.")
        return results
    if metrics is None:
        print("    Skipping XGBoost evaluation: Metrics module not imported.")
        return results

    try:
        # XGBoost yêu cầu nhãn lớp phải là số nguyên không âm (0, 1, 2, ...)
        # Chuyển đổi nhãn train từ (-1, 0, 1) thành (0, 1, 2)
        y_train_xgb = y_train + 1
        # y_test không cần chuyển đổi ở đây, chỉ dùng để so sánh cuối cùng

        print("  Training XGBoost...")
        # Lấy tham số từ kwargs hoặc dùng giá trị mặc định
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 6) # Thường sâu hơn RF một chút
        learning_rate = kwargs.get('learning_rate', 0.1)

        # objective='multi:softmax' cho phân loại đa lớp
        # num_class=3 vì có 3 lớp (0, 1, 2 sau khi chuyển đổi)
        # use_label_encoder=False để tránh warning
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            use_label_encoder=False, # Quan trọng
            eval_metric='mlogloss',   # Metric đánh giá nội bộ (nếu dùng early stopping)
            random_state=42,
            n_jobs=-1
            # Lưu ý: XGBoost không có 'class_weight' trực tiếp như RF.
            # Xử lý mất cân bằng phức tạp hơn, có thể dùng tham số 'scale_pos_weight'
            # cho từng lớp hoặc dùng 'sample_weight' trong fit. Tạm thời bỏ qua.
        )

        # Huấn luyện với nhãn đã chuyển đổi (0, 1, 2)
        xgb_model.fit(X_train_scaled, y_train_xgb)

        print("  Predicting with XGBoost...")
        # Dự đoán cũng sẽ ra nhãn (0, 1, 2)
        xgb_preds_xgb = xgb_model.predict(X_test_scaled)

        # Chuyển đổi dự đoán về lại thang đo gốc (-1, 0, 1) để đánh giá
        xgb_preds_trend = xgb_preds_xgb - 1

        # Đánh giá với nhãn test gốc (-1, 0, 1)
        xgb_metrics = metrics.calculate_classification_metrics(y_test, xgb_preds_trend, model_name="XGBoost")
        results.update(xgb_metrics) # Ghi đè NaN

    except Exception as e:
         print(f"Error during XGBoost execution: {e}")
         # Giữ nguyên kết quả NaN đã khởi tạo

    return results