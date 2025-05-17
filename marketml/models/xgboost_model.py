# File: marketml/models/xgboost_model.py
import pandas as pd
import numpy as np
try:
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
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

def run_xgboost_evaluation(X_train_scaled, y_train, X_test_scaled, y_test, n_iter_search=20, cv_folds_tuning=3, **kwargs):
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
    print("\n--- Training and Evaluating XGBoost Model (with Tuning) ---")
    results = {}
    default_metrics = {"XGBoost_Accuracy": np.nan, "XGBoost_F1_Macro": np.nan, "XGBoost_F1_Weighted": np.nan,
                       "XGBoost_Precision_Macro": np.nan, "XGBoost_Recall_Macro": np.nan}
    results.update(default_metrics) # Khởi tạo

    if not XGB_AVAILABLE or RandomizedSearchCV is None or randint is None:
        print("Error: xgboost not installed. Cannot run XGBoost.")
        return results
    if metrics is None:
        print("    Skipping XGBoost evaluation: Metrics module not imported.")
        return results

    try:
        y_train_xgb = y_train + 1 # Chuyển về 0, 1, 2 cho XGBoost

        print(f"  Performing RandomizedSearchCV for XGBoost (n_iter={n_iter_search}, cv={cv_folds_tuning})...")

        # Chiến lược tuning của bạn:
        # learning_rate: Giảm nhẹ (0.01 - 0.1)
        # max_depth: Test giá trị nhỏ hơn (3-6)
        # subsample, colsample_bytree: [0.5, 0.9]
        # n_estimators: Giữ ổn định, nhưng tăng nhẹ nếu giảm learning_rate.
        # (Chúng ta sẽ để n_estimators trong khoảng, RandomizedSearchCV sẽ tự tìm)
        param_dist = {
            'n_estimators': randint(100, 301), # Giữ ổn định hoặc tăng nhẹ
            'learning_rate': uniform(0.01, 0.09), # Từ 0.01 đến 0.1 (0.01 + 0.09 = 0.1)
            'max_depth': randint(3, 7),         # Từ 3 đến 6
            'subsample': uniform(0.5, 0.4),     # Từ 0.5 đến 0.9 (0.5 + 0.4 = 0.9)
            'colsample_bytree': uniform(0.5, 0.4),# Từ 0.5 đến 0.9
            'gamma': uniform(0, 0.5) # Thêm gamma để kiểm soát phức tạp
            # 'min_child_weight': randint(1, 6) # Có thể thêm
        }

        base_xgb = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )

        xgb_search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=cv_folds_tuning,
            verbose=0,
            random_state=42,
            n_jobs=-1,
            scoring='f1_macro'
        )

        xgb_search.fit(X_train_scaled, y_train_xgb)

        print(f"  Best XGBoost params found: {xgb_search.best_params_}")
        best_xgb_model = xgb_search.best_estimator_

        print("  Predicting with best XGBoost...")
        xgb_preds_xgb = best_xgb_model.predict(X_test_scaled)
        xgb_preds_trend = xgb_preds_xgb - 1 # Về lại -1, 0, 1

        xgb_metrics = metrics.calculate_classification_metrics(y_test, xgb_preds_trend, model_name="XGBoost")
        results.update(xgb_metrics)
        results["XGBoost_BestParams"] = str(xgb_search.best_params_)

    except Exception as e:
         print(f"Error during XGBoost execution: {e}")
         # Giữ nguyên kết quả NaN

    return results