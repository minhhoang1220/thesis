# File: marketml/models/svm_model.py
import pandas as pd
import numpy as np
try:
    from sklearn.svm import SVC
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import expon, uniform, randint
    SKLEARN_AVAILABLE = True
except ImportError as e_svm_ịnternal:
    print(f"INTERNAL SVM_MODEL.PY ERROR: Error importing scikit-learn/scipy.stats components: {e_svm_internal}")
    SVC, RandomizedSearchCV, expon, uniform, randint = None, None, None, None, None

# Đảm bảo có thể import từ thư mục cha hoặc utils đã trong sys.path
try:
    from marketml.utils import metrics
except ModuleNotFoundError:
    try:
        from ..utils import metrics
    except ImportError:
        print("Error: Cannot import metrics module in svm_model.py")
        metrics = None

def run_svm_evaluation(X_train_scaled, y_train, X_test_scaled, y_test, n_iter_search=20, cv_folds_tuning=3, **kwargs):
    """
    Huấn luyện, dự đoán và đánh giá mô hình Support Vector Classifier (SVC).

    Args:
        X_train_scaled (np.array or pd.DataFrame): Features huấn luyện đã scale.
        y_train (np.array or pd.Series): Target huấn luyện.
        X_test_scaled (np.array or pd.DataFrame): Features test đã scale.
        y_test (np.array or pd.Series): Target test.
        **kwargs: Các tham số khác cho SVC (ví dụ: C, kernel, gamma).

    Returns:
        dict: Dictionary chứa kết quả metrics của SVM.
    """
    print("\n--- Training and Evaluating SVM Model (with Tuning) ---")
    results = {}
    default_metrics = {"SVM_Accuracy": np.nan, "SVM_F1_Macro": np.nan, "SVM_F1_Weighted": np.nan,
                       "SVM_Precision_Macro": np.nan, "SVM_Recall_Macro": np.nan, "SVM_BestParams": "Error or Skipped"}
    results.update(default_metrics) # Khởi tạo

    if not SKLEARN_AVAILABLE or RandomizedSearchCV is None or expon is None or uniform is None or randint is None or SVC is None:
        print("Error: scikit-learn or required scipy.stats components not installed/imported. Cannot run SVM.")
        return results
    if metrics is None:
        print("    Skipping SVM evaluation: Metrics module not imported.")
        return results

    try:
        print(f"  Performing RandomizedSearchCV for SVM (n_iter={n_iter_search}, cv={cv_folds_tuning})...")

        # Chiến lược tuning của bạn:
        # C: trong khoảng [0.1, 10]
        # gamma: thử ‘scale’, ‘auto’, hoặc grid trong [1e-4, 1e-1]
        # Kernel: Thử 'rbf' (phổ biến) và 'linear'
        param_dist = {
            'C': expon(scale=1.0),  # Phân phối expon thường tốt cho C (tập trung gần 0 nhưng có thể lớn)
                                    # Hoặc uniform(0.1, 9.9) để có khoảng [0.1, 10]
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], # Thử các kernel khác nhau
            'gamma': ['scale', 'auto'] + list(uniform(1e-4, 1e-1).rvs(5)), # scale, auto, hoặc 5 giá trị ngẫu nhiên
            'degree': randint(2, 5),# Cho kernel 'poly'
        }

        base_svm = SVC(class_weight='balanced', probability=False, random_state=42)

        svm_search = RandomizedSearchCV(
            estimator=base_svm,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=cv_folds_tuning,
            verbose=0,
            random_state=42,
            n_jobs=-1,
            scoring='f1_macro'
        )

        svm_search.fit(X_train_scaled, y_train)

        print(f"  Best SVM params found: {svm_search.best_params_}")
        best_svm_model = svm_search.best_estimator_

        print("  Predicting with best SVM...")
        svm_preds = best_svm_model.predict(X_test_scaled)

        svm_metrics_results = metrics.calculate_classification_metrics(y_test, svm_preds, model_name="SVM")
        results.update(svm_metrics_results)
        results["SVM_BestParams"] = str(svm_search.best_params_)

    except Exception as e:
         print(f"Error during SVM execution: {e}")
         # Giữ nguyên kết quả NaN

    return results