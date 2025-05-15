# File: marketml/models/svm_model.py
import pandas as pd
import numpy as np
try:
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SVC = None

# Đảm bảo có thể import từ thư mục cha hoặc utils đã trong sys.path
try:
    from marketml.utils import metrics
except ModuleNotFoundError:
    try:
        from ..utils import metrics
    except ImportError:
        print("Error: Cannot import metrics module in svm_model.py")
        metrics = None

def run_svm_evaluation(X_train_scaled, y_train, X_test_scaled, y_test, **kwargs):
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
    print("\n--- Training and Evaluating SVM Model ---")
    results = {}
    default_metrics = {"SVM_Accuracy": np.nan, "SVM_F1_Macro": np.nan, "SVM_F1_Weighted": np.nan,
                       "SVM_Precision_Macro": np.nan, "SVM_Recall_Macro": np.nan}
    results.update(default_metrics) # Khởi tạo

    if not SKLEARN_AVAILABLE or SVC is None:
        print("Error: scikit-learn not installed or SVC not imported. Cannot run SVM.")
        return results
    if metrics is None:
        print("    Skipping SVM evaluation: Metrics module not imported.")
        return results

    try:
        print("  Training SVM...")
        # Lấy tham số từ kwargs hoặc dùng giá trị mặc định
        # SVM có thể nhạy cảm với việc chọn kernel và các tham số C, gamma
        # 'rbf' là kernel phổ biến. probability=True nếu muốn lấy xác suất dự đoán sau này.
        # class_weight='balanced' để xử lý mất cân bằng lớp.
        C_param = kwargs.get('C', 1.0)
        kernel_param = kwargs.get('kernel', 'rbf')
        gamma_param = kwargs.get('gamma', 'scale') # 'scale' hoặc 'auto' hoặc một số float

        svm_model = SVC(
            C=C_param,
            kernel=kernel_param,
            gamma=gamma_param,
            class_weight='balanced',
            probability=False, # Đặt True nếu cần predict_proba, nhưng chậm hơn
            random_state=42
        )

        svm_model.fit(X_train_scaled, y_train)

        print("  Predicting with SVM...")
        svm_preds = svm_model.predict(X_test_scaled)

        # Đánh giá
        svm_metrics = metrics.calculate_classification_metrics(y_test, svm_preds, model_name="SVM")
        results.update(svm_metrics) # Ghi đè NaN

    except Exception as e:
         print(f"Error during SVM execution: {e}")
         # Giữ nguyên kết quả NaN đã khởi tạo

    return results