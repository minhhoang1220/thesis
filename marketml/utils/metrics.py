# marketml/pipeline/build_features.py
# File: marketml/utils/metrics.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix # Có thể dùng sau này
)

def calculate_classification_metrics(y_true, y_pred, model_name="Model"):
    """
    Tính toán, in và trả về các metrics cho bài toán phân loại.

    Args:
        y_true (array-like): Nhãn thực tế.
        y_pred (array-like): Nhãn dự đoán bởi mô hình.
        model_name (str): Tên của mô hình để thêm vào tên metrics.

    Returns:
        dict: Dictionary chứa các metrics đã tính toán.
              Returns dict with NaNs if calculation fails.
    """
    results = {}
    print(f"\n--- Evaluating {model_name} ---")
    try:
        # Đảm bảo input là NumPy array hoặc tương tự để xử lý nhất quán
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            print(f"  Warning: Length mismatch between y_true ({len(y_true)}) and y_pred ({len(y_pred)}). Cannot calculate metrics.")
            raise ValueError("Length mismatch") # Gây lỗi để trả về NaN

        accuracy = accuracy_score(y_true, y_pred)
        # Lấy report dạng dict để dễ trích xuất
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        # Lấy report dạng string để in
        report_str = classification_report(y_true, y_pred, zero_division=0, target_names=['Giảm (-1)', 'Đi ngang (0)', 'Tăng (1)']) # Thêm tên lớp

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Classification Report ({model_name}):")
        print(report_str)
        # print(f"  Confusion Matrix ({model_name}):") # Có thể bỏ comment nếu muốn xem
        # print(confusion_matrix(y_true, y_pred))

        # Lưu các metrics quan trọng vào dictionary
        results[f"{model_name}_Accuracy"] = accuracy
        results[f"{model_name}_F1_Macro"] = report_dict.get('macro avg', {}).get('f1-score', np.nan)
        results[f"{model_name}_F1_Weighted"] = report_dict.get('weighted avg', {}).get('f1-score', np.nan)
        # Thêm Precision/Recall nếu muốn
        results[f"{model_name}_Precision_Macro"] = report_dict.get('macro avg', {}).get('precision', np.nan)
        results[f"{model_name}_Recall_Macro"] = report_dict.get('macro avg', {}).get('recall', np.nan)

    except Exception as e:
        print(f"  Error calculating metrics for {model_name}: {e}")
        # Trả về NaN nếu có lỗi
        results[f"{model_name}_Accuracy"] = np.nan
        results[f"{model_name}_F1_Macro"] = np.nan
        results[f"{model_name}_F1_Weighted"] = np.nan
        results[f"{model_name}_Precision_Macro"] = np.nan
        results[f"{model_name}_Recall_Macro"] = np.nan

    return results

# Có thể thêm các hàm khác cho metrics hồi quy (RMSE, MAE) nếu cần sau này