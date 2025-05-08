# File: marketml/models/arima_model.py
import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
# Đảm bảo có thể import từ thư mục cha hoặc utils đã trong sys.path
try:
    from marketml.utils import metrics
except ModuleNotFoundError:
    # Thử import tương đối nếu chạy trực tiếp file này (không khuyến khích)
    try:
        from ..utils import metrics
    except ImportError:
        print("Error: Cannot import metrics module in arima_model.py")
        metrics = None # Hoặc raise lỗi

def run_arima_evaluation(train_df, test_df, target_col_trend, trend_threshold):
    """
    Huấn luyện, dự đoán và đánh giá mô hình ARIMA cho tất cả tickers trong split.

    Args:
        train_df (pd.DataFrame): Dữ liệu train của split hiện tại (đã có target).
        test_df (pd.DataFrame): Dữ liệu test của split hiện tại (đã có target).
        target_col_trend (str): Tên cột target trend.
        trend_threshold (float): Ngưỡng xác định trend.

    Returns:
        dict: Dictionary chứa kết quả metrics của ARIMA.
    """
    print("\n--- Training and Evaluating ARIMA Model ---")
    results_this_split = {}
    arima_predictions_all = []
    arima_actual_trends_all = []

    # Mặc định kết quả là NaN
    default_metrics = {"ARIMA_Accuracy": np.nan, "ARIMA_F1_Macro": np.nan, "ARIMA_F1_Weighted": np.nan,
                       "ARIMA_Precision_Macro": np.nan, "ARIMA_Recall_Macro": np.nan}
    results_this_split.update(default_metrics) # Khởi tạo

    if metrics is None: # Không thể chạy nếu thiếu module metrics
        print("    Skipping ARIMA evaluation: Metrics module not imported.")
        return results_this_split

    for ticker in train_df['ticker'].unique():
        # print(f"  Processing ARIMA for ticker: {ticker}") # Tắt bớt
        train_ticker_df = train_df[train_df['ticker'] == ticker]
        test_ticker_df = test_df[test_df['ticker'] == ticker]

        y_train_arima_input = train_ticker_df['close'].pct_change().dropna()
        y_test_arima_input = test_ticker_df['close'].pct_change().dropna()

        potential_trends = test_df.loc[test_df['ticker'] == ticker, target_col_trend]
        y_test_trend_ticker = potential_trends.reindex(y_test_arima_input.index).dropna()

        if y_train_arima_input.empty or y_test_arima_input.empty or y_test_trend_ticker.empty:
            continue

        try:
            adf_result = adfuller(y_train_arima_input); p_value = adf_result[1]; d = 0
            if p_value > 0.05: d = 1
            auto_model = pm.auto_arima(y_train_arima_input, d=d, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=False)
            n_periods = len(y_test_arima_input)
            arima_preds_pct = auto_model.predict(n_periods=n_periods)

            if len(arima_preds_pct) == len(y_test_trend_ticker): y_test_trend_ticker_aligned = y_test_trend_ticker
            elif len(y_test_trend_ticker) > len(arima_preds_pct): y_test_trend_ticker_aligned = y_test_trend_ticker.iloc[-len(arima_preds_pct):]
            else: arima_preds_pct = arima_preds_pct[-len(y_test_trend_ticker):]; y_test_trend_ticker_aligned = y_test_trend_ticker

            arima_predictions_all.extend(arima_preds_pct); arima_actual_trends_all.extend(y_test_trend_ticker_aligned)
        except Exception as e:
            print(f"    Error processing ARIMA for {ticker}: {e}")

    if arima_predictions_all:
        arima_preds_trend = np.select([(np.array(arima_predictions_all) > trend_threshold), (np.array(arima_predictions_all) < -trend_threshold)], [1, -1], default=0)
        arima_metrics_results = metrics.calculate_classification_metrics(arima_actual_trends_all, arima_preds_trend, model_name="ARIMA")
        results_this_split.update(arima_metrics_results) # Ghi đè kết quả NaN ban đầu
    else:
        print("    ARIMA: No predictions were generated for this split.")
        # Giữ nguyên kết quả NaN đã khởi tạo

    return results_this_split