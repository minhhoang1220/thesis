# marketml/utils/environment_setup.py
import warnings
import logging
from pathlib import Path

_project_logger_configured = False # Biến cờ toàn cục cho module này

def suppress_common_warnings():
    """
    Bỏ qua các warnings thường gặp không quan trọng từ các thư viện.
    """
    print("Suppressing common warnings...")
    # Bỏ qua FutureWarning chung (thường từ pandas, numpy)
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Bỏ qua RuntimeWarning chung (ví dụ: chia cho zero trong numpy khi tính metric)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Statsmodels specific warnings
    try:
        from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
        warnings.filterwarnings("ignore", category=ValueWarning, module='statsmodels')
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module='statsmodels')
    except ImportError:
        pass # Statsmodels có thể chưa được import ở mọi nơi

    # Scikit-learn specific warnings (ví dụ: ConvergenceWarning cho một số solver)
    try:
        from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
        warnings.filterwarnings("ignore", category=SklearnConvergenceWarning)
    except ImportError:
        pass
    
    # Pmdarima specific warnings (ví dụ: UserWarning về seasonality)
    # warnings.filterwarnings("ignore", category=UserWarning, module='pmdarima') # Có thể quá mạnh, cân nhắc

    # Keras/TensorFlow deprecation warnings (có thể nhiều)
    # warnings.filterwarnings("ignore", category=DeprecationWarning, module='tensorflow')

    print("Warnings suppressed.")

def setup_basic_logging(log_level=logging.INFO, log_file_name="project.log"):
    global _project_logger_configured
    
    logger = logging.getLogger("marketml_project")  # Luôn lấy logger gốc này

    # Chỉ cấu hình nếu chưa được cấu hình trước đó
    if not _project_logger_configured:
        project_root = Path(__file__).resolve().parents[2]
        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = logs_dir / log_file_name

        # Xóa tất cả handlers hiện có để tránh lặp
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        logger.setLevel(log_level)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Chỉ thêm StreamHandler nếu chưa có
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        # Chỉ thêm FileHandler nếu chưa có
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        logger.propagate = False  # Quan trọng: ngăn không cho log lan truyền lên logger cha
        _project_logger_configured = True

    return logger


# Bạn có thể thêm các hàm thiết lập khác ở đây nếu cần
# Ví dụ: thiết lập seed cho random number generators
def set_random_seeds(seed_value=42):
    import random
    import numpy as np
    try:
        import tensorflow as tf
    except ImportError:
        tf = None
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    if tf:
        tf.random.set_seed(seed_value)
    logging.getLogger("marketml_project").info(f"Random seeds set to: {seed_value}")