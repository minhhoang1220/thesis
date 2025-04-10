# MarketML: Comparative Study on Machine Learning for Market Forecasting & Portfolio Optimization

## 📚 Overview

This project is part of a research thesis titled **"A Comparative Study of Machine Learning Algorithms for Market Trend Forecasting and Portfolio Optimization"**, conducted from **March 24, 2025**. It integrates modern machine learning techniques and optimization strategies to build robust models for:

- 📈 **Market Trend Forecasting**
- 📊 **Portfolio Optimization**

## 🧠 Key Methods

### Market Forecasting Algorithms
- ARIMA
- LSTM
- Random Forest
- XGBoost
- SVM
- Transformer

### Portfolio Optimization Techniques
- Mean-Variance (Markowitz)
- Black-Litterman
- Reinforcement Learning-based (Gymnasium)

## 🏢 Dataset

- **15 international** and **15 Vietnamese top-tier firms** (cross-sector)
- **Time range**: 2019 - 2024
- **Sources**:
  - Price data: Yahoo Finance (via code)
  - Financial data: 
    - International: Yahoo Finance (via code)
    - Vietnam: Collected manually on vietstock.vn

## 📁 Project Structure (`.ndmh`)

```plaintext
.ndmh/
│
├── marketml/                   # Core logic
│   ├── configs/                # Configuration settings
│   │   ├── __init__.py
│   │   └── configs.py
│   │
│   ├── data/                   # Data handling modules
│   │   ├── __init__.py
│   │   └── loader/
│   │       ├── __init__.py
│   │       ├── finance_loader.py
│   │       ├── price_loader.py
│   │       └── preprocess.py
│   │
│   ├── log/                    # Logging utilities (TBD)
│   ├── model/                  # Forecasting & optimization models (TBD)
│   ├── utils/                  # Shared utility functions
│   │   ├── __init__.py
│   │   └── __version__.py
│
├── notebooks/
│   └── 01_EDA.ipynb            # Exploratory Data Analysis
│
├── tests/
│   └── test_loader.py          # Unit tests for data loaders
│
├── run.py                      # Main pipeline entry point
├── setup.py                    # Packaging
├── requirements.txt            # Dependencies
└── README.md                   # Project overview

## 📦 Data Files

- `yahoo_price_data_fixed.csv`: International firms' structured price data
- `yahoo_financial_data.csv`: International firms' financial statements
- `yahoo_price_data_fixed`: Vietnam firms' structured price data
- `financial_data_vn`: Vietnam firms' financial statements

## 📊 Final Output

Evaluate and compare the models on:
- Forecasting accuracy (e.g., MAPE, RMSE)
- Portfolio performance metrics (e.g., Sharpe Ratio, Return, Volatility)

## ⚙️ Technology

- Python
- `gymnasium` for RL
- Pandas, NumPy, Scikit-learn, PyTorch (as needed)

---

## 🔍 Objective

The ultimate goal is to draw insights from the comparison of algorithms and portfolio optimization methods across two groups of companies, delivering an empirical foundation for model selection in financial applications.

```

## 🚀 How to Run

### 1. Install dependencies

Create a virtual environment and install all required packages:

```bash
python -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Mount Data Folder (if using VM)

Ensure you have mounted the shared data folder from Windows to Ubuntu at:

```bash
/mnt/shared-data/Khoa_Luan/
```

Data files should be organized as:

```bash
/mnt/shared-data/Khoa_Luan/
├── Data_Global/
│   └── price/yahoo_price_data_fixed.csv
│   └── financial/yahoo_financial_data.csv
├── Data_VN/
    └── price/yahoo_price_data_fixed.csv
    └── financial/financial_data_vn.csv
```

### 3. Run the pipeline

```bash
python run.py
```

## 🔬 Example Output (Partial)
✅ PRICE DATA
        Date    Ticker    Open     High     Low     Close    Volume   Market
0  2019-01-02   AAPL     36.94    37.88    36.78    37.66    148M     Global
...
5  2019-01-02  BID.VN  20058.87  20058.87 19504.91 19534.07  1.3M     Vietnam

✅ FINANCIAL DATA
        Date   Ticker   ROA   ROE   EPS   P/E Ratio   D/E   Dividend Yield ...
0  2019-12-31  AAPL    0.22  1.36  6.29    35.39     145.0      0.45         Global
...
5  31/12/2019 VIC.VN  2.18  6.87  2,199   50.15       97.38      0.00        Vietnam
