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
```

## 📦 Data Files

- `yahoo_price_data_fixed.csv`: International firms' structured price data
- `yahoo_financial_data.csv`: International firms' financial statements
- `yahoo_price_data_fixed`: Vietnam firms' structured price data
- `financial_data_vn`: Vietnam firms' financial statements

## 📊 Final Output

Evaluate and compare the models on:
- Forecasting accuracy (e.g., MAPE, RMSE)
- Portfolio performance metrics (e.g., Sharpe Ratio, Return, Volatility)

---

## ⚙️ Technology

- Python
- `gymnasium` for RL
- Pandas, NumPy, Scikit-learn, PyTorch (as needed)

---

## 🔍 Objective

The ultimate goal is to draw insights from the comparison of algorithms and portfolio optimization methods across two groups of companies, delivering an empirical foundation for model selection in financial applications.

---

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

<h2>PRICE DATA</h2>
<table border="1">
  <thead>
    <tr>
      <th>Date</th>
      <th>Ticker</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Market</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>2019-01-02</td><td>AAPL</td><td>36.944462</td><td>37.889005</td><td>36.787037</td><td>37.667179</td><td>148158800.0</td><td>Global</td></tr>
    <tr><td>2019-01-02</td><td>AMZN</td><td>73.260002</td><td>77.667999</td><td>73.046501</td><td>76.956497</td><td>159662000.0</td><td>Global</td></tr>
    <tr><td>2019-01-02</td><td>F</td><td>5.479391</td><td>5.835952</td><td>5.443007</td><td>5.748631</td><td>47494400.0</td><td>Global</td></tr>
    <tr><td>2019-01-02</td><td>GOOGL</td><td>51.115623</td><td>52.787132</td><td>51.020079</td><td>52.483086</td><td>31868000.0</td><td>Global</td></tr>
    <tr><td>2019-01-02</td><td>JNJ</td><td>107.615512</td><td>107.825485</td><td>106.061705</td><td>107.296349</td><td>7631700.0</td><td>Global</td></tr>
    <tr><td>2019-01-02</td><td>BID.VN</td><td>20058.873047</td><td>20058.873047</td><td>19504.919922</td><td>19534.076172</td><td>1381735.0</td><td>Vietnam</td></tr>
    <tr><td>2019-01-02</td><td>CTG.VN</td><td>13347.275391</td><td>13485.948242</td><td>13173.933594</td><td>13173.933594</td><td>3634939.0</td><td>Vietnam</td></tr>
    <tr><td>2019-01-02</td><td>HPG.VN</td><td>10260.394531</td><td>10443.023438</td><td>10227.189453</td><td>10227.189453</td><td>12637219.0</td><td>Vietnam</td></tr>
    <tr><td>2019-01-02</td><td>VCB.VN</td><td>23924.726562</td><td>24102.277344</td><td>23747.177734</td><td>23791.566406</td><td>2808551.0</td><td>Vietnam</td></tr>
    <tr><td>2019-01-03</td><td>BID.VN</td><td>19534.076172</td><td>19709.007812</td><td>18659.416016</td><td>18659.416016</td><td>1899306.0</td><td>Vietnam</td></tr>
  </tbody>
</table>

<h2>FINANCIAL DATA</h2>
<table border="1">
  <thead>
    <tr>
      <th>Date</th>
      <th>Ticker</th>
      <th>ROA</th>
      <th>ROE</th>
      <th>EPS</th>
      <th>P/E Ratio</th>
      <th>Debt/Equity</th>
      <th>Dividend Yield</th>
      <th>Revenue</th>
      <th>Net Income</th>
      <th>Market</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>2019-12-31</td><td>AAPL</td><td>0.22519</td><td>1.3652</td><td>6.29</td><td>35.391098</td><td>145.00</td><td>0.45</td><td>395760009216</td><td>96150003712</td><td>Global</td></tr>
    <tr><td>2020-12-31</td><td>AAPL</td><td>0.22519</td><td>1.3652</td><td>6.29</td><td>35.391098</td><td>145.00</td><td>0.45</td><td>395760009216</td><td>96150003712</td><td>Global</td></tr>
    <tr><td>2021-12-31</td><td>AAPL</td><td>0.22519</td><td>1.3652</td><td>6.29</td><td>35.391098</td><td>145.00</td><td>0.45</td><td>395760009216</td><td>96150003712</td><td>Global</td></tr>
    <tr><td>2022-12-31</td><td>AAPL</td><td>0.22519</td><td>1.3652</td><td>6.29</td><td>35.391098</td><td>145.00</td><td>0.45</td><td>395760009216</td><td>96150003712</td><td>Global</td></tr>
    <tr><td>2023-12-31</td><td>AAPL</td><td>0.22519</td><td>1.3652</td><td>6.29</td><td>35.391098</td><td>145.00</td><td>0.45</td><td>395760009216</td><td>96150003712</td><td>Global</td></tr>
    <tr><td>2019-12-31</td><td>VIC.VN</td><td>2.18000</td><td>6.8700</td><td>2,199.39</td><td>50.150000</td><td>97.38</td><td>0.00</td><td>130,161</td><td>7,546</td><td>Vietnam</td></tr>
    <tr><td>2020-12-31</td><td>VIC.VN</td><td>1.32000</td><td>4.2600</td><td>1,586.36</td><td>66.970000</td><td>91.48</td><td>0.00</td><td>110,755</td><td>5,465</td><td>Vietnam</td></tr>
    <tr><td>2021-12-31</td><td>VIC.VN</td><td>-0.59000</td><td>-1.7000</td><td>-649.99</td><td>-133.710000</td><td>76.48</td><td>0.00</td><td>125,781</td><td>-2,514</td><td>Vietnam</td></tr>
    <tr><td>2022-12-31</td><td>VIC.VN</td><td>1.75000</td><td>5.9500</td><td>2,269.88</td><td>23.360000</td><td>123.87</td><td>0.00</td><td>101,810</td><td>8,782</td><td>Vietnam</td></tr>
    <tr><td>2023-12-31</td><td>VIC.VN</td><td>0.35000</td><td>1.5200</td><td>556.11</td><td>78.870000</td><td>143.87</td><td>0.00</td><td>161,453</td><td>2,157</td><td>Vietnam</td></tr>
  </tbody>
</table>

