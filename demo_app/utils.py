# demo_app/utils.py
import streamlit as st
import pandas as pd
from pathlib import Path
import json

# --- Path definitions ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKETML_ROOT = PROJECT_ROOT / 'marketml'

ENRICHED_DATA_FILE = MARKETML_ROOT / 'data_processed' / 'price_data_enriched_v2.csv'
CLASSIFICATION_PROBS_FILE = MARKETML_ROOT / 'results_output' / 'classification_probabilities.csv'
MODEL_PERF_SUMMARY_FILE = MARKETML_ROOT / 'results_output' / 'model_performance_summary.csv'
PORTFOLIO_SUMMARY_FILE = MARKETML_ROOT / 'results_output' / 'portfolio_strategies_summary.csv'
RESULTS_OUTPUT_DIR = MARKETML_ROOT / 'results_output'
PERFORMANCE_PLOTS_DIR = RESULTS_OUTPUT_DIR / 'performance_plots'

# --- Data loading functions ---
@st.cache_data
def load_price_data():
    """Load enriched price data."""
    if not ENRICHED_DATA_FILE.exists():
        st.error(f"ERROR: Price data file not found at {ENRICHED_DATA_FILE}")
        return pd.DataFrame()
    df = pd.read_csv(ENRICHED_DATA_FILE, parse_dates=['date'])
    return df

@st.cache_data
def load_forecast_signals():
    """Load forecast signals from classification_probabilities.csv."""
    if not CLASSIFICATION_PROBS_FILE.exists():
        st.error(f"ERROR: Forecast signals file not found at {CLASSIFICATION_PROBS_FILE}")
        return pd.DataFrame()
    df_probs = pd.read_csv(CLASSIFICATION_PROBS_FILE, parse_dates=['date'])
    return df_probs

@st.cache_data
def load_model_performance():
    """Load model performance summary."""
    if not MODEL_PERF_SUMMARY_FILE.exists():
        st.warning(f"Model performance summary not found at: {MODEL_PERF_SUMMARY_FILE}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(MODEL_PERF_SUMMARY_FILE, index_col=0)
        return df
    except Exception as e:
        st.error(f"ERROR reading `model_performance_summary.csv`: {e}")
        return pd.DataFrame()

@st.cache_data
def load_confusion_matrix(model_name):
    """Load confusion matrix data for a specific model."""
    path = RESULTS_OUTPUT_DIR / f"{model_name.lower()}_confusion_matrix.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_feature_importance(model_name):
    """Load feature importance data."""
    path = RESULTS_OUTPUT_DIR / f"{model_name.lower()}_feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def get_plot_path(plot_name):
    """Get the path of a pre-generated plot file."""
    path = PERFORMANCE_PLOTS_DIR / plot_name
    return path if path.exists() else None

STRATEGY_NAMES_MAPPING = {
    'Markowitz': 'Markowitz',
    'BlackLitterman': 'Black-Litterman',
    'RL_Strategy': 'RL (PPO)'
}

@st.cache_data
def load_portfolio_daily_performance():
    """Load daily performance of portfolio strategies."""
    strategies = {
        'Markowitz': RESULTS_OUTPUT_DIR / "markowitz_performance_daily.csv",
        'BlackLitterman': RESULTS_OUTPUT_DIR / "blacklitterman_performance_daily.csv",
        'RL_Strategy': RESULTS_OUTPUT_DIR / "ppo_strategy_performance_daily.csv",
    }
    all_perf = {}
    for name, path in strategies.items():
        if path.exists():
            df = pd.read_csv(path, parse_dates=['date'], index_col='date')
            all_perf[name] = df
    return all_perf

@st.cache_data
def load_portfolio_summary():
    """Load portfolio strategies performance summary."""
    if not PORTFOLIO_SUMMARY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PORTFOLIO_SUMMARY_FILE, index_col=0)
    return df
