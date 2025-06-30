# demo_app/pages/3_Pipeline_and_Methods.py
import sys
from pathlib import Path

# Add sys.path for project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import load_feature_importance
from marketml.configs import configs

st.set_page_config(layout="wide")
st.title("🔄 Pipeline and Model Explanation")

# Pipeline Flow Section
st.header("Data Processing Pipeline Flow")
try:
    st.image("demo_app/pipeline_flow_updated.png", caption="MarketML System Pipeline Flow Diagram")
except FileNotFoundError:
    st.warning("File 'pipeline_flow_updated.png' not found. Please create and place it in the 'demo_app' directory.")

# --- Tab Structure ---
tab1, tab2 = st.tabs(["🤖 Model Explainability", "⚙️ Pipeline Details"])

with tab1:
    st.subheader("Model Explainability")
    
    main_signal_model_name = configs.SOFT_SIGNAL_MODEL_NAME
    
    st.markdown(f"#### Feature Importance")
    st.markdown(
        f"The chart below shows which features have the most influence on the predictions of the **`{main_signal_model_name}`** model. "
        "This helps to 'look inside' the model and understand what information it relies on for its decisions."
    )
            
    df_importance = load_feature_importance(main_signal_model_name)
    
    if not df_importance.empty:
        top_features = df_importance.head(15).sort_values('importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title_text=f"Top 15 Most Important Features of Model {main_signal_model_name}",
            xaxis_title="Importance Score",
            yaxis_title="Feature Name",
            height=500,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            f"**Feature Importance data for model `{main_signal_model_name}` not found.**\n\n"
            f"This may be because:\n"
            f"1. The pipeline `04_generate_signals.py` has not been run successfully.\n"
            f"2. The file `marketml/results_output/{main_signal_model_name.lower()}_feature_importance.csv` was deleted or moved.\n\n"
            "Please rerun the pipeline to generate this file."
        )

with tab2:
    st.subheader("Details of Methods")
    with st.expander("🔹 Comparison of Forecasting Models", expanded=True):
        st.markdown("""
        This study compares a range of algorithms from classical to modern:
        - **ARIMA:** Traditional statistical model, used as a baseline.
        - **Random Forest / XGBoost / SVM:** Powerful supervised machine learning models, perform well on tabular data after feature extraction.
        - **LSTM / Transformer:** Deep learning models designed for time series data, capable of learning long-term dependencies.
        
        **Goal:** Identify which model provides the best trend forecasting accuracy on this dataset.
        """)

    with st.expander("🔹 Comparison of Portfolio Optimization Strategies", expanded=True):
        st.markdown("""
        - **Markowitz (Mean-Variance):** Foundation of modern portfolio theory. Relies only on historical data (mean returns and covariance matrix) for optimization. Commonly used as a baseline.
        - **Black-Litterman:** An improvement over Markowitz, allows combining **investor views** (from model signals like XGBoost) with **market expectations**. This helps create more stable and realistic portfolios.
        - **Reinforcement Learning (RL - PPO/A2C):** A completely different approach. Instead of a one-time calculation, RL trains an **agent** through thousands of trial-and-error episodes in a simulated environment. The agent learns flexibly to make asset allocation decisions to maximize long-term reward.
        """)
