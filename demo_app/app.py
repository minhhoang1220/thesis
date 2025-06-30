import streamlit as st

st.set_page_config(
    page_title="MarketML Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("MarketML: A Comparative Study on Machine Learning for Market Forecasting & Portfolio Optimization 🤖")
st.markdown("""
_A demo application for the thesis research_
""")
st.markdown("---")

# Overview
st.header("Overview")
st.markdown("""
This application visualizes results from a research pipeline comparing modern Machine Learning algorithms in two main areas:

1.  **Market Trend Forecasting:** Predicting Up/Down/Sideways trends of stock prices.
2.  **Portfolio Optimization:** Building investment strategies based on forecast signals.

The data used covers **15 leading international companies** from **2020–2024**.

👈 **Select a page from the left sidebar to explore more details!**
""")

# Main Methods Used
st.subheader("Main Methods Used")

col1, col2 = st.columns(2)
with col1:
    st.info("**Market Forecasting Algorithms**")
    st.markdown("""
    - ARIMA
    - Random Forest
    - XGBoost
    - SVM
    - LSTM (Long Short-Term Memory)
    - Transformer
    """)

with col2:
    st.success("**Portfolio Optimization Techniques**")
    st.markdown("""
    - Mean-Variance (Markowitz)
    - Black-Litterman
    - Reinforcement Learning (PPO)
    """)
