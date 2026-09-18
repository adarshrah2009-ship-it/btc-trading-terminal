import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import subprocess
import os

# --- AUTOMATIC BACKGROUND PIPELINE LAUNCHER ---
@st.cache_resource
def start_pipeline():
    if os.path.exists("pipeline.py"):
        subprocess.Popen(["python", "pipeline.py"])

start_pipeline()

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="BTC Predictive Microstructure Terminal",
    page_icon="⚡",
    layout="wide"
)

# --- SIDEBAR TERMINAL CONFIG ---
st.sidebar.title("⚙️ Terminal Config")
enable_paper_trader = st.sidebar.toggle("Enable AI Paper Trader", value=True)
min_confidence = st.sidebar.slider("Min AI Confidence Threshold", 0.50, 0.95, 0.65, 0.05)

position_size = st.sidebar.number_input("Position Size ($)", value=2000.0, step=100.0)
take_profit = st.sidebar.number_input("Take Profit ($)", value=50.0, step=5.0)
stop_loss = st.sidebar.number_input("Stop Loss ($)", value=25.0, step=5.0)
fee_pct = st.sidebar.number_input("Binance Fee %", value=0.04, step=0.01)

# --- DATABASE FETCHING ---
DB_NAME = "btc_market_structure.db"

def load_data():
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM market_data ORDER BY timestamp DESC LIMIT 500", conn)
        conn.close()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        return df
    except Exception:
        return pd.DataFrame()

# --- MAIN TERMINAL INTERFACE ---
st.title("⚡ Pro BTC Predictive Microstructure Terminal")

df = load_data()

if df.empty:
    st.warning("⏳ Waiting for database entries from pipeline.py...")
else:
    # Latest Order Flow Metrics
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Micro-Price", f"${latest.get('micro_price', 0.0):,.2f}")
    col2.metric("Order Flow Imbalance (OFI)", f"{latest.get('ofi', 0.0):,.2f}")
    col3.metric("Absorption Ratio", f"{latest.get('absorption_ratio', 0.0):.4f}")
    col4.metric("CVD Acceleration", f"{latest.get('cvd_acceleration', 0.0):.2f}")

    # Micro-Price vs Real Mid Price Chart
    st.subheader("📈 Microstructure Dynamics")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mid_price'], name="Mid Price", line=dict(color="#00FFA3", width=1.5)))
    if 'micro_price' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['micro_price'], name="Micro-Price (Predictive)", line=dict(color="#FF007A", width=1.5, dash='dash')))
    
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Raw Microstructure Features Data Table
    with st.expander("📊 View Live Feature Feed"):
        st.dataframe(df.tail(20), use_container_width=True)
