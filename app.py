import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import subprocess
import os
import sys

# --- AUTOMATIC BACKGROUND PIPELINE LAUNCHER ---
@st.cache_resource
def start_pipeline_background():
    if os.path.exists("pipeline.py"):
        subprocess.Popen([sys.executable, "pipeline.py"])

start_pipeline_background()

# Safe auto-refresh import
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=3000, key="datarefresh")
except ImportError:
    pass

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BTC Predictive Microstructure Terminal",
    page_icon="⚡",
    layout="wide"
)

# --- INITIALIZE IN-MEMORY SESSION STATE FOR PAPER TRADES ---
if "positions" not in st.session_state:
    st.session_state.positions = []
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []

# --- SIDEBAR TERMINAL CONFIG ---
st.sidebar.title("⚙️ Terminal Config")
enable_paper_trader = st.sidebar.toggle("Enable AI Paper Trader", value=True)
min_confidence = st.sidebar.slider("Min AI Confidence Threshold", 0.50, 0.95, 0.65, 0.05)

position_size = st.sidebar.number_input("Position Size ($)", value=2000.0, step=100.0)
take_profit = st.sidebar.number_input("Take Profit ($)", value=50.0, step=5.0)
stop_loss = st.sidebar.number_input("Stop Loss ($)", value=25.0, step=5.0)

# --- DATABASE FETCHING ---
DB_NAME = "btc_market_structure.db"

def load_data():
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM market_data ORDER BY timestamp DESC LIMIT 300", conn)
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
    st.warning("⏳ Connecting to exchange WebSocket stream & initializing SQLite entries...")
else:
    latest = df.iloc[-1]
    current_price = latest.get('mid_price', 0.0)
    current_ofi = latest.get('ofi', 0.0)
    
    # 1. Top Bar Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Micro-Price", f"${latest.get('micro_price', 0.0):,.2f}")
    col2.metric("Order Flow Imbalance (OFI)", f"{current_ofi:,.2f}")
    col3.metric("Absorption Ratio", f"{latest.get('absorption_ratio', 0.0):.4f}")
    col4.metric("CVD Acceleration", f"{latest.get('cvd_acceleration', 0.0):.2f}")

    # 2. Main Microstructure Chart
    st.subheader("📈 Microstructure Dynamics")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mid_price'], name="Mid Price", line=dict(color="#00FFA3", width=1.5)))
    if 'micro_price' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['micro_price'], name="Micro-Price", line=dict(color="#FF007A", width=1.5, dash='dash')))
    
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 3. AI PAPER TRADING EXECUTION ENGINE
    st.markdown("---")
    st.subheader("🤖 Live AI Paper Trader Engine")

    if enable_paper_trader:
        # Check current active position management (Take-Profit / Stop-Loss)
        if st.session_state.positions:
            for pos in list(st.session_state.positions):
                entry_price = pos["entry_price"]
                side = pos["side"]
                pnl = (current_price - entry_price) if side == "LONG" else (entry_price - current_price)
                
                # Check exit conditions
                if pnl >= take_profit or pnl <= -stop_loss:
                    exit_reason = "TAKE PROFIT 🎯" if pnl >= take_profit else "STOP LOSS 🛑"
                    st.session_state.trade_history.append({
                        "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                        "side": side,
                        "entry": entry_price,
                        "exit": current_price,
                        "pnl": pnl,
                        "status": exit_reason
                    })
                    st.session_state.positions.remove(pos)

        # Trigger new signals if no open position exists
        if not st.session_state.positions:
            # Normalized signal trigger simulation based on threshold
            raw_confidence = min(abs(current_ofi) / 10.0 + 0.50, 0.99)
            
            if raw_confidence >= min_confidence and abs(current_ofi) > 0.1:
                signal_side = "LONG" if current_ofi > 0 else "SHORT"
                st.session_state.positions.append({
                    "entry_price": current_price,
                    "side": signal_side,
                    "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
                })
                st.success(f"🚀 **New AI Signal Executed:** Opened **{signal_side}** position @ **${current_price:,.2f}** (Confidence: {raw_confidence:.2f})")
            else:
                st.info(f"🔍 AI Scanning Feed... Signal Confidence ({raw_confidence:.2f}) below threshold ({min_confidence:.2f}).")

        # Display Open Position Dashboard
        if st.session_state.positions:
            active_pos = st.session_state.positions[0]
            entry_p = active_pos["entry_price"]
            s = active_pos["side"]
            live_pnl = (current_price - entry_p) if s == "LONG" else (entry_p - current_price)
            
            pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
            pos_col1.metric("Active Position", f"{s}")
            pos_col2.metric("Entry Price", f"${entry_p:,.2f}")
            pos_col3.metric("Current Price", f"${current_price:,.2f}")
            pos_col4.metric("Unrealized PnL", f"${live_pnl:,.2f}", delta=f"${live_pnl:,.2f}")
    else:
        st.write("Paper Trader is disabled. Enable it in the sidebar to run simulations.")

    # 4. Closed Trade History Table
    if st.session_state.trade_history:
        st.subheader("📜 Closed Trade History")
        st.dataframe(pd.DataFrame(st.session_state.trade_history), use_container_width=True)

    # 5. Raw Data Stream Viewer
    with st.expander("📊 View Live Microstructure Database Feed"):
        st.dataframe(df.tail(20), use_container_width=True)
