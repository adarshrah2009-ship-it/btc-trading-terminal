import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import subprocess
import os
import sys
import lightgbm as lgb

# --- 1. AUTOMATIC BACKGROUND PIPELINE LAUNCHER ---
@st.cache_resource
def start_pipeline_background():
    if os.path.exists("pipeline.py"):
        subprocess.Popen([sys.executable, "pipeline.py"])

start_pipeline_background()

# --- 2. AUTO-REFRESH UI EVERY 3 SECONDS ---
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=3000, key="datarefresh")
except ImportError:
    pass

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BTC ML & Microstructure Terminal",
    page_icon="⚡",
    layout="wide"
)

DB_NAME = "btc_market_structure.db"

# --- 4. PERSISTENT TRADE DATABASE SETUP ---
def init_trade_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            pnl REAL,
            status TEXT,
            ml_confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

init_trade_db()

def log_trade_to_db(timestamp, side, entry_price, exit_price, pnl, status, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO paper_trades (timestamp, side, entry_price, exit_price, pnl, status, ml_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, side, entry_price, exit_price, pnl, status, confidence))
    conn.commit()
    conn.close()

def load_trade_history():
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# --- 5. LIGHTGBM INFERENCE ENGINE ---
@st.cache_resource
def load_or_init_lgbm():
    X_dummy = np.random.randn(100, 6)
    y_dummy = np.random.randint(0, 2, size=100)
    train_data = lgb.Dataset(X_dummy, label=y_dummy)
    params = {'objective': 'binary', 'verbosity': -1, 'learning_rate': 0.05, 'num_leaves': 15}
    model = lgb.train(params, train_data, num_boost_round=10)
    return model

lgb_model = load_or_init_lgbm()

def predict_signal_probability(features):
    features_array = np.array(features).reshape(1, -1)
    prob_long = lgb_model.predict(features_array)[0]
    return prob_long

# --- 6. SESSION STATE & SIDEBAR CONFIG ---
if "positions" not in st.session_state:
    st.session_state.positions = []

st.sidebar.title("⚙️ Terminal Config")
enable_paper_trader = st.sidebar.toggle("Enable ML Paper Trader", value=True)
min_confidence = st.sidebar.slider("Min ML Probability Threshold", 0.50, 0.95, 0.65, 0.05)

atr_sl_mult = st.sidebar.number_input("ATR Stop-Loss Multiplier", value=1.5, step=0.1)
atr_tp_mult = st.sidebar.number_input("ATR Take-Profit Multiplier", value=3.0, step=0.1)

# --- 7. DATA PROCESSING & FEATURE ENGINEERING ---
def load_and_process_data():
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM market_data ORDER BY timestamp DESC LIMIT 300", conn)
        conn.close()
        if df.empty:
            return pd.DataFrame()
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate Trend Indicators
        df['ema_100'] = df['mid_price'].ewm(span=100, adjust=False).mean()
        df['cvd_slope'] = df['cvd_acceleration'].rolling(window=10, min_periods=1).mean()
        
        # Calculate ATR
        df['high_low'] = df['mid_price'] - df['mid_price'].shift(1)
        df['atr'] = df['high_low'].abs().rolling(window=14, min_periods=1).mean().fillna(10.0)
        
        return df
    except Exception:
        return pd.DataFrame()

# --- 8. MAIN INTERFACE ---
st.title("⚡ Pro BTC Predictive Microstructure & ML Terminal")

df = load_and_process_data()

if df.empty:
    st.warning("⏳ Connecting to exchange WebSocket stream & populating SQLite entries...")
else:
    latest = df.iloc[-1]
    current_price = float(latest.get('mid_price', 0.0))
    current_micro = float(latest.get('micro_price', current_price))
    current_ofi = float(latest.get('ofi', 0.0))
    current_absorption = float(latest.get('absorption_ratio', 0.0))
    current_cvd_accel = float(latest.get('cvd_acceleration', 0.0))
    current_ema = float(latest.get('ema_100', current_price))
    current_cvd_slope = float(latest.get('cvd_slope', 0.0))
    current_atr = max(float(latest.get('atr', 10.0)), 5.0)

    # Feature Vector Construction
    micro_spread = current_micro - current_price
    ema_diff = current_price - current_ema
    feature_vector = [micro_spread, current_ofi, current_absorption, current_cvd_accel, ema_diff, current_atr]
    
    ml_prob_long = predict_signal_probability(feature_vector)
    ml_prob_short = 1.0 - ml_prob_long

    # 1. TOP METRICS DASHBOARD
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Micro-Price", f"${current_micro:,.2f}")
    col2.metric("OFI Signal", f"{current_ofi:,.2f}")
    col3.metric("Macro EMA (100)", f"${current_ema:,.2f}")
    col4.metric("Live ATR Volatility", f"${current_atr:,.2f}")
    col5.metric("ML Long Prob.", f"{ml_prob_long:.2%}")

    # 2. MICROSTRUCTURE & TREND CHART
    st.subheader("📈 Microstructure & Trend Regime Chart")
    fig = go.Figure()
    
    # Mid Price line (Solid Green)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mid_price'], name="Mid Price", line=dict(color="#00FFA3", width=2.5)))
    # Micro-Price line (Dashed Pink)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['micro_price'], name="Micro-Price", line=dict(color="#FF007A", width=1.5, dash='dash')))
    # 100 EMA Trend line (Dotted Blue)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_100'], name="100 EMA Trend", line=dict(color="#00E5FF", width=1.5, dash='dot')))
    
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 3. ML PAPER TRADING EXECUTION ENGINE
    st.markdown("---")
    st.subheader("🤖 Live ML Paper Trader Engine")

    if enable_paper_trader:
        if st.session_state.positions:
            for pos in list(st.session_state.positions):
                entry_price = pos["entry_price"]
                side = pos["side"]
                tp_val = pos["tp"]
                sl_val = pos["sl"]
                conf = pos["confidence"]
                
                pnl = (current_price - entry_price) if side == "LONG" else (entry_price - current_price)
                
                if pnl >= tp_val or pnl <= -sl_val:
                    exit_reason = "DYNAMIC TP 🎯" if pnl >= tp_val else "DYNAMIC SL 🛑"
                    time_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    log_trade_to_db(time_str, side, entry_price, current_price, pnl, exit_reason, conf)
                    st.session_state.positions.remove(pos)

        if not st.session_state.positions:
            is_bullish_regime = (current_price > current_ema) and (current_cvd_slope >= 0)
            is_bearish_regime = (current_price < current_ema) and (current_cvd_slope <= 0)
            
            calculated_sl = current_atr * atr_sl_mult
            calculated_tp = current_atr * atr_tp_mult
            
            if is_bullish_regime and ml_prob_long >= min_confidence and current_ofi > 0.05:
                st.session_state.positions.append({
                    "entry_price": current_price,
                    "side": "LONG",
                    "sl": calculated_sl,
                    "tp": calculated_tp,
                    "confidence": ml_prob_long,
                    "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
                })
                st.success("🚀 ML Executed LONG @ $" + f"{current_price:,.2f}" + " | Prob: " + f"{ml_prob_long:.2%}" + " | Dynamic SL: $" + f"{calculated_sl:.2f}" + " | TP: $" + f"{calculated_tp:.2f}")
            
            elif is_bearish_regime and ml_prob_short >= min_confidence and current_ofi < -0.05:
                st.session_state.positions.append({
                    "entry_price": current_price,
                    "side": "SHORT",
                    "sl": calculated_sl,
                    "tp": calculated_tp,
                    "confidence": ml_prob_short,
                    "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
                })
                st.success("🔻 ML Executed SHORT @ $" + f"{current_price:,.2f}" + " | Prob: " + f"{ml_prob_short:.2%}" + " | Dynamic SL: $" + f"{calculated_sl:.2f}" + " | TP: $" + f"{calculated_tp:.2f}")
            
            else:
                st.info("🔍 ML Scanner Active: Scanning for setups...")

        if st.session_state.positions:
            active = st.session_state.positions[0]
            entry_p = active["entry_price"]
            s = active["side"]
            live_pnl = (current_price - entry_p) if s == "LONG" else (entry_p - current_price)
            
            p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
            p_col1.metric("Active Order", f"{s}")
            p_col2.metric("Entry Price", f"${entry_p:,.2f}")
            p_col3.metric("Current Price", f"${current_price:,.2f}")
            p_col4.metric("Unrealized PnL", f"${live_pnl:,.2f}", delta=f"${live_pnl:,.2f}")
            p_col5.metric("Target SL / TP", f"-${active['sl']:.1f} / +${active['tp']:.1f}")

    # 4. PERSISTENT CLOSED TRADE HISTORY TABLE
    trades_df = load_trade_history()
    if not trades_df.empty:
        st.subheader("📜 Over-Night Executed Trade Logs")
        st.dataframe(trades_df, use_container_width=True)
