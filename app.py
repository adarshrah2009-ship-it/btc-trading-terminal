import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import subprocess
import os
import sys
import lightgbm as lgb
import time

# --- 1. AUTOMATIC BACKGROUND PIPELINE & DB INITIALIZER ---
DB_NAME = "btc_market_structure.db"

def ensure_db_seeded():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mid_price REAL,
            micro_price REAL,
            ofi REAL,
            mlofi REAL,
            cvd_acceleration REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            gross_pnl REAL,
            net_pnl REAL,
            fee_slippage REAL,
            status TEXT,
            ml_confidence REAL
        )
    ''')
    conn.commit()
    
    # Check if market_data has entries; if empty, seed initial data instantly
    cursor.execute("SELECT COUNT(*) FROM market_data")
    count = cursor.fetchone()[0]
    if count == 0:
        now = pd.Timestamp.now()
        base_price = 80472.50
        rows = []
        for i in range(30):
            ts = (now - pd.Timedelta(seconds=30 - i)).strftime("%Y-%m-%d %H:%M:%S")
            p = base_price + np.random.randn() * 10
            rows.append((ts, p, p + 0.5, np.random.randn(), np.random.randn(), np.random.randn()))
        
        cursor.executemany('''
            INSERT INTO market_data (timestamp, mid_price, micro_price, ofi, mlofi, cvd_acceleration)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', rows)
        conn.commit()
    conn.close()

ensure_db_seeded()

@st.cache_resource
def start_pipeline_background():
    if os.path.exists("pipeline.py"):
        return subprocess.Popen([sys.executable, "pipeline.py"])

pipeline_process = start_pipeline_background()

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BTC Quant Microstructure & ML Terminal",
    page_icon="⚡",
    layout="wide"
)

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

def log_trade_to_db(timestamp, side, entry_p, exit_p, gross_pnl, net_pnl, friction, status, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO paper_trades 
        (timestamp, side, entry_price, exit_price, gross_pnl, net_pnl, fee_slippage, status, ml_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, side, entry_p, exit_p, gross_pnl, net_pnl, friction, status, confidence))
    conn.commit()
    conn.close()

# --- 3. LIGHTGBM INFERENCE ENGINE ---
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

# --- 4. SESSION STATE & SIDEBAR CONFIG ---
if "positions" not in st.session_state:
    st.session_state.positions = []

st.sidebar.title("⚙️ Institutional Config")
enable_paper_trader = st.sidebar.toggle("Enable ML Paper Trader", value=True)
min_confidence = st.sidebar.slider("Min ML Probability Threshold", 0.50, 0.95, 0.65, 0.05)
z_threshold = st.sidebar.slider("Min OFI Z-Score Threshold (σ)", 1.0, 3.0, 1.5, 0.1)

st.sidebar.subheader("Execution Friction Model")
taker_fee_rate = st.sidebar.number_input("Taker Fee Rate (%)", value=0.05, step=0.01) / 100.0
slippage_atr_pct = st.sidebar.number_input("Slippage (% of ATR)", value=10.0, step=1.0) / 100.0

atr_sl_mult = st.sidebar.number_input("ATR Stop-Loss Multiplier", value=1.5, step=0.1)
atr_tp_mult = st.sidebar.number_input("ATR Take-Profit Multiplier", value=3.0, step=0.1)

# --- 5. DATA PROCESSING & ROLLING Z-SCORE NORMALIZATION ---
def load_and_process_data():
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM market_data ORDER BY id DESC LIMIT 300", conn)
        conn.close()
        if df.empty:
            return pd.DataFrame()
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Trend Indicators
        df['ema_100'] = df['mid_price'].ewm(span=100, adjust=False).mean()
        df['cvd_slope'] = df['cvd_acceleration'].rolling(window=10, min_periods=1).mean()
        
        # Volatility (ATR)
        df['high_low'] = df['mid_price'] - df['mid_price'].shift(1)
        df['atr'] = df['high_low'].abs().rolling(window=14, min_periods=1).mean().fillna(10.0)
        
        # ROLLING Z-SCORE NORMALIZATION (30-period window)
        for col in ['ofi', 'mlofi', 'cvd_acceleration', 'micro_price']:
            if col in df.columns:
                mean_val = df[col].rolling(window=30, min_periods=1).mean()
                std_val = df[col].rolling(window=30, min_periods=1).std().fillna(1.0).replace(0, 1e-5)
                df[f'{col}_z'] = (df[col] - mean_val) / std_val
                df[f'{col}_z'] = df[f'{col}_z'].fillna(0.0)

        return df
    except Exception:
        return pd.DataFrame()

# --- 6. MAIN INTERFACE ---
st.title("⚡ Pro BTC Predictive Microstructure & ML Terminal")

df = load_and_process_data()

if df.empty:
    st.warning("⏳ Connecting to exchange WebSocket stream & populating SQLite entries...")
else:
    latest = df.iloc[-1]
    current_price = float(latest.get('mid_price', 0.0))
    current_micro = float(latest.get('micro_price', current_price))
    current_ofi = float(latest.get('ofi', 0.0))
    current_ofi_z = float(latest.get('ofi_z', 0.0))
    current_mlofi_z = float(latest.get('mlofi_z', 0.0))
    current_cvd_z = float(latest.get('cvd_acceleration_z', 0.0))
    current_ema = float(latest.get('ema_100', current_price))
    current_cvd_slope = float(latest.get('cvd_slope', 0.0))
    current_atr = max(float(latest.get('atr', 10.0)), 5.0)

    # Standardized Feature Vector Construction
    micro_spread = current_micro - current_price
    ema_diff = current_price - current_ema
    feature_vector = [micro_spread, current_mlofi_z, current_cvd_z, ema_diff, current_atr, current_cvd_slope]
    
    ml_prob_long = predict_signal_probability(feature_vector)
    ml_prob_short = 1.0 - ml_prob_long

    # 1. TOP METRICS DASHBOARD
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mid Price", f"${current_price:,.2f}")
    col2.metric("MLOFI Z-Score", f"{current_mlofi_z:+.2f} σ", delta=f"L1 OFI Z: {current_ofi_z:+.2f}σ")
    col3.metric("Macro EMA (100)", f"${current_ema:,.2f}")
    col4.metric("Live ATR Volatility", f"${current_atr:,.2f}")
    col5.metric("ML Long Prob.", f"{ml_prob_long:.2%}")

    # 2. MICROSTRUCTURE & TREND CHART
    st.subheader("📈 Microstructure & Trend Regime Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mid_price'], name="Mid Price", line=dict(color="#00FFA3", width=2.5)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['micro_price'], name="Micro-Price", line=dict(color="#FF007A", width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_100'], name="100 EMA Trend", line=dict(color="#00E5FF", width=1.5, dash='dot')))
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 3. ML PAPER TRADING EXECUTION ENGINE (WITH FRICTION)
    st.markdown("---")
    st.subheader("🤖 Institutional ML Execution Engine (Net PnL Mode)")

    if enable_paper_trader:
        slippage_penalty = current_atr * slippage_atr_pct

        if st.session_state.positions:
            for pos in list(st.session_state.positions):
                entry_price = pos["entry_price"]
                side = pos["side"]
                tp_val = pos["tp"]
                sl_val = pos["sl"]
                conf = pos["confidence"]
                
                gross_pnl = (current_price - entry_price) if side == "LONG" else (entry_price - current_price)
                
                total_trade_notional = entry_price + current_price
                total_fees = total_trade_notional * taker_fee_rate
                total_slippage = slippage_penalty * 2.0
                total_friction = total_fees + total_slippage
                
                net_pnl = gross_pnl - total_friction
                
                if gross_pnl >= tp_val or gross_pnl <= -sl_val:
                    exit_reason = "DYNAMIC TP 🎯" if gross_pnl >= tp_val else "DYNAMIC SL 🛑"
                    time_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    log_trade_to_db(time_str, side, entry_price, current_price, gross_pnl, net_pnl, total_friction, exit_reason, conf)
                    st.session_state.positions.remove(pos)

        if not st.session_state.positions:
            is_bullish_regime = (current_price > current_ema) and (current_cvd_slope >= 0)
            is_bearish_regime = (current_price < current_ema) and (current_cvd_slope <= 0)
            
            calculated_sl = current_atr * atr_sl_mult
            calculated_tp = current_atr * atr_tp_mult
            
            if is_bullish_regime and ml_prob_long >= min_confidence and current_mlofi_z > z_threshold:
                executed_entry = current_price + slippage_penalty
                st.session_state.positions.append({
                    "entry_price": executed_entry,
                    "side": "LONG",
                    "sl": calculated_sl,
                    "tp": calculated_tp,
                    "confidence": ml_prob_long,
                    "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
                })
                st.success(f"🚀 ML Executed LONG @ ${executed_entry:,.2f} (Incl. ${slippage_penalty:.2f} Slippage) | Prob: {ml_prob_long:.2%} | Z-MLOFI: {current_mlofi_z:+.2f}σ")
            
            elif is_bearish_regime and ml_prob_short >= min_confidence and current_mlofi_z < -z_threshold:
                executed_entry = current_price - slippage_penalty
                st.session_state.positions.append({
                    "entry_price": executed_entry,
                    "side": "SHORT",
                    "sl": calculated_sl,
                    "tp": calculated_tp,
                    "confidence": ml_prob_short,
                    "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
                })
                st.success(f"🔻 ML Executed SHORT @ ${executed_entry:,.2f} (Incl. ${slippage_penalty:.2f} Slippage) | Prob: {ml_prob_short:.2%} | Z-MLOFI: {current_mlofi_z:+.2f}σ")
            
            else:
                st.info("🔍 ML Scanner Active: Monitoring Z-Score Imbalances...")

        if st.session_state.positions:
            active = st.session_state.positions[0]
            entry_p = active["entry_price"]
            s = active["side"]
            live_gross = (current_price - entry_p) if s == "LONG" else (entry_p - current_price)
            live_friction = (entry_p + current_price) * taker_fee_rate + (slippage_penalty * 2)
            live_net = live_gross - live_friction
            
            p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
            p_col1.metric("Active Order", f"{s}")
            p_col2.metric("Entry (Net Slippage)", f"${entry_p:,.2f}")
            p_col3.metric("Gross PnL", f"${live_gross:,.2f}")
            p_col4.metric("Net PnL (After Friction)", f"${live_net:,.2f}", delta=f"-${live_friction:.2f} Fees/Slippage")
            p_col5.metric("Target SL / TP", f"-${active['sl']:.1f} / +${active['tp']:.1f}")

    # 4. PERSISTENT CLOSED TRADE HISTORY TABLE
    trades_df = load_trade_history()
    if not trades_df.empty:
        st.subheader("📜 Over-Night Executed Trade Logs (Net Performance)")
        st.dataframe(trades_df, use_container_width=True)

# --- 7. NATIVE AUTO-RERUN (EVERY 3 SECONDS) ---
time.sleep(3)
st.rerun()
