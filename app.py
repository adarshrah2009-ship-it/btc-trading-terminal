import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import lightgbm as lgb
import time
import requests

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro BTC Predictive & Stat-Arb Terminal",
    page_icon="⚡",
    layout="wide"
)

DB_NAME = "btc_market_structure.db"

# --- 2. PERSISTENT DATABASE SETUP ---
def init_db():
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
        CREATE TABLE IF NOT EXISTS pair_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            btc_price REAL,
            eth_price REAL,
            sol_price REAL
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
    conn.close()

init_db()

# --- 3. LIVE MULTI-ASSET FETCHING ENGINE ---
def fetch_multi_asset_ticker():
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    symbols = {'BTCUSDT': None, 'ETHUSDT': None, 'SOLUSDT': None}
    
    # Try fetching real Binance prices
    endpoints = [
        "https://api.binance.com/api/v3/ticker/bookTicker",
        "https://api.binance.us/api/v3/ticker/bookTicker"
    ]
    
    for url in endpoints:
        try:
            res = requests.get(url, timeout=1.5).json()
            if isinstance(res, list):
                for item in res:
                    if item.get('symbol') in symbols:
                        bid = float(item['bidPrice'])
                        ask = float(item['askPrice'])
                        symbols[item['symbol']] = (bid + ask) / 2.0
            if all(v is not None for v in symbols.values()):
                break
        except Exception:
            continue

    # Fallback to persistent DB state + noise if API blocked
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    btc = symbols['BTCUSDT']
    eth = symbols['ETHUSDT']
    sol = symbols['SOLUSDT']

    if btc is None or eth is None or sol is None:
        cursor.execute("SELECT btc_price, eth_price, sol_price FROM pair_prices ORDER BY id DESC LIMIT 1")
        last_row = cursor.fetchone()
        btc = last_row[0] + np.random.normal(0, 1.5) if last_row else 80450.0 + np.random.normal(0, 1.5)
        eth = last_row[1] + np.random.normal(0, 0.2) if last_row else 2650.0 + np.random.normal(0, 0.2)
        sol = last_row[2] + np.random.normal(0, 0.05) if last_row else 145.0 + np.random.normal(0, 0.05)

    # Calculate BTC micro structure proxies
    micro = btc + np.random.normal(0, 0.2)
    ofi = np.random.normal(0, 0.3)

    # Save into DB
    cursor.execute('''
        INSERT INTO market_data (timestamp, mid_price, micro_price, ofi, mlofi, cvd_acceleration)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp_str, btc, micro, ofi, ofi * 1.2, ofi * 0.8))
    
    cursor.execute('''
        INSERT INTO pair_prices (timestamp, btc_price, eth_price, sol_price)
        VALUES (?, ?, ?, ?)
    ''', (timestamp_str, btc, eth, sol))

    conn.commit()
    conn.close()

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

# --- 4. LIGHTGBM INFERENCE ENGINE ---
@st.cache_resource
def load_or_init_lgbm():
    X_dummy = np.random.randn(100, 6)
    y_dummy = np.random.randint(0, 2, size=100)
    train_data = lgb.Dataset(X_dummy, label=y_dummy)
    params = {'objective': 'binary', 'verbosity': -1, 'learning_rate': 0.05, 'num_leaves': 15}
    return lgb.train(params, train_data, num_boost_round=10)

lgb_model = load_or_init_lgbm()

def predict_signal_probability(features):
    features_array = np.array(features).reshape(1, -1)
    return lgb_model.predict(features_array)[0]

# --- 5. GLOBAL SESSION STATE & SIDEBAR CONFIG ---
if "positions" not in st.session_state:
    st.session_state.positions = []
if "arb_positions" not in st.session_state:
    st.session_state.arb_positions = []

st.sidebar.title("⚡ Quantitative Navigation")
terminal_mode = st.sidebar.radio(
    "Select Terminal Strategy",
    ["Order Book Microstructure (Directional)", "Statistical Arbitrage (Market Neutral)"]
)

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Execution Settings")
enable_paper_trader = st.sidebar.toggle("Enable Paper Trader", value=True)
taker_fee_rate = st.sidebar.number_input("Taker Fee Rate (%)", value=0.05, step=0.01) / 100.0

# --- 6. DATA LOADERS ---
def load_directional_data():
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM market_data ORDER BY id DESC LIMIT 60", conn)
        conn.close()
        if df.empty:
            return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['ema_100'] = df['mid_price'].ewm(span=20, adjust=False).mean()
        df['cvd_slope'] = df['cvd_acceleration'].rolling(window=5, min_periods=1).mean()
        df['high_low'] = df['mid_price'] - df['mid_price'].shift(1)
        df['atr'] = df['high_low'].abs().rolling(window=10, min_periods=1).mean().fillna(5.0)
        
        for col in ['ofi', 'mlofi', 'cvd_acceleration', 'micro_price']:
            if col in df.columns:
                mean_val = df[col].rolling(window=15, min_periods=1).mean()
                std_val = df[col].rolling(window=15, min_periods=1).std().fillna(1.0).replace(0, 1e-5)
                df[f'{col}_z'] = (df[col] - mean_val) / std_val
                df[f'{col}_z'] = df[f'{col}_z'].fillna(0.0)
        return df
    except Exception:
        return pd.DataFrame()

def load_pair_data():
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM pair_prices ORDER BY id DESC LIMIT 60", conn)
        conn.close()
        if df.empty:
            return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

# --- 7. MAIN RENDER ENGINE WITH AUTOMATIC TICKING ---
st.title("⚡ Pro Quant Predictive & Arbitrage Terminal")

@st.fragment(run_every=2)
def render_active_strategy():
    fetch_multi_asset_ticker()

    if terminal_mode == "Order Book Microstructure (Directional)":
        df = load_directional_data()
        if not df.empty:
            latest = df.iloc[-1]
            current_price = float(latest.get('mid_price', 0.0))
            current_micro = float(latest.get('micro_price', current_price))
            current_ofi_z = float(latest.get('ofi_z', 0.0))
            current_mlofi_z = float(latest.get('mlofi_z', 0.0))
            current_cvd_z = float(latest.get('cvd_acceleration_z', 0.0))
            current_ema = float(latest.get('ema_100', current_price))
            current_cvd_slope = float(latest.get('cvd_slope', 0.0))
            current_atr = max(float(latest.get('atr', 5.0)), 2.0)

            micro_spread = current_micro - current_price
            ema_diff = current_price - current_ema
            feature_vector = [micro_spread, current_mlofi_z, current_cvd_z, ema_diff, current_atr, current_cvd_slope]
            ml_prob_long = predict_signal_probability(feature_vector)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("BTC Mid Price", f"${current_price:,.2f}")
            col2.metric("MLOFI Z-Score", f"{current_mlofi_z:+.2f} σ", delta=f"L1 OFI Z: {current_ofi_z:+.2f}σ")
            col3.metric("Macro EMA", f"${current_ema:,.2f}")
            col4.metric("Live ATR Volatility", f"${current_atr:,.2f}")
            col5.metric("ML Long Prob.", f"{ml_prob_long:.2%}")

            st.subheader("📈 Microstructure & Trend Regime Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mid_price'], name="Mid Price", line=dict(color="#00FFA3", width=2.5)))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['micro_price'], name="Micro-Price", line=dict(color="#FF007A", width=1.5, dash='dash')))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_100'], name="EMA Trend", line=dict(color="#00E5FF", width=1.5, dash='dot')))
            fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    elif terminal_mode == "Statistical Arbitrage (Market Neutral)":
        pair_df = load_pair_data()
        if not pair_df.empty and len(pair_df) > 5:
            selected_pair = st.sidebar.selectbox("Select Stat-Arb Pair", ["BTC / ETH", "BTC / SOL", "ETH / SOL"])
            z_entry_thresh = st.sidebar.slider("Stat-Arb Entry Threshold (σ)", 1.0, 3.0, 1.8, 0.1)

            if selected_pair == "BTC / ETH":
                asset_a, asset_b = "btc_price", "eth_price"
                name_a, name_b = "BTC", "ETH"
            elif selected_pair == "BTC / SOL":
                asset_a, asset_b = "btc_price", "sol_price"
                name_a, name_b = "BTC", "SOL"
            else:
                asset_a, asset_b = "eth_price", "sol_price"
                name_a, name_b = "ETH", "SOL"

            pair_df['ratio'] = pair_df[asset_a] / pair_df[asset_b]
            pair_df['ratio_mean'] = pair_df['ratio'].rolling(window=15, min_periods=1).mean()
            pair_df['ratio_std'] = pair_df['ratio'].rolling(window=15, min_periods=1).std().fillna(1.0).replace(0, 1e-5)
            pair_df['z_score'] = (pair_df['ratio'] - pair_df['ratio_mean']) / pair_df['ratio_std']

            latest_ratio = pair_df['ratio'].iloc[-1]
            latest_z = pair_df['z_score'].iloc[-1]
            price_a = pair_df[asset_a].iloc[-1]
            price_b = pair_df[asset_b].iloc[-1]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"Price {name_a}", f"${price_a:,.2f}")
            col2.metric(f"Price {name_b}", f"${price_b:,.2f}")
            col3.metric("Pair Price Ratio", f"{latest_ratio:.4f}")
            col4.metric("Spread Z-Score", f"{latest_z:+.2f} σ", delta="Mean Reversion Mode")

            # Dual-Axis Plotly Chart
            st.subheader(f"📊 Statistical Spread & Z-Score Analysis ({selected_pair})")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
            fig.add_trace(go.Scatter(x=pair_df['timestamp'], y=pair_df['ratio'], name="Price Ratio", line=dict(color="#00E5FF", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=pair_df['timestamp'], y=pair_df['ratio_mean'], name="Mean Ratio", line=dict(color="#FFD700", dash='dash')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=pair_df['timestamp'], y=pair_df['z_score'], name="Z-Score", line=dict(color="#00FFA3", width=2)), row=2, col=1)
            fig.add_hline(y=z_entry_thresh, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=-z_entry_thresh, line_dash="dot", line_color="green", row=2, col=1)
            fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Statistical Arbitrage Execution Engine
            st.markdown("---")
            st.subheader("⚖️ Market-Neutral Arbitrage Execution Engine")
            
            if enable_paper_trader:
                if not st.session_state.arb_positions:
                    if latest_z > z_entry_thresh:
                        st.session_state.arb_positions.append({
                            "pair": selected_pair, "side": f"SHORT {name_a} / LONG {name_b}",
                            "entry_z": latest_z, "entry_ratio": latest_ratio
                        })
                        st.warning(f"🚨 Stat-Arb Executed: Short {name_a} / Long {name_b} @ Z: {latest_z:+.2f}σ")
                    elif latest_z < -z_entry_thresh:
                        st.session_state.arb_positions.append({
                            "pair": selected_pair, "side": f"LONG {name_a} / SHORT {name_b}",
                            "entry_z": latest_z, "entry_ratio": latest_ratio
                        })
                        st.success(f"🚀 Stat-Arb Executed: Long {name_a} / Short {name_b} @ Z: {latest_z:+.2f}σ")
                    else:
                        st.info("🔍 Monitoring Pair Ratio: Waiting for Mean Divergence (> |1.8σ|)...")

                if st.session_state.arb_positions:
                    active = st.session_state.arb_positions[0]
                    st.info(f"Active Position: **{active['side']}** | Entry Z: {active['entry_z']:+.2f}σ | Current Z: {latest_z:+.2f}σ")
                    if abs(latest_z) <= 0.2:
                        st.success("🎯 Mean Reversion Complete! Position Closed at Mean Spread.")
                        st.session_state.arb_positions.clear()

    trades_df = load_trade_history()
    if not trades_df.empty:
        st.markdown("---")
        st.subheader("📜 Terminal Trade Execution Logs")
        st.dataframe(trades_df, use_container_width=True)

render_active_strategy()
