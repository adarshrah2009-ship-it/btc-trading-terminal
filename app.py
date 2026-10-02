import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import sqlite3
import datetime
import lightgbm as lgb


# =====================================================================
# 1. DATABASE & INITIALIZATION
# =====================================================================
st.set_page_config(page_title="Quantitative Trading Terminal", layout="wide")

def init_db():
    conn = sqlite3.connect("trading_terminal.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS trade_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            gross_pnl REAL,
            net_pnl REAL,
            fee_slippage REAL,
            status TEXT,
            ml_confidence REAL,
            strategy TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

if "last_trade_time" not in st.session_state:
    st.session_state.last_trade_time = 0

# =====================================================================
# 2. MARKET DATA & INDICATOR COMPUTATION
# =====================================================================
@st.cache_data(ttl=5)
def fetch_binance_data(symbol="BTCUSDT", limit=100):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={limit}"
        res = requests.get(url, timeout=5).json()
        df = pd.DataFrame(res, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['open'] = df['open'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['taker_buy_base'] = df['taker_buy_base'].astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching exchange data: {e}")
        return pd.DataFrame()

def compute_indicators(df):
    if df.empty or len(df) < 30:
        return df

    # True Range & ATR (Volatility)
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift(1))
    df['tr2'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    # ADX (Trend Strength Filter)
    df['up'] = df['high'] - df['high'].shift(1)
    df['down'] = df['low'].shift(1) - df['low']
    df['pos_dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
    df['neg_dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
    
    smooth_pos = df['pos_dm'].rolling(14).mean()
    smooth_neg = df['neg_dm'].rolling(14).mean()
    
    df['pos_di'] = 100 * (smooth_pos / df['atr'])
    df['neg_di'] = 100 * (smooth_neg / df['atr'])
    df['dx'] = 100 * abs(df['pos_di'] - df['neg_di']) / (df['pos_di'] + df['neg_di'] + 1e-8)
    df['adx'] = df['dx'].rolling(14).mean()

    # Cumulative Volume Delta (CVD) Simulation
    df['cvd'] = (df['taker_buy_base'] - (df['volume'] - df['taker_buy_base'])).cumsum()
    
    # Order Flow Imbalance (OFI) proxy
    df['ofi'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) * df['volume']
    
    return df

# =====================================================================
# 3. MACHINE LEARNING SIGNAL ENGINE & GUARDRAILS
# =====================================================================
def generate_quant_signal(df):
    if df.empty or len(df) < 30:
        return "NEUTRAL", 0.50, "Insufficient Data"

    latest = df.iloc[-1]
    
    # -------------------------------------------------------------
    # GUARDRAIL 1: Hard ADX Trend Filter Gate
    # -------------------------------------------------------------
    if pd.isna(latest['adx']) or latest['adx'] < 20.0:
        return "NEUTRAL", 0.50, f"Blocked: Market Choppy (ADX {latest['adx']:.1f} < 20)"

    # -------------------------------------------------------------
    # GUARDRAIL 2: Trade Cooldown (15 Minutes)
    # -------------------------------------------------------------
    current_time = time.time()
    if current_time - st.session_state.last_trade_time < 900:  # 900 seconds = 15 mins
        remaining = int((900 - (current_time - st.session_state.last_trade_time)) / 60)
        return "NEUTRAL", 0.50, f"Blocked: Cooldown Active ({remaining}m remaining)"

    # Feature Setup for LightGBM Engine
    features = ['ofi', 'cvd', 'atr', 'adx']
    X = df[features].dropna()
    
    if len(X) < 20:
        return "NEUTRAL", 0.50, "Warming up indicators"

    # Simplified Model Target: Directional move over 5 periods
    y = np.where(df['close'].shift(-5) > df['close'], 1, 0)[:len(X)]
    
    model = lgb.LGBMClassifier(n_estimators=30, max_depth=3, learning_rate=0.05, verbose=-1)
    model.fit(X.iloc[:-1], y[:-1])
    
    # Predict latest probability
    prob_long = float(model.predict_proba(X.iloc[[-1]])[0][1])

    # -------------------------------------------------------------
    # GUARDRAIL 3: High-Confidence Thresholds & Sweep Shields
    # -------------------------------------------------------------
    # LONG Entry Signal
    if prob_long >= 0.68:
        # Check CVD confirmation (no divergence)
        if latest['cvd'] > df['cvd'].iloc[-5]:
            st.session_state.last_trade_time = current_time
            return "LONG", prob_long, "High Conviction Long + Volume Support"
        else:
            return "NEUTRAL", prob_long, "Blocked: CVD Liquidity Divergence"

    # SHORT Entry Signal
    elif prob_long <= 0.32:
        if latest['cvd'] < df['cvd'].iloc[-5]:
            st.session_state.last_trade_time = current_time
            return "SHORT", prob_long, "High Conviction Short + Selling Delta"
        else:
            return "NEUTRAL", prob_long, "Blocked: CVD Absorption Shield"

    return "NEUTRAL", prob_long, "Signal in Low Confidence Noise Band (0.33-0.67)"

# =====================================================================
# 4. ORDER EXECUTION & DB LOGGING
# =====================================================================
def execute_trade(signal, price, atr, confidence):
    position_size = 0.1  # 0.1 BTC
    fee_per_trade = 8.41  # Taker Roundtrip Fee + Spread
    
    # Dynamic ATR Take Profit Target (Minimum $200 BTC move)
    tp_distance = max(2.5 * atr, 200.0)
    sl_distance = 1.0 * atr

    if signal == "LONG":
        entry_price = price
        exit_price = entry_price + tp_distance
        gross_pnl = (exit_price - entry_price) * position_size
    elif signal == "SHORT":
        entry_price = price
        exit_price = entry_price - tp_distance
        gross_pnl = (entry_price - exit_price) * position_size
    else:
        return

    net_pnl = gross_pnl - fee_per_trade
    
    conn = sqlite3.connect("trading_terminal.db")
    c = conn.cursor()
    c.execute('''
        INSERT INTO trade_log (timestamp, side, entry_price, exit_price, gross_pnl, net_pnl, fee_slippage, status, ml_confidence, strategy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        signal,
        round(entry_price, 2),
        round(exit_price, 2),
        round(gross_pnl, 2),
        round(net_pnl, 2),
        fee_per_trade,
        "CLOSED",
        round(confidence, 4),
        "ML_Microstructure_v2"
    ))
    conn.commit()
    conn.close()

# =====================================================================
# 5. DASHBOARD UI
# =====================================================================
st.title("⚡ Autonomous Quantitative Trading Terminal")

df_data = fetch_binance_data()
df_data = compute_indicators(df_data)

if not df_data.empty:
    latest_price = df_data['close'].iloc[-1]
    latest_atr = df_data['atr'].iloc[-1] if not pd.isna(df_data['atr'].iloc[-1]) else 100.0
    latest_adx = df_data['adx'].iloc[-1] if not pd.isna(df_data['adx'].iloc[-1]) else 0.0

    signal, confidence, reason = generate_quant_signal(df_data)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BTC Price", f"${latest_price:,.2f}")
    col2.metric("Market Volatility (ATR)", f"${latest_atr:.2f}")
    col3.metric("Trend Strength (ADX)", f"{latest_adx:.1f}", delta="Strong" if latest_adx >= 20 else "Flat")
    col4.metric("ML Signal Conviction", f"{confidence:.1%}", delta=signal)

    st.subheader("System Execution Status")
    if signal in ["LONG", "SHORT"]:
        st.success(f"**Action Triggered:** Executing {signal} position | Reason: {reason}")
        execute_trade(signal, latest_price, latest_atr, confidence)
    else:
        st.info(f"**System Idle:** {reason}")

    # Display Trade History Table
    st.markdown("---")
    st.subheader("Trade Execution Logs")
    conn = sqlite3.connect("trading_terminal.db")
    trade_df = pd.read_sql_query("SELECT * FROM trade_log ORDER BY id DESC LIMIT 20", conn)
    conn.close()

    if not trade_df.empty:
        st.dataframe(trade_df, use_container_width=True)
    else:
        st.write("No trade executions logged yet.")
