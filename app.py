import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import time
import os
import importlib
import plotly.express as px

def get_ai_prediction(features):
    try:
        import ai_engine
        importlib.reload(ai_engine)
        prob = ai_engine.predict_trade_probability(features)
        return float(prob)
    except Exception as e:
        st.sidebar.warning(f"⚠️ AI Engine Alert: {str(e)}")
        return 0.60 

def run_ai_training():
    try:
        import ai_engine
        importlib.reload(ai_engine)
        return ai_engine.train_local_ai()
    except Exception as e:
        return f"Error loading ai_engine.py: {str(e)}"

st.set_page_config(page_title="Pro BTC Microstructure Terminal", layout="wide")

DB_NAME = "btc_market_structure.db"

# Session State
if "capital" not in st.session_state:
    st.session_state.capital = 10000.0
if "position" not in st.session_state:
    st.session_state.position = None
if "entry_price" not in st.session_state:
    st.session_state.entry_price = 0.0
if "scalp_count" not in st.session_state:
    st.session_state.scalp_count = 0
if "wins" not in st.session_state:
    st.session_state.wins = 0
if "realized_pnl" not in st.session_state:
    st.session_state.realized_pnl = 0.0
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []
if "chart_key_counter" not in st.session_state:
    st.session_state.chart_key_counter = 0

# Sidebar Settings
st.sidebar.title("⚙️ Terminal Config")
enable_trader = st.sidebar.toggle("Enable AI Paper Trader", value=True)
min_ai_conf = st.sidebar.slider("Min AI Confidence Threshold", 0.50, 0.90, 0.65, 0.05)
position_size = st.sidebar.number_input("Position Size ($)", value=2000.0, step=100.0)
tp_dist = st.sidebar.number_input("Take Profit ($)", value=50.0, step=10.0)
sl_dist = st.sidebar.number_input("Stop Loss ($)", value=25.0, step=5.0)
fee_pct = st.sidebar.number_input("Binance Fee %", value=0.04, step=0.01) / 100.0

st.sidebar.markdown("---")
if st.sidebar.button("⚡ Retrain AI Model"):
    training_msg = run_ai_training()
    st.sidebar.info(training_msg)

st.title("⚡ Pro BTC Predictive Microstructure Terminal")

placeholder = st.empty()

def load_latest_data():
    if not os.path.exists(DB_NAME):
        return None, None
    try:
        conn = sqlite3.connect(f"file:{DB_NAME}?mode=ro", uri=True)
        trades = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 500", conn)
        ob = pd.read_sql_query("SELECT * FROM orderbook_snapshots ORDER BY timestamp DESC LIMIT 20", conn)
        conn.close()
        
        if trades.empty:
            return None, None
        return trades, ob
    except Exception:
        return None, None

while True:
    st.session_state.chart_key_counter += 1
    trades, ob = load_latest_data()
    
    if trades is None or trades.empty:
        with placeholder.container():
            st.warning("⏳ Waiting for database entries from pipeline.py...")
        time.sleep(1)
        continue

    latest_price = trades.iloc[0]['price']
    trades_chrono = trades.sort_values('timestamp').copy()
    
    # 1. Feature Calculations
    trades_chrono['signed_vol'] = trades_chrono.apply(lambda r: r['quantity'] if r['is_buyer_maker'] == 0 else -r['quantity'], axis=1)
    trades_chrono['buy_vol'] = trades_chrono.apply(lambda r: r['quantity'] if r['is_buyer_maker'] == 0 else 0.0, axis=1)
    trades_chrono['sell_vol'] = trades_chrono.apply(lambda r: r['quantity'] if r['is_buyer_maker'] == 1 else 0.0, axis=1)
    
    buy_vol_total = trades_chrono['buy_vol'].sum()
    sell_vol_total = trades_chrono['sell_vol'].sum()
    delta = buy_vol_total - sell_vol_total
    cvd = trades_chrono['signed_vol'].sum()
    vwap = (trades_chrono['price'] * trades_chrono['quantity']).sum() / trades_chrono['quantity'].sum()
    
    trades_chrono['cvd_series'] = trades_chrono['signed_vol'].cumsum()
    trades_chrono['aggression_ratio'] = (trades_chrono['buy_vol'].rolling(20).sum() + 1e-9) / (trades_chrono['sell_vol'].rolling(20).sum() + 1e-9)
    trades_chrono['cvd_accel'] = trades_chrono['cvd_series'] - trades_chrono['cvd_series'].shift(20)

    latest_row = trades_chrono.iloc[-1]
    
    # Orderbook Features
    obi = ob.iloc[0]['obi'] if (ob is not None and not ob.empty and 'obi' in ob.columns) else 0.0
    bids_vol = ob.iloc[0]['bids_vol'] if (ob is not None and not ob.empty and 'bids_vol' in ob.columns) else 0.0
    asks_vol = ob.iloc[0]['asks_vol'] if (ob is not None and not ob.empty and 'asks_vol' in ob.columns) else 0.0
    depth_ratio = (bids_vol + 1e-9) / (asks_vol + 1e-9)
    absorption = (latest_row['buy_vol'] + 1e-9) / (asks_vol + 1e-9)
    
    ofi = 0.0
    if ob is not None and len(ob) > 1 and 'bids_vol' in ob.columns:
        ofi = (ob.iloc[0]['bids_vol'] - ob.iloc[1]['bids_vol']) - (ob.iloc[0]['asks_vol'] - ob.iloc[1]['asks_vol'])

    # Micro-Price Calculation
    micro_price = (asks_vol * (latest_price - 0.5) + bids_vol * (latest_price + 0.5)) / (bids_vol + asks_vol + 1e-9) if (bids_vol + asks_vol) > 0 else latest_price

    # Confluence Score
    score = 0
    if latest_price > vwap: score += 1
    else: score -= 1
    
    if delta > 0.5: score += 1
    elif delta < -0.5: score -= 1
    
    if obi > 0.1: score += 1
    elif obi < -0.1: score -= 1

    current_features = {
        'signed_vol': float(latest_row['signed_vol']),
        'cvd': float(cvd),
        'vwap_dist': float(latest_price - vwap),
        'aggression_ratio': float(latest_row['aggression_ratio']) if not np.isnan(latest_row['aggression_ratio']) else 1.0,
        'cvd_accel': float(latest_row['cvd_accel']) if not np.isnan(latest_row['cvd_accel']) else 0.0,
        'obi': float(obi),
        'depth_ratio': float(depth_ratio),
        'absorption_ratio': float(absorption),
        'ofi': float(ofi)
    }

    ai_prob = get_ai_prediction(current_features)

    # Paper Execution Logic
    if enable_trader:
        if st.session_state.position is None:
            if score >= 2 and ai_prob >= min_ai_conf:
                st.session_state.position = "LONG"
                st.session_state.entry_price = latest_price
            elif score <= -2 and ai_prob >= min_ai_conf:
                st.session_state.position = "SHORT"
                st.session_state.entry_price = latest_price
        
        elif st.session_state.position == "LONG":
            pnl = latest_price - st.session_state.entry_price
            if pnl >= tp_dist or pnl <= -sl_dist:
                net_pnl = (pnl / st.session_state.entry_price * position_size) - (position_size * fee_pct * 2)
                st.session_state.capital += net_pnl
                st.session_state.realized_pnl += net_pnl
                st.session_state.scalp_count += 1
                if net_pnl > 0: st.session_state.wins += 1
                st.session_state.trade_history.append({
                    "Side": "LONG", "Entry": f"${st.session_state.entry_price:,.2f}",
                    "Exit": f"${latest_price:,.2f}", "PnL": f"${net_pnl:+.2f}"
                })
                st.session_state.position = None

        elif st.session_state.position == "SHORT":
            pnl = st.session_state.entry_price - latest_price
            if pnl >= tp_dist or pnl <= -sl_dist:
                net_pnl = (pnl / st.session_state.entry_price * position_size) - (position_size * fee_pct * 2)
                st.session_state.capital += net_pnl
                st.session_state.realized_pnl += net_pnl
                st.session_state.scalp_count += 1
                if net_pnl > 0: st.session_state.wins += 1
                st.session_state.trade_history.append({
                    "Side": "SHORT", "Entry": f"${st.session_state.entry_price:,.2f}",
                    "Exit": f"${latest_price:,.2f}", "PnL": f"${net_pnl:+.2f}"
                })
                st.session_state.position = None

    win_rate = (st.session_state.wins / st.session_state.scalp_count * 100) if st.session_state.scalp_count > 0 else 0.0

    # Live Floating PnL
    unrealized_pnl = 0.0
    if st.session_state.position == "LONG":
        price_diff = latest_price - st.session_state.entry_price
        unrealized_pnl = (price_diff / st.session_state.entry_price) * position_size
    elif st.session_state.position == "SHORT":
        price_diff = st.session_state.entry_price - latest_price
        unrealized_pnl = (price_diff / st.session_state.entry_price) * position_size

    # UI Rendering
    with placeholder.container():
        if score >= 2:
            st.success(f"🟢 **BULLISH LIQUIDITY SWEEP DETECTED** | Signal Score: +{score} | AI Breakout Prob: {ai_prob*100:.1f}%")
        elif score <= -2:
            st.error(f"🔴 **BEARISH LIQUIDITY SWEEP DETECTED** | Signal Score: {score} | AI Breakout Prob: {ai_prob*100:.1f}%")
        else:
            st.warning(f"🟡 **MARKET RANGING / NEUTRAL** | Signal Score: {score} | AI Breakout Prob: {ai_prob*100:.1f}%")

        # Top Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Paper Capital", f"${st.session_state.capital:,.2f}", f"{st.session_state.realized_pnl:+.2f} USDT")
        col2.metric("Total Scalps", f"{st.session_state.scalp_count}")
        col3.metric("Win Rate", f"{win_rate:.1f}%")

        if st.session_state.position:
            col4.metric(
                label=f"Active {st.session_state.position}", 
                value=f"${st.session_state.entry_price:,.2f}", 
                delta=f"Unrealized PnL: ${unrealized_pnl:+.2f}"
            )
        else:
            col4.metric("Active Position", "FLAT")

        # Microstructure Bar
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Live Price", f"${latest_price:,.2f}")
        m2.metric("Micro-Price", f"${micro_price:,.2f}")
        m3.metric("OFI Imbalance", f"{ofi:+.2f}")
        m4.metric("CVD Accel", f"{latest_row['cvd_accel']:+.2f}" if not np.isnan(latest_row['cvd_accel']) else "+0.00")
        m5.metric("Absorption Ratio", f"{absorption:.2f}")

        st.markdown("---")

        chart_col, log_col = st.columns([2, 1])

        with chart_col:
            st.subheader("📈 Real-Time Price vs VWAP")
            chart_df = trades_chrono[['timestamp', 'price']].copy()
            chart_df['vwap'] = vwap
            
            fig = px.line(chart_df, x='timestamp', y=['price', 'vwap'], 
                          color_discrete_map={'price': '#00E676', 'vwap': '#FFEA00'})
            
            min_p = min(chart_df['price'].min(), vwap) - 5
            max_p = max(chart_df['price'].max(), vwap) + 5
            fig.update_layout(
                yaxis_range=[min_p, max_p], 
                margin=dict(l=10, r=10, t=10, b=10), 
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"btc_chart_{st.session_state.chart_key_counter}")

        with log_col:
            st.subheader("📋 Executed Trades")
            if st.session_state.trade_history:
                st.dataframe(pd.DataFrame(st.session_state.trade_history[::-1]), height=320, use_container_width=True)
            else:
                st.caption("Waiting for predictive sweep signals...")

    time.sleep(1)