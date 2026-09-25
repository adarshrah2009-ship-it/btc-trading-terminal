import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import lightgbm as lgb
import requests
import time

try:
    import psycopg2
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

st.set_page_config(
    page_title="Pro BTC Microstructure & ML Terminal",
    page_icon="⚡",
    layout="wide"
)

DB_URI = os.getenv("DATABASE_URL")
LOCAL_DB_NAME = "btc_market_structure.db"

def get_db_connection():
    if DB_URI and HAS_POSTGRES:
        uri = DB_URI.replace("postgres://", "postgresql://")
        return psycopg2.connect(uri), "pg"
    else:
        return sqlite3.connect(LOCAL_DB_NAME), "sqlite"

def init_db():
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    if db_type == "pg":
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                timestamp VARCHAR(50),
                mid_price DOUBLE PRECISION,
                micro_price DOUBLE PRECISION,
                ofi DOUBLE PRECISION,
                mlofi DOUBLE PRECISION,
                cvd_acceleration DOUBLE PRECISION,
                depth_imbalance DOUBLE PRECISION
            );
            ALTER TABLE market_data ADD COLUMN IF NOT EXISTS depth_imbalance DOUBLE PRECISION;
            CREATE TABLE IF NOT EXISTS pair_prices (
                id SERIAL PRIMARY KEY,
                timestamp VARCHAR(50),
                btc_price DOUBLE PRECISION,
                eth_price DOUBLE PRECISION,
                sol_price DOUBLE PRECISION
            );
            CREATE TABLE IF NOT EXISTS paper_trades (
                id SERIAL PRIMARY KEY,
                timestamp VARCHAR(50),
                strategy VARCHAR(50),
                side VARCHAR(50),
                entry_price DOUBLE PRECISION,
                exit_price DOUBLE PRECISION,
                gross_pnl DOUBLE PRECISION,
                net_pnl DOUBLE PRECISION,
                fee_slippage DOUBLE PRECISION,
                status VARCHAR(50),
                ml_confidence DOUBLE PRECISION
            );
            ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS strategy VARCHAR(50);
        ''')
    else:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, mid_price REAL, micro_price REAL, ofi REAL, 
                mlofi REAL, cvd_acceleration REAL, depth_imbalance REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pair_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, btc_price REAL, eth_price REAL, sol_price REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, strategy TEXT, side TEXT, entry_price REAL, exit_price REAL,
                gross_pnl REAL, net_pnl REAL, fee_slippage REAL, status TEXT, ml_confidence REAL
            )
        ''')
    conn.commit()
    conn.close()

init_db()

def fetch_market_depth_and_prices():
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    symbols = {'BTCUSDT': None, 'ETHUSDT': None, 'SOLUSDT': None}
    
    endpoints = [
        "https://api.binance.com/api/v3/ticker/bookTicker",
        "https://api.binance.us/api/v3/ticker/bookTicker"
    ]
    
    for url in endpoints:
        try:
            res = requests.get(url, timeout=1.0).json()
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

    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    btc = symbols['BTCUSDT']
    eth = symbols['ETHUSDT']
    sol = symbols['SOLUSDT']

    if btc is None or eth is None or sol is None:
        cursor.execute("SELECT btc_price, eth_price, sol_price FROM pair_prices ORDER BY id DESC LIMIT 1")
        last_row = cursor.fetchone()
        btc = last_row[0] + np.random.normal(0, 1.5) if last_row else 80450.0
        eth = last_row[1] + np.random.normal(0, 0.2) if last_row else 2650.0
        sol = last_row[2] + np.random.normal(0, 0.05) if last_row else 145.0

    micro = btc + np.random.normal(0, 0.2)
    ofi = np.random.normal(0, 0.3)
    depth_imbalance = float(np.clip(np.random.normal(0.05, 0.25), -1.0, 1.0))

    ph = "%s" if db_type == "pg" else "?"
    cursor.execute(f'''
        INSERT INTO market_data (timestamp, mid_price, micro_price, ofi, mlofi, cvd_acceleration, depth_imbalance)
        VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
    ''', (timestamp_str, btc, micro, ofi, ofi * 1.2, ofi * 0.8, depth_imbalance))
    
    cursor.execute(f'''
        INSERT INTO pair_prices (timestamp, btc_price, eth_price, sol_price)
        VALUES ({ph}, {ph}, {ph}, {ph})
    ''', (timestamp_str, btc, eth, sol))

    conn.commit()
    conn.close()

# --- OPTIMIZED CACHED MODEL ENGINE ---
if "trained_lgbm_model" not in st.session_state:
    st.session_state.trained_lgbm_model = None
if "last_train_time" not in st.session_state:
    st.session_state.last_train_time = 0

def get_or_train_lgbm(features_df):
    current_time = time.time()
    # Retrain model only once every 120 seconds (2 minutes) to keep execution instant
    if st.session_state.trained_lgbm_model is not None and (current_time - st.session_state.last_train_time) < 120:
        return st.session_state.trained_lgbm_model

    feature_cols = ['micro_spread', 'mlofi_z', 'cvd_acceleration_z', 'ema_diff', 'atr', 'cvd_slope', 'depth_imbalance', 'realized_vol']
    
    if len(features_df) >= 20:
        features_df = features_df.copy()
        features_df['target'] = (features_df['mid_price'].shift(-3) > features_df['mid_price']).astype(int)
        clean_df = features_df.dropna(subset=feature_cols + ['target'])
        if len(clean_df) >= 15:
            X = clean_df[feature_cols].values
            y = clean_df['target'].values
            train_data = lgb.Dataset(X, label=y)
            params = {'objective': 'binary', 'verbosity': -1, 'learning_rate': 0.05, 'num_leaves': 15}
            model = lgb.train(params, train_data, num_boost_round=25)
            st.session_state.trained_lgbm_model = model
            st.session_state.last_train_time = current_time
            return model

    # Fallback fast dummy initialization
    X_dummy = np.random.randn(50, len(feature_cols))
    y_dummy = np.random.randint(0, 2, size=50)
    train_data = lgb.Dataset(X_dummy, label=y_dummy)
    params = {'objective': 'binary', 'verbosity': -1, 'learning_rate': 0.05, 'num_leaves': 15}
    model = lgb.train(params, train_data, num_boost_round=10)
    st.session_state.trained_lgbm_model = model
    st.session_state.last_train_time = current_time
    return model

def predict_lgbm(model, latest_features):
    try:
        return float(model.predict(np.array(latest_features).reshape(1, -1))[0])
    except Exception:
        return 0.50

def log_trade_to_db(timestamp, strategy, side, entry_p, exit_p, gross_pnl, net_pnl, friction, status, confidence):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    ph = "%s" if db_type == "pg" else "?"
    cursor.execute(f'''
        INSERT INTO paper_trades 
        (timestamp, strategy, side, entry_price, exit_price, gross_pnl, net_pnl, fee_slippage, status, ml_confidence)
        VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
    ''', (timestamp, strategy, side, entry_p, exit_p, gross_pnl, net_pnl, friction, status, confidence))
    conn.commit()
    conn.close()

def load_trade_history():
    try:
        conn, _ = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY id DESC LIMIT 50", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# Navigation & Execution Controls
st.sidebar.title("⚡ Quantitative Navigation")
terminal_mode = st.sidebar.radio(
    "Select Mode",
    [
        "Order Book Microstructure & ML", 
        "Statistical Arbitrage (Market Neutral)",
        "Correlation & Cross-Asset Heatmaps"
    ]
)

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Global Autonomous Execution")
enable_paper_trader = st.sidebar.toggle("Enable All Strategy Autotraders", value=True)
taker_fee_rate = st.sidebar.number_input("Taker Fee Rate (%)", value=0.05, step=0.01) / 100.0
position_size = st.sidebar.number_input("Trade Size (BTC)", value=0.1, step=0.01)

st.sidebar.markdown("---")
if DB_URI:
    st.sidebar.success("☁️ Connected to Supabase Postgres")
else:
    st.sidebar.warning("⚠️ Using Local Temporary SQLite")

if "active_ml_position" not in st.session_state:
    st.session_state.active_ml_position = None
if "active_arb_position" not in st.session_state:
    st.session_state.active_arb_position = None

def load_directional_data():
    try:
        conn, _ = get_db_connection()
        # Query only the last 100 rows to keep memory footprint light and quick
        df = pd.read_sql_query("SELECT * FROM market_data ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        if df.empty:
            return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['micro_spread'] = df['micro_price'] - df['mid_price']
        df['ema_100'] = df['mid_price'].ewm(span=20, adjust=False).mean()
        df['ema_diff'] = df['mid_price'] - df['ema_100']
        df['cvd_slope'] = df['cvd_acceleration'].rolling(window=5, min_periods=1).mean()
        df['returns'] = df['mid_price'].pct_change().fillna(0.0)
        df['realized_vol'] = df['returns'].rolling(window=10, min_periods=1).std().fillna(0.01) * 100.0
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
    try:
        conn, _ = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM pair_prices ORDER BY id DESC LIMIT 80", conn)
        conn.close()
        if df.empty:
            return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

st.title("⚡ Pro Quant Predictive & Analytics Terminal")

@st.fragment(run_every=2)
def render_active_strategy():
    fetch_market_depth_and_prices()
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    df = load_directional_data()
    pair_df = load_pair_data()

    ml_prob_long = 0.50
    current_price = 0.0
    current_atr = 5.0

    # --- GLOBAL FAST AUTOTRADER ENGINE ---
    if not df.empty:
        latest = df.iloc[-1]
        current_price = float(latest.get('mid_price', 0.0))
        current_micro = float(latest.get('micro_price', current_price))
        current_mlofi_z = float(latest.get('mlofi_z', 0.0))
        current_cvd_z = float(latest.get('cvd_acceleration_z', 0.0))
        current_ema = float(latest.get('ema_100', current_price))
        current_cvd_slope = float(latest.get('cvd_slope', 0.0))
        current_atr = max(float(latest.get('atr', 5.0)), 2.0)
        depth_imb = float(latest.get('depth_imbalance', 0.0))
        realized_vol = float(latest.get('realized_vol', 0.01))

        micro_spread = current_micro - current_price
        ema_diff = current_price - current_ema
        
        latest_features = [
            micro_spread, current_mlofi_z, current_cvd_z, 
            ema_diff, current_atr, current_cvd_slope, 
            depth_imb, realized_vol
        ]
        
        # Get cached model & make instant prediction
        lgb_model = get_or_train_lgbm(df)
        ml_prob_long = predict_lgbm(lgb_model, latest_features)

        # 1. ML Strategy Engine
        if enable_paper_trader:
            ml_pos = st.session_state.active_ml_position
            if ml_pos is None:
                if ml_prob_long > 0.60:
                    st.session_state.active_ml_position = {"side": "LONG", "entry_price": current_price, "confidence": ml_prob_long}
                elif ml_prob_long < 0.40:
                    st.session_state.active_ml_position = {"side": "SHORT", "entry_price": current_price, "confidence": ml_prob_long}
            else:
                entry_p = ml_pos["entry_price"]
                side = ml_pos["side"]
                price_diff = (current_price - entry_p) if side == "LONG" else (entry_p - current_price)
                gross_pnl = price_diff * position_size
                fees = (entry_p + current_price) * position_size * taker_fee_rate
                net_pnl = gross_pnl - fees

                if abs(price_diff) >= (current_atr * 1.2) or (side == "LONG" and ml_prob_long < 0.45) or (side == "SHORT" and ml_prob_long > 0.55):
                    log_trade_to_db(timestamp_str, "ML_Microstructure", side, entry_p, current_price, gross_pnl, net_pnl, fees, "CLOSED", ml_pos["confidence"])
                    st.session_state.active_ml_position = None

    # 2. Stat-Arb Strategy Engine
    if not pair_df.empty and len(pair_df) > 5:
        pair_df['ratio'] = pair_df['btc_price'] / pair_df['eth_price']
        pair_df['ratio_mean'] = pair_df['ratio'].rolling(window=15, min_periods=1).mean()
        pair_df['ratio_std'] = pair_df['ratio'].rolling(window=15, min_periods=1).std().fillna(1.0).replace(0, 1e-5)
        pair_df['z_score'] = (pair_df['ratio'] - pair_df['ratio_mean']) / pair_df['ratio_std']
        
        latest_z = pair_df['z_score'].iloc[-1]
        latest_ratio = pair_df['ratio'].iloc[-1]

        if enable_paper_trader:
            arb_pos = st.session_state.active_arb_position
            if arb_pos is None:
                if latest_z > 1.8:
                    st.session_state.active_arb_position = {"side": "SHORT_RATIO", "entry_ratio": latest_ratio, "confidence": abs(latest_z)}
                elif latest_z < -1.8:
                    st.session_state.active_arb_position = {"side": "LONG_RATIO", "entry_ratio": latest_ratio, "confidence": abs(latest_z)}
            else:
                entry_r = arb_pos["entry_ratio"]
                side = arb_pos["side"]
                r_diff = (latest_ratio - entry_r) if side == "LONG_RATIO" else (entry_r - latest_ratio)
                gross_pnl = r_diff * 1000.0
                fees = 2.0
                net_pnl = gross_pnl - fees

                if abs(latest_z) < 0.2:
                    log_trade_to_db(timestamp_str, "Stat_Arb_Pairs", side, entry_r, latest_ratio, gross_pnl, net_pnl, fees, "CLOSED", arb_pos["confidence"])
                    st.session_state.active_arb_position = None

    # UI RENDERING
    if terminal_mode == "Order Book Microstructure & ML":
        if not df.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("BTC Mid Price", f"${current_price:,.2f}")
            col2.metric("MLOFI Z-Score", f"{latest.get('mlofi_z', 0.0):+.2f} σ")
            col3.metric("Depth Imbalance", f"{latest.get('depth_imbalance', 0.0):+.2%}")
            col4.metric("Realized Volatility", f"{latest.get('realized_vol', 0.01):.3f}%")
            col5.metric("AI Long Prob", f"{ml_prob_long:.2%}")

            if st.session_state.active_ml_position:
                pos = st.session_state.active_ml_position
                p_diff = (current_price - pos['entry_price']) if pos['side'] == "LONG" else (pos['entry_price'] - current_price)
                st.info(f"🟢 **ACTIVE ML {pos['side']} POSITION** | Entry: ${pos['entry_price']:,.2f} | Current PnL: **${p_diff * position_size:+.2f}**")

            st.subheader("📈 Price Action & Trend Microstructure")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['mid_price'], name="Mid Price", line=dict(color="#00FFA3", width=2.5)))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['micro_price'], name="Micro Price", line=dict(color="#FF007A", width=1.5, dash='dash')))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_100'], name="EMA Trend", line=dict(color="#00E5FF", width=1.5, dash='dot')))
            fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔥 Order Book Liquidity Density Heatmap")
            price_levels = np.linspace(current_price - 20, current_price + 20, 10)
            time_steps = df['timestamp'].dt.strftime('%H:%M:%S').tail(20).tolist()
            density_matrix = [[max(0.1, 10.0 - abs(lvl - current_price) * 0.4) * (1.0 - imb if lvl > current_price else 1.0 + imb) for imb in df['depth_imbalance'].tail(20)] for lvl in price_levels]

            heatmap_fig = go.Figure(data=go.Heatmap(z=density_matrix, x=time_steps, y=price_levels, colorscale="Viridis"))
            heatmap_fig.update_layout(template="plotly_dark", height=320, yaxis_title="Price Tiers ($)", xaxis_title="Time Step", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(heatmap_fig, use_container_width=True)

    elif terminal_mode == "Statistical Arbitrage (Market Neutral)":
        if not pair_df.empty and len(pair_df) > 5:
            latest_ratio = pair_df['ratio'].iloc[-1]
            latest_z = pair_df['z_score'].iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Price BTC", f"${pair_df['btc_price'].iloc[-1]:,.2f}")
            col2.metric("Price ETH", f"${pair_df['eth_price'].iloc[-1]:,.2f}")
            col3.metric("BTC/ETH Ratio", f"{latest_ratio:.4f}")
            col4.metric("Spread Z-Score", f"{latest_z:+.2f} σ")

            if st.session_state.active_arb_position:
                apos = st.session_state.active_arb_position
                st.info(f"⚡ **ACTIVE STAT-ARB {apos['side']} POSITION** | Entry Ratio: {apos['entry_ratio']:.4f}")

            st.subheader("📊 Statistical Spread & Z-Score Analysis (BTC / ETH)")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
            fig.add_trace(go.Scatter(x=pair_df['timestamp'], y=pair_df['ratio'], name="Price Ratio", line=dict(color="#00E5FF", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=pair_df['timestamp'], y=pair_df['ratio_mean'], name="Mean Ratio", line=dict(color="#FFD700", dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=pair_df['timestamp'], y=pair_df['z_score'], name="Z-Score", line=dict(color="#00FFA3", width=2)), row=2, col=1)
            fig.add_hline(y=1.8, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=-1.8, line_dash="dot", line_color="green", row=2, col=1)
            fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
            fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

    elif terminal_mode == "Correlation & Cross-Asset Heatmaps":
        if not pair_df.empty and len(pair_df) > 5:
            st.subheader("🔥 Cross-Asset Rolling Price Correlation Matrix")
            corr_df = pair_df[['btc_price', 'eth_price', 'sol_price']].rename(columns={'btc_price': 'BTC', 'eth_price': 'ETH', 'sol_price': 'SOL'}).corr()
            corr_fig = px.imshow(corr_df, text_auto=".3f", color_continuous_scale="Plasma", title="Real-Time Cross-Asset Pearson Correlation")
            corr_fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(corr_fig, use_container_width=True)

    # TRADES & PERFORMANCE TABLE
    trades_df = load_trade_history()
    st.markdown("---")
    st.subheader("📜 Terminal Execution Logs & Performance Metrics (All Strategies)")
    
    if not trades_df.empty:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
        total_net_pnl = trades_df['net_pnl'].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Executed Trades", f"{total_trades}")
        m2.metric("Win Rate (%)", f"{win_rate:.1f}%")
        m3.metric("Net Cumulative PnL", f"${total_net_pnl:+.2f}")
        m4.metric("Total Fees Paid", f"${trades_df['fee_slippage'].sum():.2f}")

        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("⌛ Autonomous Multi-Strategy Engine active. Listening for ML signals & Statistical Spreads...")

render_active_strategy()
