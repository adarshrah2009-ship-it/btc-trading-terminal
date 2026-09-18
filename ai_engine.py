import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier

DB_NAME = "btc_market_structure.db"
MODEL_FILE = "btc_ai_model.pkl"

def apply_triple_barrier(df, tp_dist=50.0, sl_dist=25.0, horizon=50):
    """
    Triple-Barrier Labeling:
    Label = 1 if price hits Entry + TP before Entry - SL within 'horizon' ticks.
    Label = 0 otherwise.
    """
    labels = []
    prices = df['price'].values
    n = len(prices)
    
    for i in range(n):
        if i + horizon >= n:
            labels.append(np.nan)
            continue
            
        entry = prices[i]
        tp_target = entry + tp_dist
        sl_target = entry - sl_dist
        
        future_prices = prices[i+1 : i+1+horizon]
        
        label = 0
        for p in future_prices:
            if p >= tp_target:
                label = 1
                break
            elif p <= sl_target:
                label = 0
                break
        labels.append(label)
        
    return pd.Series(labels, index=df.index)

def train_local_ai():
    """Reads SQLite database, builds predictive microstructure features, and trains XGBoost."""
    if not os.path.exists(DB_NAME):
        return "Database file not found. Ensure pipeline.py is running!"
        
    try:
        conn = sqlite3.connect(f"file:{DB_NAME}?mode=ro", uri=True)
        trades = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp ASC", conn)
        ob = pd.read_sql_query("SELECT * FROM orderbook_snapshots ORDER BY timestamp ASC", conn)
        conn.close()
        
        if len(trades) < 250:
            return f"Need more tick data to train accurately ({len(trades)}/250 rows collected)."
            
        # 1. Order Flow Features
        trades['signed_vol'] = trades.apply(lambda r: r['quantity'] if r['is_buyer_maker'] == 0 else -r['quantity'], axis=1)
        trades['buy_vol'] = trades.apply(lambda r: r['quantity'] if r['is_buyer_maker'] == 0 else 0.0, axis=1)
        trades['sell_vol'] = trades.apply(lambda r: r['quantity'] if r['is_buyer_maker'] == 1 else 0.0, axis=1)
        
        trades['cvd'] = trades['signed_vol'].cumsum()
        trades['vwap'] = (trades['price'] * trades['quantity']).cumsum() / trades['quantity'].cumsum()
        trades['vwap_dist'] = trades['price'] - trades['vwap']
        
        # 2. Predictive Features: Aggression & Delta Acceleration
        trades['aggression_ratio'] = (trades['buy_vol'].rolling(20).sum() + 1e-9) / (trades['sell_vol'].rolling(20).sum() + 1e-9)
        trades['cvd_accel'] = trades['cvd'] - trades['cvd'].shift(20)
        
        # 3. Merge Orderbook & Calculate Micro-Price / Absorption
        if not ob.empty and 'obi' in ob.columns:
            df = pd.merge_asof(trades, ob[['timestamp', 'obi', 'bids_vol', 'asks_vol']], on='timestamp', direction='backward')
            df['depth_ratio'] = (df['bids_vol'] + 1e-9) / (df['asks_vol'] + 1e-9)
            
            # Absorption: Executed buy volume vs resting ask volume
            df['absorption_ratio'] = (df['buy_vol'].rolling(10).sum() + 1e-9) / (df['asks_vol'] + 1e-9)
            
            # Order Flow Imbalance Proxy
            df['ofi'] = df['bids_vol'].diff() - df['asks_vol'].diff()
        else:
            df = trades.copy()
            df['obi'] = 0.0
            df['bids_vol'] = 0.0
            df['asks_vol'] = 0.0
            df['depth_ratio'] = 1.0
            df['absorption_ratio'] = 1.0
            df['ofi'] = 0.0

        # 4. Triple-Barrier Target Labeling
        df['target'] = apply_triple_barrier(df, tp_dist=50.0, sl_dist=25.0, horizon=50)
        
        df = df.dropna()
        
        features = ['signed_vol', 'cvd', 'vwap_dist', 'aggression_ratio', 'cvd_accel', 'obi', 'depth_ratio', 'absorption_ratio', 'ofi']
        X = df[features]
        y = df['target'].astype(int)
        
        if len(np.unique(y)) < 2:
            return "Dataset imbalance detected. Let pipeline.py gather more ticks across active price movement!"
            
        # Train XGBoost
        clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42)
        clf.fit(X, y)
        
        joblib.dump(clf, MODEL_FILE)
        
        acc = clf.score(X, y) * 100
        return f"⚡ XGBoost Predictive Order Flow Model Trained! Rows: {len(X)} | Target Accuracy: {acc:.1f}%"
        
    except Exception as e:
        return f"Training Error: {str(e)}"


def predict_trade_probability(features_dict):
    """Predicts breakout probability using predictive order flow inputs."""
    if not os.path.exists(MODEL_FILE):
        return 0.60  
        
    try:
        model = joblib.load(MODEL_FILE)
        
        df_input = pd.DataFrame([{
            'signed_vol': features_dict.get('signed_vol', 0.0),
            'cvd': features_dict.get('cvd', 0.0),
            'vwap_dist': features_dict.get('vwap_dist', 0.0),
            'aggression_ratio': features_dict.get('aggression_ratio', 1.0),
            'cvd_accel': features_dict.get('cvd_accel', 0.0),
            'obi': features_dict.get('obi', 0.0),
            'depth_ratio': features_dict.get('depth_ratio', 1.0),
            'absorption_ratio': features_dict.get('absorption_ratio', 1.0),
            'ofi': features_dict.get('ofi', 0.0)
        }])
        
        probabilities = model.predict_proba(df_input)[0]
        
        if len(probabilities) < 2:
            return 0.60
            
        return float(probabilities[1])
        
    except Exception:
        return 0.60