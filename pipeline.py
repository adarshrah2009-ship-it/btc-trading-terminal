import asyncio
import json
import sqlite3
import pandas as pd
import numpy as np
import websockets
import os
import time

DB_NAME = "btc_market_structure.db"

# --- 1. INITIALIZE SQLITE MARKET DATABASE ---
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
            bid_depth_vol REAL,
            ask_depth_vol REAL,
            absorption_ratio REAL,
            cvd_acceleration REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- 2. MULTI-LEVEL ORDER FLOW IMBALANCE (MLOFI) CALCULATOR ---
class MicrostructureEngineL2:
    def __init__(self, depth_levels=20):
        self.depth_levels = depth_levels
        self.prev_bids = None
        self.prev_asks = None
        self.cvd = 0.0
        self.last_cvd_accel = 0.0

    def process_l2_depth(self, bids, asks):
        # Convert bids/asks to float arrays: [[price, qty], ...]
        bids = np.array(bids[:self.depth_levels], dtype=float)
        asks = np.array(asks[:self.depth_levels], dtype=float)
        
        best_bid, bid_vol_l1 = bids[0, 0], bids[0, 1]
        best_ask, ask_vol_l1 = asks[0, 0], asks[0, 1]
        
        mid_price = (best_bid + best_ask) / 2.0
        
        # Volume-weighted Micro-Price (L1)
        micro_price = ((best_bid * ask_vol_l1) + (best_ask * bid_vol_l1)) / (bid_vol_l1 + ask_vol_l1)
        
        # Top-of-Book OFI (L1)
        ofi_l1 = 0.0
        if self.prev_bids is not None and self.prev_asks is not None:
            prev_best_bid = self.prev_bids[0, 0]
            prev_bid_qty = self.prev_bids[0, 1]
            prev_best_ask = self.prev_asks[0, 0]
            prev_ask_qty = self.prev_asks[0, 1]
            
            # Bid-side change
            if best_bid > prev_best_bid:
                delta_bid = bid_vol_l1
            elif best_bid == prev_best_bid:
                delta_bid = bid_vol_l1 - prev_bid_qty
            else:
                delta_bid = 0.0
                
            # Ask-side change
            if best_ask < prev_best_ask:
                delta_ask = ask_vol_l1
            elif best_ask == prev_best_ask:
                delta_ask = ask_vol_l1 - prev_ask_qty
            else:
                delta_ask = 0.0
                
            ofi_l1 = delta_bid - delta_ask

        # Multi-Level Order Flow Imbalance (MLOFI across 20 depth levels)
        # Weights decay exponentially with depth level
        weights = 1.0 / np.arange(1, len(bids) + 1)
        bid_depth_vol = np.sum(bids[:, 1] * weights)
        ask_depth_vol = np.sum(asks[:, 1] * weights)
        
        mlofi = bid_depth_vol - ask_depth_vol
        absorption_ratio = bid_depth_vol / (ask_depth_vol + 1e-5)
        
        # Aggregate CVD Acceleration Proxy
        self.cvd += ofi_l1
        cvd_acceleration = ofi_l1 - self.last_cvd_accel
        self.last_cvd_accel = ofi_l1
        
        # Save state for next tick comparison
        self.prev_bids = bids
        self.prev_asks = asks
        
        return {
            "mid_price": mid_price,
            "micro_price": micro_price,
            "ofi": ofi_l1,
            "mlofi": mlofi,
            "bid_depth_vol": bid_depth_vol,
            "ask_depth_vol": ask_depth_vol,
            "absorption_ratio": absorption_ratio,
            "cvd_acceleration": cvd_acceleration
        }

# --- 3. PERSIST TO SQLITE ---
def save_to_db(metrics):
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO market_data 
        (timestamp, mid_price, micro_price, ofi, mlofi, bid_depth_vol, ask_depth_vol, absorption_ratio, cvd_acceleration)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp_str,
        metrics["mid_price"],
        metrics["micro_price"],
        metrics["ofi"],
        metrics["mlofi"],
        metrics["bid_depth_vol"],
        metrics["ask_depth_vol"],
        metrics["absorption_ratio"],
        metrics["cvd_acceleration"]
    ))
    conn.commit()
    conn.close()

# --- 4. WEBSOCKET ENGINE (BINANCE L2 DEPTH STREAM) ---
async def stream_l2_microstructure():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"
    engine = MicrostructureEngineL2(depth_levels=20)
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("⚡ L2 Depth Stream Connected successfully.")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if "bids" in data and "asks" in data:
                        metrics = engine.process_l2_depth(data["bids"], data["asks"])
                        save_to_db(metrics)
                        
                    await asyncio.sleep(0.1) # 100ms processing loop
        except Exception as e:
            print(f"⚠️ Connection interrupted: {e}. Reconnecting in 3 seconds...")
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(stream_l2_microstructure())
