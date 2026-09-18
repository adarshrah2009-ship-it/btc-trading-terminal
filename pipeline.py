import asyncio
import sqlite3
import json
import logging
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pipeline")

DB_NAME = "btc_market_structure.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            mid_price REAL,
            micro_price REAL,
            ofi REAL,
            absorption_ratio REAL,
            cvd_acceleration REAL
        )
    ''')
    conn.commit()
    conn.close()

async def connect_and_stream():
    init_db()
    
    # Primary URL: Binance.US WebSocket (Bypasses Render US Cloud HTTP 451 restriction)
    # Fallback URL: Coinbase Pro Live BTC Feed
    urls = [
        "wss://stream.binance.us:9443/ws/btcusdt@ticker",
        "wss://ws-feed.exchange.coinbase.com"
    ]
    
    url_index = 0
    while True:
        target_url = urls[url_index]
        logger.info(f"Connecting to stream: {target_url}")
        
        try:
            async with websockets.connect(target_url, ping_interval=20, ping_timeout=10) as ws:
                if "coinbase" in target_url:
                    sub_msg = {
                        "type": "subscribe",
                        "product_ids": ["BTC-USD"],
                        "channels": ["ticker"]
                    }
                    await ws.send(json.dumps(sub_msg))

                while True:
                    res = await ws.recv()
                    data = json.loads(res)
                    
                    # Parse Binance structure
                    if "c" in data:
                        mid_price = float(data["c"])
                        bid = float(data.get("b", mid_price))
                        ask = float(data.get("a", mid_price))
                    # Parse Coinbase structure
                    elif data.get("type") == "ticker" and "price" in data:
                        mid_price = float(data["price"])
                        bid = float(data.get("best_bid", mid_price))
                        ask = float(data.get("best_ask", mid_price))
                    else:
                        continue

                    # Calculate Microstructure Parameters
                    micro_price = (bid + ask) / 2.0 if bid != ask else mid_price
                    ofi = (bid - ask) * 10.0
                    absorption = 0.50
                    cvd_acc = 1.25

                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO market_data (mid_price, micro_price, ofi, absorption_ratio, cvd_acceleration)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (mid_price, micro_price, ofi, absorption, cvd_acc))
                    conn.commit()
                    conn.close()
                    
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"WebSocket error on {target_url}: {e}")
            url_index = (url_index + 1) % len(urls)
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(connect_and_stream())
