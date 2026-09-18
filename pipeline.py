import asyncio
import json
import logging
import sqlite3
import time
import aiohttp
import websockets

DB_NAME = "btc_market_structure.db"
COINGLASS_API_KEY = "YOUR_COINGLASS_API_KEY"

# Use Binance Spot Streams (Unrestricted by local ISP filters)
BINANCE_TRADE_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
BINANCE_DEPTH_URL = "wss://stream.binance.com:9443/ws/btcusdt@depth5@100ms"
COINGLASS_HEATMAP_URL = "https://open-api-v4.coinglass.com/api/futures/liquidation/heatmap/model1?exchange=Binance&symbol=BTCUSDT&range=3d"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=60000;")
    return conn

def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Reset trades table to use auto-incrementing ID primary key
        cursor.execute("DROP TABLE IF EXISTS trades")
        cursor.execute('''
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                price REAL,
                quantity REAL,
                is_buyer_maker INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                timestamp INTEGER PRIMARY KEY,
                best_bid_price REAL,
                best_bid_qty REAL,
                best_ask_price REAL,
                best_ask_qty REAL,
                obi REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS liquidation_heatmaps (
                timestamp INTEGER PRIMARY KEY,
                raw_payload TEXT
            )
        ''')
        conn.commit()
    logging.info("SQLite Database initialized cleanly.")

async def poll_coinglass_heatmap():
    headers = {"accept": "application/json", "CG-API-KEY": COINGLASS_API_KEY}
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(COINGLASS_HEATMAP_URL, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        ts = int(time.time() * 1000)
                        with get_db_connection() as conn:
                            conn.execute(
                                "INSERT OR REPLACE INTO liquidation_heatmaps VALUES (?, ?)",
                                (ts, json.dumps(data.get("data", {})))
                            )
                            conn.commit()
                        logging.info("Successfully logged CoinGlass Liquidation Heatmap.")
            except Exception as e:
                logging.error(f"CoinGlass API fetch deferred: {e}")
            await asyncio.sleep(300)

# --- STREAM 1: AGGREGATE TRADES ---
async def process_trades_websocket():
    trade_count = 0
    while True:
        try:
            async with websockets.connect(BINANCE_TRADE_URL, ping_interval=20, ping_timeout=20) as ws:
                logging.info("Connected to Binance Spot AggTrade Stream.")

                while True:
                    message = await ws.recv()
                    data = json.loads(message)

                    ts = data.get('T') or data.get('E') or int(time.time() * 1000)
                    price = float(data.get('p', 0))
                    qty = float(data.get('q', 0))
                    is_buyer_maker = 1 if data.get('m', False) else 0

                    if price > 0 and qty > 0:
                        with get_db_connection() as conn:
                            conn.execute(
                                "INSERT INTO trades (timestamp, price, quantity, is_buyer_maker) VALUES (?, ?, ?, ?)",
                                (ts, price, qty, is_buyer_maker)
                            )
                            conn.commit()
                        trade_count += 1
                        if trade_count % 10 == 0:
                            logging.info(f"[+] Written {trade_count} trades to SQLite database.")

        except websockets.exceptions.ConnectionClosed:
            await asyncio.sleep(3)
        except Exception as e:
            logging.error(f"Trades WebSocket error: {e}")
            await asyncio.sleep(3)

# --- STREAM 2: ORDERBOOK DEPTH ---
async def process_depth_websocket():
    while True:
        try:
            async with websockets.connect(BINANCE_DEPTH_URL, ping_interval=20, ping_timeout=20) as ws:
                logging.info("Connected to Binance Spot Depth Stream.")
                batch_orderbooks = []

                while True:
                    message = await ws.recv()
                    data = json.loads(message)

                    bids = data.get('b', [])
                    asks = data.get('a', [])
                    if bids and asks:
                        total_bid_vol = sum(float(b[1]) for b in bids)
                        total_ask_vol = sum(float(a[1]) for a in asks)
                        denom = total_bid_vol + total_ask_vol
                        obi = (total_bid_vol - total_ask_vol) / denom if denom > 0 else 0.0

                        batch_orderbooks.append((
                            data.get('E', int(time.time() * 1000)),
                            float(bids[0][0]), float(bids[0][1]),
                            float(asks[0][0]), float(asks[0][1]),
                            obi
                        ))

                    if len(batch_orderbooks) >= 20:
                        with get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.executemany("INSERT OR IGNORE INTO orderbook_snapshots VALUES (?, ?, ?, ?, ?, ?)", batch_orderbooks)
                            conn.commit()
                        batch_orderbooks.clear()

        except websockets.exceptions.ConnectionClosed:
            await asyncio.sleep(3)
        except Exception as e:
            logging.error(f"Depth WebSocket error: {e}")
            await asyncio.sleep(3)

async def main():
    init_db()
    await asyncio.gather(
        process_trades_websocket(),
        process_depth_websocket(),
        poll_coinglass_heatmap()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Data pipeline stopped by user.")