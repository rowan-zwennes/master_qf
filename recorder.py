"""Binance WebSocket recorder: book diffs, snapshots, trades and mark price."""
import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

import aiohttp
import orjson
import time
import gzip
import os
import csv
import threading
import queue
import gc
from datetime import datetime, timezone
import signal
import sys

gc.set_threshold(50000, 10, 10)
SYMBOL = "btcusdt"

STREAM_URL = f"wss://fstream.binance.com/stream?streams={SYMBOL}@aggTrade/{SYMBOL}@depth20@100ms/{SYMBOL}@depth@100ms/{SYMBOL}@markPrice@1s"

# VPS Storage Config
QUEUE_SIZE = 2_000_000
GZIP_LEVEL = 1
FLUSH_INTERVAL = 2.0
BATCH_SIZE = 10000

# Global control
running = True
write_queue = queue.Queue(maxsize=QUEUE_SIZE)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class LogFileHandler:
    def __init__(self, symbol, log_type):
        self.symbol = symbol
        self.type = log_type
        self.current_hour = None
        self.file_handle = None
        self.csv_writer = None

    def _get_path(self, now_utc):
        custom_path = "/mnt/extra_storage"
        base_dir = os.path.join(custom_path, "data", self.symbol, now_utc.strftime('%Y-%m-%d'))
        os.makedirs(base_dir, exist_ok=True)
        hour_str = now_utc.strftime("%H")
        filename = f"{self.type}_{hour_str}h.csv.gz"
        return os.path.join(base_dir, filename), hour_str

    def write(self, rows):
        if not rows: return
        now_utc = datetime.now(timezone.utc)
        target_hour = now_utc.strftime("%H")

        if self.file_handle is None or self.current_hour != target_hour:
            self.close()
            filepath, self.current_hour = self._get_path(now_utc)
            if self.file_handle is None:
                log(f"[DISK] Writing to: {filepath}")
            exists = os.path.exists(filepath)
            self.file_handle = gzip.open(filepath, "at", encoding="utf-8", newline='', compresslevel=GZIP_LEVEL)
            self.csv_writer = csv.writer(self.file_handle)

            if not exists:
                if self.type == "trades":
                    self.csv_writer.writerow(
                        ["LocalTime", "EventTime", "TransactTime", "id", "Price", "Quantity", "MakerWasBuyer"])
                elif self.type == "book":
                    self.csv_writer.writerow(
                        ["LocalTime", "EventTime", "TransactTime", "FirstUpdateID", "FinalUpdateID", "Bids", "Asks"])
                elif self.type == "diffs":
                    self.csv_writer.writerow(
                        ["LocalTime", "EventTime", "TransactTime", "FirstUpdateID", "FinalUpdateID",
                         "PrevFinalUpdateID", "Bids", "Asks"])
                elif self.type == "markprice":
                    self.csv_writer.writerow(
                        ["LocalTime", "EventTime", "MarkPrice", "IndexPrice", "EstSettlePrice", "FundingRate",
                         "NextFundingTime"])

        try:
            self.csv_writer.writerows(rows)
        except Exception as e:
            log(f"DISK ERROR ({self.type}): {e}")

    def close(self):
        if self.file_handle:
            try:
                self.file_handle.close()
            except:
                pass
            self.file_handle = None


def writer_thread_loop():
    """FIX: Correcte Mapping voor Binance Futures (fstream)"""
    log(f"Writer Thread Started [PID: {os.getpid()}]")

    loggers = {
        't': LogFileHandler(SYMBOL, "trades"),
        'b': LogFileHandler(SYMBOL, "book"),
        'd': LogFileHandler(SYMBOL, "diffs"),
        'm': LogFileHandler(SYMBOL, "markprice")  
    }

    buffers = {'t': [], 'b': [], 'd': [], 'm': []}  
    last_flush = time.time()

    while running or not write_queue.empty():
        try:
            try:
                item = write_queue.get(timeout=0.5)
            except queue.Empty:
                if not running: break
                continue

            tag, data = item[0], item[1]

            if tag == 't':  # Trades
                buffers['t'].append(data)

            elif tag == 'b':  # Snapshot 
                ts, payload = data
                try:
                    bids = orjson.dumps(payload['b']).decode('utf-8')
                    asks = orjson.dumps(payload['a']).decode('utf-8')
                    row = [
                        ts,
                        payload.get('E', 0),
                        payload.get('T', 0),
                        payload.get('U', 0),
                        payload.get('u', 0),
                        bids,
                        asks
                    ]
                    buffers['b'].append(row)
                except Exception:
                    pass

            elif tag == 'd':  # Diffs (Depth)
                ts, payload = data
                try:
                    bids = orjson.dumps(payload['b']).decode('utf-8')
                    asks = orjson.dumps(payload['a']).decode('utf-8')
                    row = [
                        ts,
                        payload.get('E', 0),
                        payload.get('T', 0),
                        payload.get('U'),
                        payload.get('u'),
                        payload.get('pu'),
                        bids,
                        asks
                    ]
                    buffers['d'].append(row)
                except Exception:
                    pass

            elif tag == 'm':
                ts, payload = data
                try:
                    row = [
                        ts,
                        payload.get('E', 0),  # Event Time
                        payload.get('p', 0),  # Mark Price
                        payload.get('i', 0),  # Index Price
                        payload.get('P', 0),  # Est Settle Price
                        payload.get('r', 0),  # Funding Rate 
                        payload.get('T', 0)  # Next Funding Time
                    ]
                    buffers['m'].append(row)
                except Exception:
                    pass

            write_queue.task_done()

            for _ in range(BATCH_SIZE):
                if write_queue.empty(): break
                try:
                    item = write_queue.get_nowait()
                    tag, data = item[0], item[1]

                    if tag == 't':
                        buffers['t'].append(data)
                    elif tag == 'b':
                        ts, payload = data
                        buffers['b'].append([
                            ts, payload.get('E', 0), payload.get('T', 0),
                            payload.get('U', 0), payload.get('u', 0),
                            orjson.dumps(payload['b']).decode('utf-8'),
                            orjson.dumps(payload['a']).decode('utf-8')
                        ])
                    elif tag == 'd':
                        ts, payload = data
                        buffers['d'].append([
                            ts, payload.get('E', 0), payload.get('T', 0),
                            payload.get('U'), payload.get('u'), payload.get('pu'),
                            orjson.dumps(payload['b']).decode('utf-8'),
                            orjson.dumps(payload['a']).decode('utf-8')
                        ])

                    elif tag == 'm':
                        ts, payload = data
                        buffers['m'].append([
                            ts, payload.get('E', 0), payload.get('p', 0),
                            payload.get('i', 0), payload.get('P', 0),
                            payload.get('r', 0), payload.get('T', 0)
                        ])

                    write_queue.task_done()
                except queue.Empty:
                    break
                except Exception:
                    pass

        except Exception as e:
            log(f"Thread Error: {e}")

        # Flush Timer
        now = time.time()
        if now - last_flush > FLUSH_INTERVAL:
            for tag, buf in buffers.items():
                if buf:
                    loggers[tag].write(buf)
                    buf.clear()
            last_flush = now

    # Cleanup 
    for tag, buf in buffers.items():
        if buf: loggers[tag].write(buf)
        loggers[tag].close()


async def process_messages(ws):
    """FIX: Super lichtgewicht Producer."""
    msg_count = 0
    last_report = time.time()

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            ts = time.time_ns() // 1000

            try:
                data = orjson.loads(msg.data)
                if 'data' not in data: continue

                payload = data['data']
                stream = data.get('stream', '')

                if 'aggTrade' in stream:
                    row = [ts, payload.get('E', 0), payload.get('T', 0), payload.get('a', 0), payload['p'],
                           payload['q'], payload['m']]
                    write_queue.put_nowait(('t', row))

                elif 'depth20' in stream:
                    write_queue.put_nowait(('b', (ts, payload)))

                elif 'depth' in stream:
                    write_queue.put_nowait(('d', (ts, payload)))

                elif 'markPrice' in stream:
                    write_queue.put_nowait(('m', (ts, payload)))

                msg_count += 1

            except queue.Full:
                log("QUEUE FULL!")
            except Exception:
                pass

            now = time.time()
            if now - last_report > 60:
                q_size = write_queue.qsize()
                log(f"Heartbeat | Processed: {msg_count} msgs | Queue: {q_size}")
                msg_count = 0
                last_report = now
        elif msg.type == aiohttp.WSMsgType.ERROR:
            break


async def main_loop():
    log(f"Async Recorder FINAL (Threaded): {SYMBOL}")

    writer = threading.Thread(target=writer_thread_loop, daemon=True)
    writer.start()

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)

    while running:
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                log(f"Connecting to {STREAM_URL}...")

                async with session.ws_connect(
                    STREAM_URL,
                    heartbeat=20,
                    autoping=True,
                    compress=0,
                    max_msg_size=0 
                ) as ws:
                    log("Connected! Waiting for data...")
                    await process_messages(ws)

        except aiohttp.ClientError as e:
            log(f"Network Error: {e} - Reconnecting in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            log(f"Critical Error: {e} - Reconnecting in 5s...")
            await asyncio.sleep(5)

    writer.join()


def handle_exit(signum, frame):
    global running
    log("\nStopping...")
    running = False


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_loop())
