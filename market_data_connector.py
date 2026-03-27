"""
Real Market Data Connector — Philip AJ Sogah
Connects the ForensicSignalFusion + RMD engines to live market data.

Supports:
  - Alpaca Markets  (free tier, US equities + crypto, WebSocket + REST)
  - Polygon.io      (paid, best order book depth data, WebSocket + REST)

Usage:
    # Alpaca (free)
    connector = AlpacaConnector(api_key="YOUR_KEY", api_secret="YOUR_SECRET")
    runner = LiveForensicsRunner(connector, symbols=["AAPL", "TSLA"])
    await runner.start()

    # Polygon.io
    connector = PolygonConnector(api_key="YOUR_KEY")
    runner = LiveForensicsRunner(connector, symbols=["AAPL"])
    await runner.start()

    # Paper-trade mode (no real money, logs signals only)
    runner = LiveForensicsRunner(connector, symbols=["AAPL"], paper_trade=True)
"""

import asyncio
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import websockets

from forensics_engine import (
    ForensicSignalFusion,
    ForensicResult,
    MarketSnapshot,
    RMDAlgorithm,
    RMDSignal,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
#  ROLLING BASELINE TRACKER
#  Computes per-symbol, per-time-of-day baselines from
#  live data so the forensic engine has real reference values.
# ─────────────────────────────────────────────────────────

class RollingBaseline:
    """
    Tracks rolling statistics per symbol:
      - median volume per 1-min bar (last 20 bars)
      - median trade count per bar  (last 20 bars)
    These become the baseline_volume / baseline_trade_rate
    fed into ForensicSignalFusion.
    """
    def __init__(self, window: int = 20):
        self.window = window
        self._volumes: deque = deque(maxlen=window)
        self._trades:  deque = deque(maxlen=window)

    def update(self, volume: float, trade_count: int):
        self._volumes.append(volume)
        self._trades.append(trade_count)

    @property
    def baseline_volume(self) -> float:
        if not self._volumes:
            return 1000.0
        s = sorted(self._volumes)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else (s[mid-1] + s[mid]) / 2

    @property
    def baseline_trade_rate(self) -> float:
        if not self._trades:
            return 10.0
        s = sorted(self._trades)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else (s[mid-1] + s[mid]) / 2


# ─────────────────────────────────────────────────────────
#  ORDER BOOK AGGREGATOR
#  Maintains a live L2 book from incremental updates
# ─────────────────────────────────────────────────────────

class OrderBook:
    """
    Maintains a live Level-2 order book.
    Bids and asks are stored as {price: size} dicts.
    """
    def __init__(self, depth: int = 10):
        self.depth = depth
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self._trade_count:  int   = 0
        self._quote_total:  int   = 0
        self._quote_cancels:int   = 0
        self._volume:       float = 0.0
        self._last_reset:   float = time.time()

    def apply_bid(self, price: float, size: float):
        """Update or remove a bid level."""
        self._quote_total += 1
        if size == 0:
            self._quote_cancels += 1
            self.bids.pop(price, None)
        else:
            self.bids[price] = size

    def apply_ask(self, price: float, size: float):
        """Update or remove an ask level."""
        self._quote_total += 1
        if size == 0:
            self._quote_cancels += 1
            self.asks.pop(price, None)
        else:
            self.asks[price] = size

    def record_trade(self, price: float, size: float):
        self._trade_count += 1
        self._volume += size

    def snapshot(self, current_price: float) -> MarketSnapshot:
        """Convert current book state into a MarketSnapshot for the engine."""
        best_bids = sorted(self.bids.items(), reverse=True)[:self.depth]
        best_asks = sorted(self.asks.items())[:self.depth]

        # Pad to depth with zeros if book is thin
        bid_depth = [s for _, s in best_bids] + [0.0] * (self.depth - len(best_bids))
        ask_depth = [s for _, s in best_asks] + [0.0] * (self.depth - len(best_asks))

        snap = MarketSnapshot(
            timestamp    = time.time(),
            price        = current_price,
            volume       = self._volume,
            bid_depth    = bid_depth,
            ask_depth    = ask_depth,
            trade_count  = self._trade_count,
            quote_cancels= self._quote_cancels,
            quote_total  = max(self._quote_total, 1),
        )
        self._reset_interval_stats()
        return snap

    def _reset_interval_stats(self):
        self._trade_count   = 0
        self._quote_total   = 0
        self._quote_cancels = 0
        self._volume        = 0.0
        self._last_reset    = time.time()


# ─────────────────────────────────────────────────────────
#  BASE CONNECTOR INTERFACE
# ─────────────────────────────────────────────────────────

@dataclass
class LiveQuote:
    symbol:     str
    price:      float
    bid:        float
    ask:        float
    bid_size:   float
    ask_size:   float
    timestamp:  float


class BaseConnector(ABC):
    """Abstract base — implement this to add a new data provider."""

    @abstractmethod
    async def connect(self): ...

    @abstractmethod
    async def subscribe(self, symbols: list[str]): ...

    @abstractmethod
    async def stream(self, on_quote: Callable, on_trade: Callable, on_book: Callable): ...

    @abstractmethod
    async def disconnect(self): ...

    @abstractmethod
    def provider_name(self) -> str: ...


# ─────────────────────────────────────────────────────────
#  ALPACA CONNECTOR  (free tier, US equities + crypto)
#  Docs: https://docs.alpaca.markets/reference/marketplace
# ─────────────────────────────────────────────────────────

class AlpacaConnector(BaseConnector):
    """
    Alpaca Markets WebSocket connector.
    Free plan: real-time trades + quotes (SIP feed).
    Paid plan: full L2 order book via "iex" or "sip" feed.

    Sign up: https://alpaca.markets  (free)
    Keys:    Settings → API Keys → Generate New Key
    """

    WS_URL_PAPER = "wss://stream.data.alpaca.markets/v2/iex"
    WS_URL_LIVE  = "wss://stream.data.alpaca.markets/v2/sip"

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.ws_url     = self.WS_URL_PAPER if paper else self.WS_URL_LIVE
        self._ws        = None

    def provider_name(self) -> str:
        return "Alpaca Markets"

    async def connect(self):
        self._ws = await websockets.connect(self.ws_url)
        msg = json.loads(await self._ws.recv())
        log.info("Alpaca: %s", msg)

        # Authenticate
        await self._ws.send(json.dumps({
            "action": "auth",
            "key":    self.api_key,
            "secret": self.api_secret,
        }))
        auth_resp = json.loads(await self._ws.recv())
        log.info("Alpaca auth: %s", auth_resp)

    async def subscribe(self, symbols: list[str]):
        await self._ws.send(json.dumps({
            "action": "subscribe",
            "trades": symbols,
            "quotes": symbols,
            "bars":   symbols,
        }))
        resp = json.loads(await self._ws.recv())
        log.info("Alpaca subscribed: %s", resp)

    async def stream(self,
                     on_quote: Callable,
                     on_trade: Callable,
                     on_book:  Callable):
        """
        Stream live messages and dispatch to callbacks.
        on_quote(symbol, bid, ask, bid_size, ask_size, price, ts)
        on_trade(symbol, price, size, ts)
        on_book  — not available on free Alpaca plan; pass None
        """
        async for raw in self._ws:
            messages = json.loads(raw)
            if not isinstance(messages, list):
                messages = [messages]
            for msg in messages:
                t = msg.get("T")
                if t == "q":   # quote update
                    await on_quote(
                        msg["S"],               # symbol
                        float(msg.get("bp", 0)),# bid price
                        float(msg.get("ap", 0)),# ask price
                        float(msg.get("bs", 0)),# bid size
                        float(msg.get("as", 0)),# ask size
                        (float(msg.get("bp",0)) + float(msg.get("ap",0))) / 2,
                        msg.get("t", ""),
                    )
                elif t == "t": # trade
                    await on_trade(
                        msg["S"],
                        float(msg.get("p", 0)),
                        float(msg.get("s", 0)),
                        msg.get("t", ""),
                    )

    async def disconnect(self):
        if self._ws:
            await self._ws.close()
            log.info("Alpaca: disconnected")


# ─────────────────────────────────────────────────────────
#  POLYGON.IO CONNECTOR  (paid, best L2 order book data)
#  Docs: https://polygon.io/docs/stocks/ws_stocks_q
# ─────────────────────────────────────────────────────────

class PolygonConnector(BaseConnector):
    """
    Polygon.io WebSocket connector.
    Starter plan ($29/mo): real-time trades + quotes + L2 snapshots.
    Best for full order book depth — needed for entropy signal.

    Sign up: https://polygon.io
    Keys:    Dashboard → API Keys
    """

    WS_URL = "wss://socket.polygon.io/stocks"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._ws     = None

    def provider_name(self) -> str:
        return "Polygon.io"

    async def connect(self):
        self._ws = await websockets.connect(self.WS_URL)
        await self._ws.recv()  # connected message
        await self._ws.send(json.dumps({
            "action": "auth",
            "params": self.api_key,
        }))
        resp = json.loads(await self._ws.recv())
        log.info("Polygon auth: %s", resp)

    async def subscribe(self, symbols: list[str]):
        # Q.* = quotes, T.* = trades, A.* = per-second aggregates
        channels = []
        for s in symbols:
            channels += [f"Q.{s}", f"T.{s}", f"A.{s}"]
        await self._ws.send(json.dumps({
            "action":  "subscribe",
            "params":  ",".join(channels),
        }))
        resp = json.loads(await self._ws.recv())
        log.info("Polygon subscribed: %s", resp)

    async def stream(self,
                     on_quote: Callable,
                     on_trade: Callable,
                     on_book:  Callable):
        async for raw in self._ws:
            messages = json.loads(raw)
            if not isinstance(messages, list):
                messages = [messages]
            for msg in messages:
                ev = msg.get("ev")
                if ev == "Q":   # quote
                    await on_quote(
                        msg["sym"],
                        float(msg.get("bp", 0)),
                        float(msg.get("ap", 0)),
                        float(msg.get("bs", 0)),
                        float(msg.get("as", 0)),
                        (float(msg.get("bp",0)) + float(msg.get("ap",0))) / 2,
                        str(msg.get("t", "")),
                    )
                elif ev == "T": # trade
                    await on_trade(
                        msg["sym"],
                        float(msg.get("p", 0)),
                        float(msg.get("s", 0)),
                        str(msg.get("t", "")),
                    )

    async def disconnect(self):
        if self._ws:
            await self._ws.close()
            log.info("Polygon: disconnected")


# ─────────────────────────────────────────────────────────
#  PAPER TRADE LOG
# ─────────────────────────────────────────────────────────

@dataclass
class PaperTrade:
    symbol:        str
    direction:     str
    entry_price:   float
    t1_price:      float
    t2_price:      float
    stop_price:    float
    size_pct:      float
    opened_at:     str
    forensic_score:float
    patterns:      list[str]
    status:        str = "OPEN"
    closed_at:     str = ""
    pnl_pct:       float = 0.0


class PaperTradeLog:
    def __init__(self):
        self.trades: list[PaperTrade] = []

    def open_trade(self, symbol: str, price: float,
                   rmd: RMDSignal, forensic: ForensicResult) -> PaperTrade:
        snap_pct = rmd.expected_snapback_pct / 100
        direction = rmd.direction
        t1 = price * (1 - rmd.t1_target_pct/100) if direction=="SHORT" else price * (1 + rmd.t1_target_pct/100)
        t2 = price * (1 - rmd.t2_target_pct/100) if direction=="SHORT" else price * (1 + rmd.t2_target_pct/100)
        stop = price * (1 + rmd.stop_loss_pct/100) if direction=="SHORT" else price * (1 - rmd.stop_loss_pct/100)

        trade = PaperTrade(
            symbol        = symbol,
            direction     = direction,
            entry_price   = price,
            t1_price      = round(t1, 4),
            t2_price      = round(t2, 4),
            stop_price    = round(stop, 4),
            size_pct      = rmd.position_size_pct,
            opened_at     = datetime.now(timezone.utc).isoformat(),
            forensic_score= forensic.manipulation_score,
            patterns      = forensic.patterns_detected,
        )
        self.trades.append(trade)
        log.info(
            "📊 PAPER TRADE OPENED  %s %s @ %.4f | T1=%.4f T2=%.4f Stop=%.4f | Score=%.1f | %s",
            direction, symbol, price, t1, t2, stop,
            forensic.manipulation_score, forensic.patterns_detected,
        )
        return trade

    def check_exits(self, symbol: str, current_price: float):
        for t in self.trades:
            if t.symbol != symbol or t.status != "OPEN":
                continue
            hit_t1 = (t.direction=="SHORT" and current_price <= t.t1_price) or \
                     (t.direction=="LONG"  and current_price >= t.t1_price)
            hit_stop = (t.direction=="SHORT" and current_price >= t.stop_price) or \
                       (t.direction=="LONG"  and current_price <= t.stop_price)
            if hit_stop:
                pnl = ((t.entry_price - current_price) / t.entry_price * 100) \
                      if t.direction=="SHORT" else \
                      ((current_price - t.entry_price) / t.entry_price * 100)
                t.status = "STOPPED OUT"
                t.closed_at = datetime.now(timezone.utc).isoformat()
                t.pnl_pct = round(pnl, 3)
                log.warning("🛑 STOPPED OUT  %s @ %.4f | PnL=%.3f%%", symbol, current_price, pnl)
            elif hit_t1:
                pnl = ((t.entry_price - current_price) / t.entry_price * 100) \
                      if t.direction=="SHORT" else \
                      ((current_price - t.entry_price) / t.entry_price * 100)
                t.status = "T1 HIT"
                t.closed_at = datetime.now(timezone.utc).isoformat()
                t.pnl_pct = round(pnl, 3)
                log.info("✅ T1 HIT  %s @ %.4f | PnL=%.3f%%", symbol, current_price, pnl)

    def summary(self) -> dict:
        closed = [t for t in self.trades if t.status != "OPEN"]
        winners = [t for t in closed if t.pnl_pct > 0]
        return {
            "total":    len(self.trades),
            "open":     len([t for t in self.trades if t.status=="OPEN"]),
            "closed":   len(closed),
            "winners":  len(winners),
            "win_rate": round(len(winners)/max(len(closed),1)*100, 1),
            "avg_pnl":  round(sum(t.pnl_pct for t in closed)/max(len(closed),1), 3),
        }


# ─────────────────────────────────────────────────────────
#  LIVE FORENSICS RUNNER
#  Ties everything together: data → book → forensics → RMD
# ─────────────────────────────────────────────────────────

class LiveForensicsRunner:
    """
    Main orchestrator. Feeds live market data into the forensic engine
    and RMD algorithm, then logs (or paper-trades) signals.

    Architecture:
        DataProvider (WebSocket)
            → OrderBook (per symbol)
            → RollingBaseline (per symbol)
            → ForensicSignalFusion (per snapshot interval)
            → RMDAlgorithm (gate check → signal)
            → PaperTradeLog (paper mode) / signal callback (live mode)
    """

    def __init__(self,
                 connector:        BaseConnector,
                 symbols:          list[str],
                 snapshot_interval: float = 30.0,   # seconds between forensic runs
                 paper_trade:      bool   = True,
                 manip_threshold:  float  = 60.0,
                 entropy_threshold:float  = 1.5,
                 on_signal:        Optional[Callable] = None):
        """
        Args:
            connector:          AlpacaConnector or PolygonConnector
            symbols:            list of ticker symbols e.g. ["AAPL", "TSLA"]
            snapshot_interval:  how often to run the forensic engine (seconds)
            paper_trade:        if True, log paper trades; if False, call on_signal
            manip_threshold:    RMD gate — minimum manipulation score
            entropy_threshold:  RMD gate — minimum order book entropy
            on_signal:          optional callback(symbol, forensic, rmd_signal, price)
        """
        self.connector         = connector
        self.symbols           = [s.upper() for s in symbols]
        self.snapshot_interval = snapshot_interval
        self.paper_trade       = paper_trade
        self.on_signal         = on_signal

        # Per-symbol state
        self._books:     dict[str, OrderBook]         = {s: OrderBook() for s in self.symbols}
        self._baselines: dict[str, RollingBaseline]   = {s: RollingBaseline() for s in self.symbols}
        self._engines:   dict[str, ForensicSignalFusion] = {}
        self._prices:    dict[str, float]             = {s: 0.0 for s in self.symbols}
        self._last_snap: dict[str, float]             = {s: 0.0 for s in self.symbols}

        self._rmd      = RMDAlgorithm(
            manip_threshold   = manip_threshold,
            entropy_threshold = entropy_threshold,
        )
        self._paper_log = PaperTradeLog()

    async def start(self):
        log.info("Starting LiveForensicsRunner — provider: %s — symbols: %s",
                 self.connector.provider_name(), self.symbols)
        await self.connector.connect()
        await self.connector.subscribe(self.symbols)
        log.info("Connected and subscribed. Waiting for data...")
        await self.connector.stream(
            on_quote = self._on_quote,
            on_trade = self._on_trade,
            on_book  = self._on_book,
        )

    async def stop(self):
        await self.connector.disconnect()
        log.info("Runner stopped.")
        if self.paper_trade:
            log.info("Paper trade summary: %s", self._paper_log.summary())

    # ── callbacks ──────────────────────────────────────────

    async def _on_quote(self, symbol, bid, ask, bid_size, ask_size, price, ts):
        if symbol not in self._books:
            return
        book = self._books[symbol]
        book.apply_bid(bid, bid_size)
        book.apply_ask(ask, ask_size)
        if price > 0:
            self._prices[symbol] = price

        # Check paper trade exits on every tick
        if self.paper_trade and price > 0:
            self._paper_log.check_exits(symbol, price)

        await self._maybe_run_forensics(symbol)

    async def _on_trade(self, symbol, price, size, ts):
        if symbol not in self._books:
            return
        self._books[symbol].record_trade(price, size)
        if price > 0:
            self._prices[symbol] = price

    async def _on_book(self, symbol, bids, asks):
        """Full book snapshot (Polygon L2). bids/asks = list of [price, size]."""
        if symbol not in self._books:
            return
        book = self._books[symbol]
        book.bids = {float(p): float(s) for p, s in bids}
        book.asks = {float(p): float(s) for p, s in asks}

    # ── forensic run ──────────────────────────────────────

    async def _maybe_run_forensics(self, symbol: str):
        now = time.time()
        if now - self._last_snap[symbol] < self.snapshot_interval:
            return
        self._last_snap[symbol] = now

        price = self._prices[symbol]
        if price == 0:
            return

        book     = self._books[symbol]
        baseline = self._baselines[symbol]

        # Build / update the forensic engine with real baselines
        if symbol not in self._engines:
            self._engines[symbol] = ForensicSignalFusion(
                baseline_volume     = baseline.baseline_volume,
                baseline_trade_rate = baseline.baseline_trade_rate,
            )
        else:
            engine = self._engines[symbol]
            engine.baseline_volume     = baseline.baseline_volume
            engine.baseline_trade_rate = baseline.baseline_trade_rate

        snap     = book.snapshot(price)
        baseline.update(snap.volume, snap.trade_count)

        forensic = self._engines[symbol].analyse(snap)

        log.info(
            "%-6s  score=%-5.1f  entropy=%-5.3f  patterns=%s",
            symbol, forensic.manipulation_score,
            forensic.order_book_entropy, forensic.patterns_detected or "none",
        )

        # RMD evaluation
        spike_est = self._estimate_spike(symbol, price)
        rmd_signal = self._rmd.evaluate(forensic, spike_est, time_of_day_risk=self._tod_risk())

        if rmd_signal.trade:
            log.info(
                "  ➜ RMD SIGNAL  %s %s | snap=%.3f%% | size=%.2f%% | conf=%s",
                rmd_signal.direction, symbol,
                rmd_signal.expected_snapback_pct,
                rmd_signal.position_size_pct,
                rmd_signal.confidence,
            )
            if self.paper_trade:
                self._paper_log.open_trade(symbol, price, rmd_signal, forensic)
            if self.on_signal:
                await self.on_signal(symbol, forensic, rmd_signal, price)

    def _estimate_spike(self, symbol: str, current_price: float) -> float:
        """
        Rough spike magnitude estimate: compare current price to the
        rolling median stored in the baseline engine.
        In production, use a proper VWAP or rolling mid calculation.
        """
        engine = self._engines.get(symbol)
        if not engine or not engine._price_history:
            return 0.0
        hist = engine._price_history
        if len(hist) < 2:
            return 0.0
        ref = sorted(hist[-10:])[len(hist[-10:])//2]  # median of last 10 prices
        return (current_price - ref) / ref * 100 if ref > 0 else 0.0

    @staticmethod
    def _tod_risk() -> float:
        """
        Time-of-day risk: higher near market open (9:30–10:00 ET)
        and close (15:30–16:00 ET) when liquidity is thin.
        Returns 0.0–1.0.
        """
        now_utc = datetime.now(timezone.utc)
        hour    = now_utc.hour
        minute  = now_utc.minute
        et_hour = (hour - 4) % 24  # rough ET offset (no DST handling)
        t = et_hour + minute / 60

        # Market open risk window
        if 9.5 <= t <= 10.0:
            return 0.75
        # Market close risk window
        if 15.5 <= t <= 16.0:
            return 0.70
        # Mid-day: low risk
        if 11.0 <= t <= 14.0:
            return 0.25
        return 0.45


# ─────────────────────────────────────────────────────────
#  QUICK-START HELPERS
# ─────────────────────────────────────────────────────────

def run_alpaca(api_key: str, api_secret: str,
               symbols: list[str] = None,
               paper: bool = True):
    """
    One-line launcher for Alpaca.

    Example:
        from market_data_connector import run_alpaca
        run_alpaca("PKXXX", "secretXXX", symbols=["AAPL", "TSLA"])
    """
    connector = AlpacaConnector(api_key, api_secret, paper=paper)
    runner    = LiveForensicsRunner(connector, symbols or ["AAPL", "SPY"])
    asyncio.run(runner.start())


def run_polygon(api_key: str, symbols: list[str] = None):
    """
    One-line launcher for Polygon.io.

    Example:
        from market_data_connector import run_polygon
        run_polygon("YOUR_POLYGON_KEY", symbols=["AAPL"])
    """
    connector = PolygonConnector(api_key)
    runner    = LiveForensicsRunner(connector, symbols or ["AAPL", "SPY"])
    asyncio.run(runner.start())


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    print("""
╔══════════════════════════════════════════════════════╗
║   MARKET FORENSICS ENGINE — Live Data Connector      ║
║   Philip AJ Sogah  |  philipajsogah.io               ║
╚══════════════════════════════════════════════════════╝

Set environment variables before running:

  Alpaca (free):
    export ALPACA_API_KEY="your_key"
    export ALPACA_SECRET="your_secret"
    python market_data_connector.py

  Polygon ($29/mo):
    export POLYGON_API_KEY="your_key"
    PROVIDER=polygon python market_data_connector.py
""")

    provider = os.getenv("PROVIDER", "alpaca").lower()
    symbols  = os.getenv("SYMBOLS", "AAPL,SPY,TSLA").split(",")

    if provider == "polygon":
        key = os.getenv("POLYGON_API_KEY")
        if not key:
            raise SystemExit("Set POLYGON_API_KEY environment variable")
        run_polygon(key, symbols)
    else:
        key    = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET")
        if not key or not secret:
            raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET environment variables")
        run_alpaca(key, secret, symbols)
