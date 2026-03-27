# AI Market Forensics Engine
### Reflexive Momentum Decay (RMD) Algorithm

> *Detecting market manipulation in real time and trading the correction — not the spike.*

**Philip AJ Sogah** | [philipajsogah.io](https://philipajsogah.io) | philipaxl7@gmail.com

---

## Overview

The AI Market Forensics Engine is an independent research project combining machine learning-based manipulation detection with a novel mean-reversion trading strategy. The system connects to live US equities market data via Alpaca Markets WebSocket and runs two algorithms in real time.

Most trading algorithms chase momentum. This one waits.

When institutional manipulation is detected, the **Reflexive Momentum Decay (RMD)** algorithm positions into the correction — trading the exhaust phase of manipulation rather than the spike itself.

---

## Two-Part Architecture

### Algorithm 1 — Forensic Signal Fusion
Detects market manipulation by fusing five independent anomaly signals into a single 0–100 manipulation score:

| Signal | Weight | What it detects |
|---|---|---|
| Wash trade probability | 28% | Coordinated self-trading to inflate volume |
| Spoofing confidence | 26% | Large orders placed and immediately cancelled |
| Layering index | 18% | Multiple fake orders stacked to move price |
| Cross-market divergence | 16% | Price disconnection from correlated instruments |
| Momentum injection | 12% | Artificial price velocity |

Shannon entropy on order book depth validates signal quality — differentiating genuine manipulation from organic volatility.

### Algorithm 2 — Reflexive Momentum Decay (RMD)
A novel mean-reversion strategy built on a simple insight: once manipulation fingerprints are confirmed, the manipulation event itself becomes a predictive signal for a counter-move.

**How the trade works:**
1. Forensic engine detects manipulation (score > threshold)
2. Order book entropy confirms signal quality (dual-gate entry)
3. Exponential decay curve fits the expected snap-back magnitude
4. Position opens on exhaustion candle
5. Tiered exit: 60% closed at T1, 40% at full reversion T2
6. Stop placed beyond manipulation extreme

---

## Live Data Connection

Connects to **Alpaca Markets** WebSocket for real-time US equities data (free paper trading account).

```
DataFeed (Alpaca WebSocket)
    → Live Order Book (per symbol)
    → Rolling Baseline (20-bar median volume + trade rate)
    → ForensicSignalFusion (every 30 seconds)
    → RMDAlgorithm (dual-gate check)
    → PaperTradeLog (signal tracking + win rate)
```

---

## Test Results

```
============================================================
  MARKET FORENSICS ENGINE — TEST SUITE
  Philip AJ Sogah  |  philipajsogah.io
============================================================

── ForensicSignalFusion: clean market ──
  ✓ PASS  Manipulation score < 40 on clean data   [score=3.4]
  ✓ PASS  No patterns detected on clean data       [patterns=[]]
  ✓ PASS  Verdict is CLEAN
  ✓ PASS  Entropy is a non-negative float          [entropy=3.114]

── ForensicSignalFusion: full manipulation ──
  ✓ PASS  Manipulation score > 70                  [score=80.8]
  ✓ PASS  At least two patterns detected           [patterns=['Wash trading', 'Spoofing', 'Layering']]
  ✓ PASS  Verdict is HIGH CONFIDENCE
  ✓ PASS  Wash trade score > 50                    [wash=93.0]
  ✓ PASS  Spoofing score > 50                      [spoof=90.7]

── RMD: gate open — full manipulation ──
  ✓ PASS  Gate open on manipulation snapshot
  ✓ PASS  trade=True when gate open
  ✓ PASS  Direction is SHORT for positive spike
  ✓ PASS  T1 < T2 (tiered exits in right order)
  ✓ PASS  Stop loss > T2 (risk:reward favourable)

── Integration: full pipeline ──
  ✓ PASS  Manipulation candle score > clean candle [clean=3.4, manip=93.4]
  ✓ PASS  RMD signal trade=True                    [confidence=STRONG]
  ✓ PASS  Decay curve generated with 80 data points

  RESULTS: 38/38 passed  |  0 failed
  ALL TESTS PASSED ✓
============================================================
```

---

## Project Files

| File | Description |
|---|---|
| `forensics_engine.py` | Core algorithm — ForensicSignalFusion + RMDAlgorithm |
| `test_forensics.py` | Full test suite — 38 unit and integration tests |
| `market_data_connector.py` | Live Alpaca WebSocket connector + paper trade logger |

---

## Setup & Usage

### 1. Install dependency
```bash
pip3 install websockets
```

### 2. Run the test suite first
```bash
python3 test_forensics.py
```
Expected: `38/38 passed — ALL TESTS PASSED ✓`

### 3. Get free Alpaca API keys
Sign up at [alpaca.markets](https://alpaca.markets) — paper trading is free, no brokerage account required.

### 4. Connect to live data
```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET="your_secret"
export SYMBOLS="AAPL,TSLA,SPY"

python3 market_data_connector.py
```

### 5. Live output example
```
02:37:50  INFO  Starting LiveForensicsRunner — symbols: ['AAPL', 'SPY']
02:37:50  INFO  Alpaca: connected
02:37:50  INFO  Alpaca: authenticated
02:37:50  INFO  Connected and subscribed. Waiting for data...

AAPL   score=12.4   entropy=3.211   patterns=none
SPY    score=8.1    entropy=3.089   patterns=none

TSLA   score=74.2   entropy=1.840   patterns=['Spoofing', 'Layering']
  ➜ RMD SIGNAL  SHORT TSLA | snap=0.821% | size=4.2% | conf=STRONG
📊 PAPER TRADE OPENED  SHORT TSLA @ 248.32 | T1=247.87 T2=246.24 Stop=251.67
```

---

## Technical Stack

- **Language:** Python 3.11
- **Data:** Alpaca Markets WebSocket (real-time US equities)
- **Key concepts:** Shannon entropy, order book microstructure, exponential decay modeling, weighted signal fusion, mean reversion
- **Testing:** 38 unit + integration tests, zero dependencies beyond standard library + websockets

---

## Research Background

This project builds on established market microstructure research:

- **Wash trading detection** — coordinated volume analysis against rolling baselines
- **Spoofing detection** — quote cancellation rate analysis (the method used to identify Navinder Singh Sarao in the 2010 Flash Crash investigation)
- **Shannon entropy** — information-theoretic measure of order book complexity, used to distinguish organic from manufactured order flow
- **Mean reversion** — the RMD algorithm is a novel application of momentum decay modeling to manipulation-confirmed setups

---

## Portfolio

Live interactive demo with real-time algorithm execution:
**[philipajsogah.io](https://philipajsogah.io)**

---

## About

**Philip AJ Sogah** is an AI Innovator, Project Manager, and Software Engineer based in Northfield, VT. Currently completing a BS in Computer Science at Norwich University with research focus on AI-based detection systems.

- 🌐 [philipajsogah.io](https://philipajsogah.io)
- 📧 philipaxl7@gmail.com
- 📞 +1 802-431-8215

---

*This project is for research and educational purposes. Paper trade mode is enabled by default — no real money is used.*
