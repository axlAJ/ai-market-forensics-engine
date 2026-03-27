"""
Market Forensics Engine — Philip AJ Sogah
Forensic Signal Fusion + Reflexive Momentum Decay (RMD) Trading Algorithm

Two-part system:
  1. ForensicSignalFusion  — detects manipulation patterns in market data
  2. RMDAlgorithm          — trades the *decay* after manipulation, not the spike
"""

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """A single point-in-time view of market microstructure."""
    timestamp: float
    price: float
    volume: float
    bid_depth: list[float]   # quantity at each bid level (best → worst)
    ask_depth: list[float]   # quantity at each ask level (best → worst)
    trade_count: int          # trades in this interval
    quote_cancels: int        # quotes cancelled in this interval
    quote_total: int          # total quotes in this interval


@dataclass
class ForensicResult:
    """Output of the Forensic Signal Fusion engine."""
    manipulation_score: float          # 0–100 composite score
    wash_trade_prob: float             # 0–100
    spoofing_confidence: float         # 0–100
    layering_index: float              # 0–100
    cross_market_divergence: float     # 0–100
    momentum_injection: float          # 0–100
    patterns_detected: list[str]
    verdict: str
    order_book_entropy: float          # Shannon entropy of order book


@dataclass
class RMDSignal:
    """Output of the Reflexive Momentum Decay algorithm."""
    trade: bool
    direction: str                     # 'SHORT' or 'LONG' (fade the spike)
    confidence: str                    # 'STRONG' | 'VALID' | 'NO SIGNAL'
    decay_confidence: float            # 0–100
    expected_snapback_pct: float       # expected price reversion %
    position_size_pct: float          # % of available capital to deploy
    entry_note: str
    t1_target_pct: float              # first exit target (60% of position)
    t2_target_pct: float              # full reversion target (40% of position)
    stop_loss_pct: float              # stop beyond manipulation extreme
    gate_open: bool
    gate_reason: str


# ─────────────────────────────────────────────
#  FORENSIC SIGNAL FUSION
# ─────────────────────────────────────────────

class ForensicSignalFusion:
    """
    Detects market manipulation by fusing five independent anomaly signals:
      - Wash trade probability    (weight 28%)
      - Spoofing confidence       (weight 26%)
      - Layering index            (weight 18%)
      - Cross-market divergence   (weight 16%)
      - Momentum injection        (weight 12%)

    Each signal is scored 0–100, then fused into a composite manipulation score.
    """

    WEIGHTS = {
        "wash":  0.28,
        "spoof": 0.26,
        "layer": 0.18,
        "cross": 0.16,
        "mom":   0.12,
    }

    THRESHOLDS = {
        "wash":  60,
        "spoof": 60,
        "layer": 60,
        "cross": 50,
        "mom":   55,
    }

    def __init__(self,
                 baseline_volume: float = 1000.0,
                 baseline_trade_rate: float = 10.0):
        """
        Args:
            baseline_volume:     typical volume per interval for this instrument
            baseline_trade_rate: typical number of trades per interval
        """
        self.baseline_volume = baseline_volume
        self.baseline_trade_rate = baseline_trade_rate
        self._price_history: list[float] = []

    # ── public API ────────────────────────────

    def analyse(self, snap: MarketSnapshot) -> ForensicResult:
        """Run all five forensic signals on a single market snapshot."""
        self._price_history.append(snap.price)

        vol_ratio     = snap.volume / max(self.baseline_volume, 1)
        obi           = self._order_book_imbalance(snap)
        trade_sigma   = self._trade_clustering_sigma(snap)
        cancel_rate   = snap.quote_cancels / max(snap.quote_total, 1) * 100
        cmd           = self._cross_market_divergence()
        entropy       = self._order_book_entropy(snap)

        wash   = self._wash_trade_score(vol_ratio, obi, trade_sigma)
        spoof  = self._spoofing_score(cancel_rate, obi, vol_ratio)
        layer  = self._layering_score(cancel_rate, trade_sigma, obi)
        cross  = self._cross_market_score(cmd, vol_ratio, trade_sigma)
        mom    = self._momentum_injection_score(vol_ratio, trade_sigma, cmd)

        composite = (
            wash  * self.WEIGHTS["wash"]  +
            spoof * self.WEIGHTS["spoof"] +
            layer * self.WEIGHTS["layer"] +
            cross * self.WEIGHTS["cross"] +
            mom   * self.WEIGHTS["mom"]
        )
        composite = min(100.0, round(composite, 1))

        patterns = self._flag_patterns(wash, spoof, layer, cross, mom)
        verdict  = self._verdict(composite)

        return ForensicResult(
            manipulation_score    = composite,
            wash_trade_prob       = round(wash, 1),
            spoofing_confidence   = round(spoof, 1),
            layering_index        = round(layer, 1),
            cross_market_divergence = round(cross, 1),
            momentum_injection    = round(mom, 1),
            patterns_detected     = patterns,
            verdict               = verdict,
            order_book_entropy    = round(entropy, 3),
        )

    # ── signal calculators ────────────────────

    def _wash_trade_score(self, vol_ratio: float, obi: float, sigma: float) -> float:
        return min(100.0, (vol_ratio / 10) * 40 + (obi / 100) * 30 + (sigma / 5) * 30)

    def _spoofing_score(self, cancel_rate: float, obi: float, vol_ratio: float) -> float:
        return min(100.0, (cancel_rate / 100) * 55 + (obi / 100) * 25 + (vol_ratio / 10) * 20)

    def _layering_score(self, cancel_rate: float, sigma: float, obi: float) -> float:
        return min(100.0, (cancel_rate / 100) * 40 + (sigma / 5) * 35 + (obi / 100) * 25)

    def _cross_market_score(self, cmd: float, vol_ratio: float, sigma: float) -> float:
        return min(100.0, (cmd / 100) * 60 + (vol_ratio / 10) * 25 + (sigma / 5) * 15)

    def _momentum_injection_score(self, vol_ratio: float, sigma: float, cmd: float) -> float:
        return min(100.0, (vol_ratio / 10) * 45 + (sigma / 5) * 30 + (cmd / 100) * 25)

    # ── microstructure helpers ─────────────────

    def _order_book_imbalance(self, snap: MarketSnapshot) -> float:
        """
        OBI = |bid_qty - ask_qty| / (bid_qty + ask_qty) × 100
        High imbalance suggests one-sided pressure — a spoofing hallmark.
        """
        bid_total = sum(snap.bid_depth)
        ask_total = sum(snap.ask_depth)
        total = bid_total + ask_total
        if total == 0:
            return 0.0
        return abs(bid_total - ask_total) / total * 100

    def _trade_clustering_sigma(self, snap: MarketSnapshot) -> float:
        """
        How many standard deviations above normal is the trade count?
        Clustered trades in a short window indicate coordinated activity.
        """
        if self.baseline_trade_rate == 0:
            return 0.0
        std_approx = math.sqrt(self.baseline_trade_rate)
        return min(5.0, (snap.trade_count - self.baseline_trade_rate) / max(std_approx, 0.1))

    def _cross_market_divergence(self) -> float:
        """
        Simulated cross-market divergence. In production, compare this instrument's
        return to a correlated benchmark. Returns 0–100.
        """
        if len(self._price_history) < 2:
            return 0.0
        raw_return = (self._price_history[-1] - self._price_history[-2]) / self._price_history[-2]
        return min(100.0, abs(raw_return) * 10000)

    def _order_book_entropy(self, snap: MarketSnapshot) -> float:
        """
        Shannon entropy of the order book depth distribution.
        High entropy = noisy / manufactured order flow.
        Low entropy  = clean, organic liquidity.
        """
        depths = snap.bid_depth + snap.ask_depth
        total = sum(depths)
        if total == 0:
            return 0.0
        probs = [d / total for d in depths if d > 0]
        return -sum(p * math.log2(p) for p in probs)

    # ── labelling ──────────────────────────────

    def _flag_patterns(self, wash, spoof, layer, cross, mom) -> list[str]:
        patterns = []
        if wash  >= self.THRESHOLDS["wash"]:  patterns.append("Wash trading")
        if spoof >= self.THRESHOLDS["spoof"]: patterns.append("Spoofing")
        if layer >= self.THRESHOLDS["layer"]: patterns.append("Layering")
        if cross >= self.THRESHOLDS["cross"]: patterns.append("Cross-market manipulation")
        if mom   >= self.THRESHOLDS["mom"]:   patterns.append("Momentum injection")
        return patterns

    def _verdict(self, score: float) -> str:
        if score > 75:  return "HIGH CONFIDENCE — manipulation detected, flag for review"
        if score > 50:  return "MODERATE ANOMALY — further pattern correlation advised"
        if score > 25:  return "LOW-LEVEL IRREGULARITY — monitor for escalation"
        return "CLEAN — no significant manipulation signal"


# ─────────────────────────────────────────────
#  REFLEXIVE MOMENTUM DECAY (RMD) ALGORITHM
# ─────────────────────────────────────────────

class RMDAlgorithm:
    """
    Reflexive Momentum Decay — trade the exhaust, not the spike.

    Core idea: once the ForensicSignalFusion engine detects institutional
    manipulation fingerprints, the manipulation itself becomes a predictive
    signal for a *mean-reversion move*. Most algos chase the spike.
    RMD waits for the snap-back.

    Gate conditions (BOTH must be satisfied):
      1. manipulation_score > manip_threshold
      2. order_book_entropy  > entropy_threshold

    Entry: on exhaustion candle (pace of manipulation slows, cancel rate drops)
    Exit:  tiered — 60% at T1, 40% at T2 (full reversion)
    Stop:  beyond manipulation extreme × stop_buffer
    """

    def __init__(self,
                 manip_threshold:   float = 55.0,
                 entropy_threshold: float = 2.0,
                 decay_half_life:   float = 3.5,
                 stop_buffer:       float = 1.35,
                 max_position_pct:  float = 8.0,
                 tod_risk_weight:   float = 0.20):
        """
        Args:
            manip_threshold:   minimum forensic score to open the gate
            entropy_threshold: minimum order book entropy to open the gate
            decay_half_life:   exponential decay rate (higher = faster snap-back)
            stop_buffer:       stop placed at spike_magnitude × stop_buffer beyond entry
            max_position_pct:  maximum position as % of available capital
            tod_risk_weight:   how much time-of-day risk discounts confidence
        """
        self.manip_threshold   = manip_threshold
        self.entropy_threshold = entropy_threshold
        self.decay_half_life   = decay_half_life
        self.stop_buffer       = stop_buffer
        self.max_position_pct  = max_position_pct
        self.tod_risk_weight   = tod_risk_weight

    def evaluate(self,
                 forensic: ForensicResult,
                 spike_magnitude_pct: float,
                 time_of_day_risk: float = 0.40) -> RMDSignal:
        """
        Evaluate whether to trade and compute trade parameters.

        Args:
            forensic:            output of ForensicSignalFusion.analyse()
            spike_magnitude_pct: % move of the suspected manipulation spike
            time_of_day_risk:    0–1, higher near open/close (thinner liquidity)

        Returns:
            RMDSignal with full trade plan or NO SIGNAL
        """
        gate_open, gate_reason = self._check_gate(forensic)

        if not gate_open:
            return RMDSignal(
                trade=False, direction="—", confidence="NO SIGNAL",
                decay_confidence=0, expected_snapback_pct=0,
                position_size_pct=0, entry_note=gate_reason,
                t1_target_pct=0, t2_target_pct=0, stop_loss_pct=0,
                gate_open=False, gate_reason=gate_reason,
            )

        decay_conf = self._decay_confidence(forensic, time_of_day_risk)
        snapback   = self._expected_snapback(spike_magnitude_pct, decay_conf)
        size       = self._position_size(decay_conf)
        direction  = "SHORT" if spike_magnitude_pct > 0 else "LONG"
        confidence = "STRONG" if forensic.manipulation_score > 75 else "VALID"

        t1   = snapback * 0.45
        t2   = snapback
        stop = abs(spike_magnitude_pct) * self.stop_buffer

        return RMDSignal(
            trade=True,
            direction=direction,
            confidence=confidence,
            decay_confidence=round(decay_conf, 1),
            expected_snapback_pct=round(snapback, 3),
            position_size_pct=round(size, 2),
            entry_note="Enter on exhaustion candle — wait for pace slow-down + cancel rate drop",
            t1_target_pct=round(t1, 3),
            t2_target_pct=round(t2, 3),
            stop_loss_pct=round(stop, 3),
            gate_open=True,
            gate_reason=gate_reason,
        )

    def decay_curve(self, spike_pct: float, decay_conf: float, steps: int = 80) -> list[float]:
        """
        Generate a simulated price path showing the manipulation spike and decay.
        Useful for visualisation and backtesting.

        Returns a list of price values (base = 100).
        """
        spike_at, entry_at = 20, 30
        prices, p = [], 100.0
        rate = self.decay_half_life * (decay_conf / 100)

        for i in range(steps):
            if i < spike_at:
                p += (random.random() - 0.5) * 0.3
            elif i == spike_at:
                p += spike_pct
            elif i < entry_at:
                p += spike_pct * 0.05 / (entry_at - spike_at)
            else:
                t = (i - entry_at) / (steps - entry_at)
                decay_val = spike_pct * (decay_conf / 100) * math.exp(-t * rate)
                p = 100 + decay_val + (random.random() - 0.5) * 0.12
            prices.append(round(p, 4))
        return prices

    # ── private ───────────────────────────────

    def _check_gate(self, forensic: ForensicResult) -> tuple[bool, str]:
        if forensic.manipulation_score < self.manip_threshold:
            return False, f"Gate closed — manipulation score {forensic.manipulation_score} < threshold {self.manip_threshold}"
        if forensic.order_book_entropy < self.entropy_threshold:
            return False, f"Gate closed — entropy {forensic.order_book_entropy:.2f} < threshold {self.entropy_threshold}"
        return True, "Both gates open — manipulation + entropy conditions met"

    def _decay_confidence(self, forensic: ForensicResult, tod_risk: float) -> float:
        entropy_score = min(100.0, forensic.order_book_entropy * 20)
        raw = (
            forensic.manipulation_score * 0.45 +
            entropy_score               * 0.35 +
            (1 - tod_risk) * 100        * self.tod_risk_weight
        )
        return min(100.0, raw)

    def _expected_snapback(self, spike_pct: float, decay_conf: float) -> float:
        return abs(spike_pct) * (decay_conf / 100) * 0.70

    def _position_size(self, decay_conf: float) -> float:
        return min(self.max_position_pct, decay_conf * (self.max_position_pct / 100))
