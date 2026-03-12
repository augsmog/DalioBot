"""
Macro Regime Detection System
==============================
Bridgewater-inspired regime classification using free data sources.

Regimes:
  risk_on    — Growth rising, inflation stable → full allocation
  risk_off   — Growth falling or inflation spiking → half allocation
  crisis     — VIX > 30, market below 200SMA, EMA clouds bearish → all cash
  recovery   — Post-crisis, bullish momentum returning → aggressive (LEAPs)
"""

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RegimeState:
    """Current market regime assessment."""
    regime: str                 # risk_on, risk_off, crisis, recovery
    confidence: float           # 0-1
    allocation_mult: float      # Multiplier for position sizing
    signals: dict               # Individual signal readings
    timestamp: str
    rationale: str              # Human-readable explanation


class MacroRegimeDetector:
    """Detects the current market regime from macro and technical signals."""

    REGIME_ALLOCATION = {
        "risk_on": 1.2,
        "risk_off": 0.5,
        "crisis": 0.0,
        "recovery": 1.5,
    }

    def __init__(self, data_pipeline):
        self.pipeline = data_pipeline
        self.regime_history: list[RegimeState] = []
        self._prev_regime: Optional[str] = None

    def detect_regime(self) -> RegimeState:
        """Run all regime signals and classify current environment."""
        signals = self._gather_signals()
        regime, confidence, rationale = self._classify(signals)

        state = RegimeState(
            regime=regime,
            confidence=confidence,
            allocation_mult=self.REGIME_ALLOCATION[regime],
            signals=signals,
            timestamp=dt.datetime.now().isoformat(),
            rationale=rationale,
        )

        # Track regime transitions
        if self._prev_regime and self._prev_regime != regime:
            logger.warning(f"REGIME CHANGE: {self._prev_regime} → {regime} | {rationale}")
        self._prev_regime = regime
        self.regime_history.append(state)

        return state

    def _gather_signals(self) -> dict:
        """Collect all macro and technical signals."""
        signals = {}

        # VIX
        vix = self.pipeline.get_vix()
        signals["vix"] = vix
        signals["vix_elevated"] = vix is not None and vix > 25
        signals["vix_crisis"] = vix is not None and vix > 30

        # SPY technicals
        spy = self.pipeline.get_price_history("SPY", period="1y")
        if not spy.empty:
            spy = self.pipeline.add_technicals(spy)
            latest = spy.iloc[-1]

            signals["spy_above_200sma"] = bool(latest["above_200sma"]) if not pd.isna(latest["above_200sma"]) else None
            signals["spy_above_50sma"] = float(latest["Close"]) > float(latest["sma_50"]) if not pd.isna(latest["sma_50"]) else None
            signals["spy_ema_cloud_bullish"] = bool(latest["ema_cloud_bullish"]) if not pd.isna(latest["ema_cloud_bullish"]) else None
            signals["spy_rsi"] = float(latest["rsi_14"]) if not pd.isna(latest["rsi_14"]) else None

            # Check if we're in a drawdown
            recent_high = spy["Close"].rolling(window=50).max().iloc[-1]
            current = float(latest["Close"])
            signals["spy_drawdown_pct"] = round(((current - recent_high) / recent_high) * 100, 1) if not pd.isna(recent_high) else 0

            # Trend: 5-day vs 20-day momentum
            if len(spy) >= 20:
                signals["spy_5d_return"] = float(spy["Close"].pct_change(5).iloc[-1]) * 100
                signals["spy_20d_return"] = float(spy["Close"].pct_change(20).iloc[-1]) * 100
            else:
                signals["spy_5d_return"] = 0
                signals["spy_20d_return"] = 0

            # Volume spike detection
            signals["spy_volume_ratio"] = float(latest["volume_ratio"]) if not pd.isna(latest["volume_ratio"]) else 1.0

        # Yield curve (from FRED)
        yield_curve = self.pipeline.get_yield_curve()
        signals["yield_curve_10y2y"] = yield_curve
        signals["yield_curve_inverted"] = yield_curve is not None and yield_curve < 0

        # Breadth proxy: compare QQQ vs IWM (tech vs small caps)
        qqq = self.pipeline.get_price_history("QQQ", period="3mo")
        iwm = self.pipeline.get_price_history("IWM", period="3mo")
        if not qqq.empty and not iwm.empty:
            qqq_ret = float(qqq["Close"].pct_change(20).iloc[-1]) if len(qqq) >= 20 else 0
            iwm_ret = float(iwm["Close"].pct_change(20).iloc[-1]) if len(iwm) >= 20 else 0
            signals["breadth_divergence"] = round(qqq_ret - iwm_ret, 3)
        else:
            signals["breadth_divergence"] = 0

        return signals

    def _classify(self, signals: dict) -> tuple[str, float, str]:
        """Classify the regime based on collected signals."""

        # ── Crisis Detection (highest priority) ────────────────────────
        crisis_score = 0
        crisis_reasons = []

        if signals.get("vix_crisis"):
            crisis_score += 3
            crisis_reasons.append(f"VIX at {signals['vix']:.1f} (>30)")

        if signals.get("spy_above_200sma") is False:
            crisis_score += 2
            crisis_reasons.append("SPY below 200 SMA")

        if signals.get("spy_ema_cloud_bullish") is False:
            crisis_score += 2
            crisis_reasons.append("EMA clouds bearish")

        drawdown = signals.get("spy_drawdown_pct", 0)
        if drawdown and drawdown < -10:
            crisis_score += 2
            crisis_reasons.append(f"SPY drawdown {drawdown:.1f}%")

        if crisis_score >= 5:
            return "crisis", min(crisis_score / 9, 1.0), "CRISIS: " + "; ".join(crisis_reasons)

        # ── Recovery Detection ─────────────────────────────────────────
        # Post-crisis: was recently in crisis, now improving
        if self._prev_regime == "crisis":
            recovery_signals = 0
            if signals.get("spy_ema_cloud_bullish"):
                recovery_signals += 1
            if signals.get("spy_5d_return", 0) > 2:
                recovery_signals += 1
            if not signals.get("vix_crisis"):
                recovery_signals += 1

            if recovery_signals >= 2:
                return "recovery", 0.7, "RECOVERY: Post-crisis momentum returning. EMA clouds resetting bullish."

        # ── Risk Off ───────────────────────────────────────────────────
        risk_off_score = 0
        risk_off_reasons = []

        if signals.get("vix_elevated"):
            risk_off_score += 1
            risk_off_reasons.append(f"VIX elevated at {signals.get('vix', 0):.1f}")

        if signals.get("yield_curve_inverted"):
            risk_off_score += 1
            risk_off_reasons.append("Yield curve inverted")

        if signals.get("spy_20d_return", 0) < -3:
            risk_off_score += 1
            risk_off_reasons.append(f"SPY 20d return negative ({signals.get('spy_20d_return', 0):.1f}%)")

        spy_rsi = signals.get("spy_rsi")
        if spy_rsi and spy_rsi > 70:
            risk_off_score += 1
            risk_off_reasons.append(f"SPY RSI overbought ({spy_rsi:.0f})")

        if risk_off_score >= 2:
            return "risk_off", min(risk_off_score / 4, 1.0), "RISK OFF: " + "; ".join(risk_off_reasons)

        # ── Default: Risk On ───────────────────────────────────────────
        risk_on_reasons = []
        if signals.get("spy_above_200sma"):
            risk_on_reasons.append("SPY above 200 SMA")
        if signals.get("spy_ema_cloud_bullish"):
            risk_on_reasons.append("EMA clouds bullish")
        if not signals.get("vix_elevated"):
            vix = signals.get('vix')
            risk_on_reasons.append(f"VIX calm ({vix:.1f})" if vix is not None else "VIX data unavailable")

        return "risk_on", 0.7, "RISK ON: " + "; ".join(risk_on_reasons)

    def get_regime_summary(self) -> str:
        """Human-readable regime summary for the morning check."""
        state = self.detect_regime()
        s = state.signals

        vix_val = s.get('vix')
        vix_str = f"{vix_val:.1f}" if vix_val is not None else "N/A"
        spy_rsi = s.get('spy_rsi', 0) or 0
        spy_dd = s.get('spy_drawdown_pct', 0) or 0
        yc_val = s.get('yield_curve_10y2y')
        yc_str = f"{yc_val:.2f}" if yc_val is not None else "N/A"

        summary = f"""
╔══════════════════════════════════════════════════════════╗
║  MACRO REGIME: {state.regime.upper():>10}  (confidence: {state.confidence:.0%})      ║
╠══════════════════════════════════════════════════════════╣
║  Allocation Multiplier: {state.allocation_mult:.1f}x                          ║
║  Rationale: {state.rationale[:45]:<45}║
╠══════════════════════════════════════════════════════════╣
║  VIX:          {vix_str:>8}  {'!! ELEVATED' if s.get('vix_elevated') else '   Normal':>16}  ║
║  SPY > 200SMA: {'Yes' if s.get('spy_above_200sma') else 'No':>8}  SPY RSI: {spy_rsi:>6.1f}        ║
║  EMA Clouds:   {'Bullish' if s.get('spy_ema_cloud_bullish') else 'BEARISH':>8}  Drawdown: {spy_dd:>5.1f}%     ║
║  Yield Curve:  {yc_str:>8}  {'!! INVERTED' if s.get('yield_curve_inverted') else '   Normal':>16}  ║
╚══════════════════════════════════════════════════════════╝
"""
        return summary.strip()
