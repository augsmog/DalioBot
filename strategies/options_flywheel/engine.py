"""
Capital Flywheel Options Engine
================================
Stage 1: Credit Spreads ($500-$2,000)
Stage 2: Cash-Secured Puts ($2,000-$5,000)
Stage 3: Full Wheel — CSPs + Covered Calls ($5,000-$25,000)
Stage 4: Scaled Income + LEAPs ($25,000+)

Core philosophy: Be the casino, not the gambler.
Sell premium, let time decay work FOR you, close at 50% profit.
"""

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class AccountStage(Enum):
    CREDIT_SPREADS = 1      # $500-$2,000
    SMALL_CSPS = 2          # $2,000-$5,000
    FULL_FLYWHEEL = 3       # $5,000-$25,000
    SCALED_INCOME = 4       # $25,000+


@dataclass
class TradeSignal:
    """A signal to open or close an options position."""
    ticker: str
    action: str                 # "SELL_PUT", "SELL_CALL", "BUY_SPREAD", "CLOSE"
    strike: float
    expiry: str
    premium: float
    collateral_required: float
    contracts: int
    confidence: float           # 0-1
    reason: str
    # For spreads
    spread_long_strike: Optional[float] = None
    spread_width: Optional[float] = None
    max_loss: Optional[float] = None
    # Metadata
    rsi: Optional[float] = None
    iv_rank: Optional[float] = None
    daily_change: Optional[float] = None
    timestamp: str = field(default_factory=lambda: dt.datetime.now().isoformat())


@dataclass
class Position:
    """An open options position."""
    ticker: str
    position_type: str          # "SHORT_PUT", "SHORT_CALL", "CREDIT_SPREAD", "COVERED_CALL", "LEAP"
    strike: float
    expiry: str
    premium_collected: float
    contracts: int
    entry_date: str
    entry_price: float          # Underlying price at entry
    close_target: float         # 50% of premium (GTC buy-to-close price)
    # For spreads
    long_strike: Optional[float] = None
    max_loss: Optional[float] = None
    # State
    current_value: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "OPEN"        # OPEN, CLOSED, ASSIGNED, EXPIRED


class CapitalFlywheelEngine:
    """The core options selling engine implementing the Capital Flywheel strategy."""

    STAGE_THRESHOLDS = {
        AccountStage.CREDIT_SPREADS: 500,
        AccountStage.SMALL_CSPS: 2000,
        AccountStage.FULL_FLYWHEEL: 5000,
        AccountStage.SCALED_INCOME: 25000,
    }

    def __init__(self, config: dict):
        self.config = config
        self.positions: list[Position] = []
        self.closed_trades: list[Position] = []
        self.trade_log: list[dict] = []

    def get_account_stage(self, capital: float) -> AccountStage:
        """Determine which stage of the flywheel we're in."""
        if capital >= 25000:
            return AccountStage.SCALED_INCOME
        elif capital >= 5000:
            return AccountStage.FULL_FLYWHEEL
        elif capital >= 2000:
            return AccountStage.SMALL_CSPS
        else:
            return AccountStage.CREDIT_SPREADS

    def generate_signals(
        self,
        candidates: list[dict],
        capital: float,
        macro_regime: str = "risk_on",
    ) -> list[TradeSignal]:
        """Generate trade signals based on screened candidates and account stage.

        Args:
            candidates: Output from MarketDataPipeline.screen_for_puts()
            capital: Current account equity
            macro_regime: Current regime from macro detector
        """
        stage = self.get_account_stage(capital)
        signals = []

        # Don't trade in crisis regime
        if macro_regime == "crisis":
            logger.warning("CRISIS regime detected — no new positions. Cash is a position.")
            return []

        # Get regime allocation multiplier
        regime_mult = self.config.get("macro_regime", {}).get("regimes", {}).get(
            macro_regime, {}
        ).get("allocation_mult", 1.0)

        # Max position size (30% of account, adjusted by regime)
        max_position = capital * self.config["options_flywheel"].get("max_position_pct", 0.30) * regime_mult

        # Only trade candidates that pass all filters
        valid_candidates = [c for c in candidates if c.get("passes_all", False)]

        if not valid_candidates:
            logger.info("No candidates pass all filters today. No trade.")
            return []

        # Sort by confidence (highest first), then by biggest red day
        valid_candidates.sort(key=lambda c: (-c.get("confidence", 0.5), c.get("daily_change_pct", 0)))

        # Take the best candidate
        best = valid_candidates[0]

        # Scale position by candidate confidence
        candidate_confidence = best.get("confidence", 1.0)
        max_position *= candidate_confidence
        ticker = best["ticker"]
        price = best["price"]

        logger.info(f"Best candidate: {ticker} at ${price:.2f} "
                    f"(change: {best['daily_change_pct']:.1f}%, RSI: {best['rsi']})")

        if stage == AccountStage.CREDIT_SPREADS:
            signal = self._generate_credit_spread_signal(best, capital, max_position)
        elif stage == AccountStage.SMALL_CSPS:
            signal = self._generate_csp_signal(best, capital, max_position)
        else:
            signal = self._generate_csp_signal(best, capital, max_position)

        if signal:
            signals.append(signal)

        return signals

    def _generate_credit_spread_signal(
        self, candidate: dict, capital: float, max_position: float
    ) -> Optional[TradeSignal]:
        """Generate a bull put credit spread signal.

        Sell a put, buy a lower put for protection.
        Max risk = spread width - credit received.
        """
        ticker = candidate["ticker"]
        price = candidate["price"]
        target_credit_pct = self.config["options_flywheel"].get("target_credit_pct", 0.33)

        # Adapt spread width to stock price and account size
        if price < 15:
            spread_width = 1    # $1 wide for cheap stocks
        elif price < 50:
            spread_width = 2    # $2 wide for mid-price
        else:
            spread_width = min(5, max(1, int(max_position / 100)))

        # Short put strike: ~5-7% below current price
        short_strike = round(price * 0.94, 0)
        long_strike = short_strike - spread_width

        # Estimate credit (target ~33% of spread width)
        estimated_credit = spread_width * target_credit_pct * 100  # Per contract
        max_loss = (spread_width * 100) - estimated_credit

        # Position sizing: how many spreads can we afford?
        max_contracts = int(max_position / max_loss) if max_loss > 0 else 1
        contracts = max(1, min(max_contracts, 3))

        if max_loss * contracts > capital * 0.30:
            contracts = max(1, int((capital * 0.30) / max_loss))

        # Confidence score
        confidence = self._calculate_confidence(candidate)

        return TradeSignal(
            ticker=ticker,
            action="SELL_CREDIT_SPREAD",
            strike=short_strike,
            expiry=self._get_target_expiry(),
            premium=estimated_credit * contracts,
            collateral_required=max_loss * contracts,
            contracts=contracts,
            confidence=confidence,
            reason=f"Bull put spread on {ticker}: sell {short_strike}P / buy {long_strike}P. "
                   f"RSI={candidate['rsi']}, daily change={candidate['daily_change_pct']}%",
            spread_long_strike=long_strike,
            spread_width=spread_width,
            max_loss=max_loss * contracts,
            rsi=candidate["rsi"],
            iv_rank=candidate["iv_rank"],
            daily_change=candidate["daily_change_pct"],
        )

    def _generate_csp_signal(
        self, candidate: dict, capital: float, max_position: float
    ) -> Optional[TradeSignal]:
        """Generate a cash-secured put signal.

        Sell a put at a strike where premium is ~5% of collateral.
        """
        ticker = candidate["ticker"]
        price = candidate["price"]
        target_premium_pct = self.config["options_flywheel"].get("target_premium_pct", 0.05)

        # Strike ~7-10% below current price
        strike = round(price * 0.92, 0)
        collateral = strike * 100  # Per contract

        # Target 5% premium
        estimated_premium = strike * target_premium_pct * 100

        # Can we afford it?
        if collateral > max_position:
            logger.info(f"CSP on {ticker} requires ${collateral:.0f} collateral, "
                        f"exceeds max position ${max_position:.0f}. Consider credit spread instead.")
            return self._generate_credit_spread_signal(candidate, capital, max_position)

        contracts = max(1, int(max_position / collateral))

        confidence = self._calculate_confidence(candidate)

        return TradeSignal(
            ticker=ticker,
            action="SELL_PUT",
            strike=strike,
            expiry=self._get_target_expiry(),
            premium=estimated_premium * contracts,
            collateral_required=collateral * contracts,
            contracts=contracts,
            confidence=confidence,
            reason=f"Cash-secured put on {ticker} at ${strike}. "
                   f"Premium ~${estimated_premium:.0f}/contract. "
                   f"RSI={candidate['rsi']}, daily change={candidate['daily_change_pct']}%",
            rsi=candidate["rsi"],
            iv_rank=candidate["iv_rank"],
            daily_change=candidate["daily_change_pct"],
        )

    def _calculate_confidence(self, candidate: dict) -> float:
        """Score confidence 0-1 based on multiple factors."""
        score = 0.5  # Base

        rsi = candidate.get("rsi")
        if rsi:
            if 30 <= rsi <= 40:
                score += 0.2   # Ideal RSI zone
            elif 40 < rsi <= 50:
                score += 0.1
            elif rsi < 30:
                score -= 0.1   # Oversold, risky

        iv_rank = candidate.get("iv_rank")
        if iv_rank:
            if iv_rank > 50:
                score += 0.15  # High IV = fat premiums
            elif iv_rank > 30:
                score += 0.05

        daily_change = candidate.get("daily_change_pct", 0)
        if -5 < daily_change < -1:
            score += 0.1       # Decent red day without panic
        elif daily_change <= -5:
            score -= 0.1       # Too much panic

        if candidate.get("ema_cloud_bullish"):
            score += 0.1

        return round(min(1.0, max(0.0, score)), 2)

    def create_exit_order(self, position: Position) -> dict:
        """Create the GTC buy-to-close order at 50% profit."""
        close_price = position.premium_collected * 0.50 / position.contracts
        return {
            "ticker": position.ticker,
            "action": "BUY_TO_CLOSE",
            "strike": position.strike,
            "expiry": position.expiry,
            "limit_price": close_price / 100,  # Per-share price
            "contracts": position.contracts,
            "order_type": "GTC",
            "reason": f"Auto-close at 50% profit (target ${close_price:.2f})",
        }

    def check_assignments(self, positions: list[Position], current_prices: dict) -> list[dict]:
        """Check if any short puts are likely to be assigned (ITM near expiry)."""
        actions = []
        today = dt.date.today()

        for pos in positions:
            if pos.status != "OPEN" or pos.position_type != "SHORT_PUT":
                continue

            expiry = dt.datetime.strptime(pos.expiry, "%Y-%m-%d").date()
            dte = (expiry - today).days
            price = current_prices.get(pos.ticker)

            if price and price < pos.strike and dte <= 3:
                actions.append({
                    "position": pos,
                    "action": "PREPARE_FOR_ASSIGNMENT",
                    "reason": f"{pos.ticker} at ${price:.2f} below strike ${pos.strike}. "
                              f"Assignment likely in {dte} days. Prepare to sell covered calls.",
                })

        return actions

    def generate_covered_call_signal(
        self, ticker: str, shares: int, current_price: float, cost_basis: float
    ) -> Optional[TradeSignal]:
        """After assignment, sell covered calls on the shares."""
        # Sell call above cost basis to ensure profit if called away
        strike = round(max(current_price * 1.05, cost_basis * 1.02), 0)
        estimated_premium = current_price * 0.02 * 100  # ~2% for weekly

        contracts = shares // 100

        if contracts < 1:
            return None

        return TradeSignal(
            ticker=ticker,
            action="SELL_CALL",
            strike=strike,
            expiry=self._get_target_expiry(dte=7),  # Weekly covered calls
            premium=estimated_premium * contracts,
            collateral_required=0,  # Shares are the collateral
            contracts=contracts,
            confidence=0.7,
            reason=f"Covered call on {ticker} at ${strike}. "
                   f"Cost basis ${cost_basis:.2f}, current ${current_price:.2f}",
        )

    def _get_target_expiry(self, dte: int = 30) -> str:
        """Find the next expiration date ~DTE days out (Fridays)."""
        target = dt.date.today() + dt.timedelta(days=dte)
        # Options expire on Fridays
        while target.weekday() != 4:  # 4 = Friday
            target += dt.timedelta(days=1)
        return target.strftime("%Y-%m-%d")

    # ── Performance Tracking ────────────────────────────────────────────

    def log_trade(self, signal: TradeSignal, fill_price: float):
        """Log a completed trade for performance analysis."""
        self.trade_log.append({
            "timestamp": dt.datetime.now().isoformat(),
            "ticker": signal.ticker,
            "action": signal.action,
            "strike": signal.strike,
            "expiry": signal.expiry,
            "premium": signal.premium,
            "fill_price": fill_price,
            "contracts": signal.contracts,
            "confidence": signal.confidence,
        })

    def get_performance_summary(self) -> dict:
        """Calculate overall strategy performance."""
        if not self.closed_trades:
            return {"total_trades": 0, "message": "No closed trades yet"}

        wins = [t for t in self.closed_trades if (t.pnl or 0) > 0]
        losses = [t for t in self.closed_trades if (t.pnl or 0) <= 0]

        total_pnl = sum(t.pnl or 0 for t in self.closed_trades)
        total_premium = sum(t.premium_collected for t in self.closed_trades)

        return {
            "total_trades": len(self.closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.closed_trades) * 100,
            "total_pnl": round(total_pnl, 2),
            "total_premium_collected": round(total_premium, 2),
            "avg_pnl_per_trade": round(total_pnl / len(self.closed_trades), 2),
            "avg_days_held": None,  # Populated when we have date tracking
        }
