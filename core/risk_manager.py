"""
Risk Management System
=======================
Two Sigma-inspired risk controls protecting capital from catastrophic losses.

Rules:
1. Never risk more than 30% of account on a single trade
2. Stop trading if daily loss exceeds 5%
3. Halt all trading if drawdown exceeds 20%
4. Position size using fractional Kelly Criterion
5. Kill switch for emergency shutdown
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class RiskState:
    """Current risk assessment."""
    can_trade: bool
    reason: str
    account_value: float
    peak_value: float
    drawdown_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    open_position_count: int
    total_exposure: float
    exposure_pct: float
    available_capital: float


class RiskManager:
    """Enforces risk limits and position sizing rules."""

    def __init__(self, config: dict):
        self.config = config.get("account", {})
        self.max_position_pct = self.config.get("max_position_pct", 0.30)
        self.max_daily_loss_pct = self.config.get("max_daily_loss_pct", 0.05)
        self.max_drawdown_pct = self.config.get("max_drawdown_pct", 0.20)

        self.peak_value: float = self.config.get("starting_capital", 1000)
        self.daily_start_value: Optional[float] = None
        self.kill_switch_active: bool = False
        self.trade_halt_until: Optional[dt.datetime] = None

    def check_risk(
        self,
        account_value: float,
        open_positions: list,
        current_exposures: dict,
    ) -> RiskState:
        """Run all risk checks and return current risk state."""
        # If account_value is provided and peak was never updated from a real value,
        # initialize peak to account_value (handles unfunded paper accounts)
        if account_value > 0 and self.peak_value > account_value * 2:
            self.peak_value = account_value
        # Update peak
        self.peak_value = max(self.peak_value, account_value)

        # Daily P&L tracking
        if self.daily_start_value is None:
            self.daily_start_value = account_value
        daily_pnl = account_value - self.daily_start_value
        daily_pnl_pct = (daily_pnl / self.daily_start_value) * 100 if self.daily_start_value > 0 else 0

        # Drawdown
        drawdown_pct = ((account_value - self.peak_value) / self.peak_value) * 100 if self.peak_value > 0 else 0

        # Total exposure
        total_exposure = sum(current_exposures.values()) if current_exposures else 0
        exposure_pct = (total_exposure / account_value) * 100 if account_value > 0 else 0

        # Available capital
        available = account_value - total_exposure

        # Determine if we can trade
        can_trade = True
        reason = "All clear"

        if self.kill_switch_active:
            can_trade = False
            reason = "KILL SWITCH ACTIVE — all trading halted"

        elif self.trade_halt_until and dt.datetime.now() < self.trade_halt_until:
            can_trade = False
            reason = f"Trading halted until {self.trade_halt_until.strftime('%Y-%m-%d %H:%M')}"

        elif drawdown_pct <= -self.max_drawdown_pct * 100:
            can_trade = False
            reason = f"MAX DRAWDOWN BREACHED: {drawdown_pct:.1f}% (limit: {-self.max_drawdown_pct*100:.0f}%)"
            logger.critical(reason)

        elif daily_pnl_pct <= -self.max_daily_loss_pct * 100:
            can_trade = False
            reason = f"DAILY LOSS LIMIT: {daily_pnl_pct:.1f}% (limit: {-self.max_daily_loss_pct*100:.0f}%)"
            logger.error(reason)

        elif available < 100:
            can_trade = False
            reason = f"Insufficient available capital: ${available:.2f}"

        state = RiskState(
            can_trade=can_trade,
            reason=reason,
            account_value=account_value,
            peak_value=self.peak_value,
            drawdown_pct=round(drawdown_pct, 2),
            daily_pnl=round(daily_pnl, 2),
            daily_pnl_pct=round(daily_pnl_pct, 2),
            open_position_count=len(open_positions),
            total_exposure=round(total_exposure, 2),
            exposure_pct=round(exposure_pct, 2),
            available_capital=round(available, 2),
        )

        if not can_trade:
            logger.warning(f"RISK BLOCK: {reason}")

        return state

    def calculate_position_size(
        self,
        account_value: float,
        win_rate: float = 0.70,
        avg_win: float = 1.0,
        avg_loss: float = 2.0,
        regime_mult: float = 1.0,
    ) -> float:
        """Calculate position size using fractional Kelly Criterion.

        Kelly % = W - [(1 - W) / R]
        where W = win probability, R = win/loss ratio

        We use half-Kelly for safety (fractional Kelly).
        """
        if avg_loss == 0:
            return account_value * 0.05  # Minimum safe size

        r = avg_win / avg_loss  # Win/loss ratio
        kelly_pct = win_rate - ((1 - win_rate) / r)

        # Half-Kelly for safety
        half_kelly = kelly_pct * 0.5

        # Clamp to max position percentage
        position_pct = max(0.02, min(half_kelly, self.max_position_pct))

        # Apply regime multiplier
        position_pct *= regime_mult

        # Never exceed max
        position_pct = min(position_pct, self.max_position_pct)

        position_size = account_value * position_pct
        logger.debug(f"Kelly: {kelly_pct:.2%}, Half-Kelly: {half_kelly:.2%}, "
                     f"Final: {position_pct:.2%} = ${position_size:.2f}")

        return round(position_size, 2)

    def activate_kill_switch(self):
        """Emergency: cancel all orders and prevent new ones."""
        self.kill_switch_active = True
        logger.critical("🚨 KILL SWITCH ACTIVATED — ALL TRADING HALTED")

    def deactivate_kill_switch(self):
        """Re-enable trading after manual review."""
        self.kill_switch_active = False
        logger.info("Kill switch deactivated. Trading resumed.")

    def reset_daily_tracking(self, account_value: float):
        """Call at start of each trading day."""
        self.daily_start_value = account_value
        logger.info(f"Daily tracking reset. Starting value: ${account_value:.2f}")

    def get_risk_dashboard(self, account_value: float, open_positions: list, exposures: dict) -> str:
        """Morning risk check dashboard."""
        state = self.check_risk(account_value, open_positions, exposures)

        dashboard = f"""
╔══════════════════════════════════════════════════════════╗
║                    RISK DASHBOARD                        ║
╠══════════════════════════════════════════════════════════╣
║  Status:           {'🟢 CAN TRADE' if state.can_trade else '🔴 HALTED':>20}     ║
║  Reason:           {state.reason[:35]:>35}     ║
╠══════════════════════════════════════════════════════════╣
║  Account Value:    ${state.account_value:>12,.2f}                    ║
║  Peak Value:       ${state.peak_value:>12,.2f}                    ║
║  Drawdown:         {state.drawdown_pct:>11.2f}%  (max: {-self.max_drawdown_pct*100:.0f}%)          ║
║  Daily P&L:        ${state.daily_pnl:>12,.2f}  ({state.daily_pnl_pct:>+.2f}%)        ║
╠══════════════════════════════════════════════════════════╣
║  Open Positions:   {state.open_position_count:>12}                    ║
║  Total Exposure:   ${state.total_exposure:>12,.2f}                    ║
║  Exposure %:       {state.exposure_pct:>11.1f}%                     ║
║  Available:        ${state.available_capital:>12,.2f}                    ║
╚══════════════════════════════════════════════════════════╝
"""
        return dashboard.strip()
