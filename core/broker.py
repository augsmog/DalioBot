"""
Alpaca Broker Integration
==========================
Handles order placement, position tracking, and account management.
Supports both paper trading and live trading modes.

Setup:
1. Create account at https://alpaca.markets
2. Enable options trading (apply for approval)
3. Set API keys in .env file:
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret
"""

import os
import datetime as dt
from typing import Optional

from loguru import logger

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        OrderType,
        OrderStatus,
        QueryOrderStatus,
    )
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed. pip install alpaca-py")


class AlpacaBroker:
    """Broker interface for Alpaca (paper and live trading)."""

    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = None
        self._connected = False

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            logger.warning(
                "Alpaca API keys not set. Create a .env file with:\n"
                "  ALPACA_API_KEY=your_key\n"
                "  ALPACA_SECRET_KEY=your_secret\n"
                "Get keys at https://alpaca.markets"
            )
            return

        if not ALPACA_AVAILABLE:
            logger.error("alpaca-py package not installed")
            return

        try:
            self.client = TradingClient(api_key, secret_key, paper=paper)
            account = self.client.get_account()
            self._connected = True
            mode = "PAPER" if paper else "LIVE"
            logger.info(f"Connected to Alpaca ({mode}). "
                        f"Equity: ${float(account.equity):,.2f}, "
                        f"Cash: ${float(account.cash):,.2f}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")

    @property
    def connected(self) -> bool:
        return self._connected

    def get_account(self) -> Optional[dict]:
        """Get account information."""
        if not self.connected:
            return None
        try:
            account = self.client.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "status": account.status,
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
            }
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return None

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        if not self.connected:
            return []
        try:
            positions = self.client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "side": p.side,
                    "market_value": float(p.market_value),
                    "cost_basis": float(p.cost_basis),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "current_price": float(p.current_price),
                    "avg_entry_price": float(p.avg_entry_price),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_open_orders(self) -> list[dict]:
        """Get all open/pending orders."""
        if not self.connected:
            return []
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self.client.get_orders(filter=request)
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "side": str(o.side),
                    "qty": str(o.qty),
                    "type": str(o.type),
                    "status": str(o.status),
                    "limit_price": str(o.limit_price) if o.limit_price else None,
                    "submitted_at": str(o.submitted_at),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    def place_stock_order(
        self,
        symbol: str,
        qty: int,
        side: str = "buy",
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "gtc",
    ) -> Optional[dict]:
        """Place a stock order (for covered call assignments etc.)."""
        if not self.connected:
            logger.error("Not connected to broker")
            return None

        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.GTC if time_in_force == "gtc" else TimeInForce.DAY

            if order_type == "limit" and limit_price:
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price,
                )
            else:
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                )

            order = self.client.submit_order(request)
            logger.info(f"Order placed: {side.upper()} {qty} {symbol} @ "
                        f"{'$'+str(limit_price) if limit_price else 'MARKET'}")

            return {
                "id": str(order.id),
                "symbol": order.symbol,
                "side": str(order.side),
                "qty": str(order.qty),
                "status": str(order.status),
            }
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None

    def cancel_all_orders(self) -> bool:
        """Emergency: cancel all open orders."""
        if not self.connected:
            return False
        try:
            self.client.cancel_orders()
            logger.warning("ALL ORDERS CANCELLED")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Emergency: liquidate all positions."""
        if not self.connected:
            return False
        try:
            self.client.close_all_positions(cancel_orders=True)
            logger.warning("ALL POSITIONS CLOSED")
            return True
        except Exception as e:
            logger.error(f"Failed to close positions: {e}")
            return False

    def emergency_shutdown(self) -> dict:
        """KILL SWITCH: Cancel everything and flatten."""
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        orders_cancelled = self.cancel_all_orders()
        positions_closed = self.close_all_positions()
        return {
            "orders_cancelled": orders_cancelled,
            "positions_closed": positions_closed,
            "timestamp": dt.datetime.now().isoformat(),
        }

    # ── Options Trading ─────────────────────────────────────────────────
    # NOTE: Alpaca's options API is newer and may require additional approval.
    # The methods below provide the interface — actual implementation depends
    # on Alpaca SDK version and account permissions.

    def place_options_order(self, order_params: dict) -> Optional[dict]:
        """Place an options order.

        For credit spreads and CSPs, we need to use Alpaca's options endpoints.
        This is a structured interface that can adapt to API changes.
        """
        if not self.connected:
            logger.error("Not connected to broker")
            return None

        logger.info(f"OPTIONS ORDER: {order_params}")

        # Log the intended trade for manual execution if API not available
        trade_instruction = {
            "action": order_params.get("action"),
            "ticker": order_params.get("ticker"),
            "strike": order_params.get("strike"),
            "expiry": order_params.get("expiry"),
            "contracts": order_params.get("contracts"),
            "limit_price": order_params.get("limit_price"),
            "order_type": "GTC",
            "timestamp": dt.datetime.now().isoformat(),
            "status": "INSTRUCTION_GENERATED",
            "note": "Execute via broker platform if API options trading not available",
        }

        return trade_instruction

    def get_options_positions(self) -> list[dict]:
        """Get open options positions."""
        # Placeholder — depends on Alpaca options API availability
        logger.info("Checking options positions...")
        return []
