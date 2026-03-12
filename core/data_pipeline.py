"""
DalioBot Data Pipeline - Free Market Data Ingestion
====================================================
Sources: Yahoo Finance (price/options), FRED (macro), calculated (technicals)
"""

import os
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


class MarketDataPipeline:
    """Fetches and processes market data from free sources."""

    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
        if FRED_AVAILABLE and self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
        else:
            self.fred = None
            logger.warning("FRED API not available. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")

    # ── Price Data ──────────────────────────────────────────────────────

    def get_price_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV price history from Yahoo Finance."""
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                logger.error(f"No data returned for {ticker}")
                return pd.DataFrame()
            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            logger.info(f"Fetched {len(data)} bars for {ticker}")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()

    def get_multiple_prices(self, tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
        """Fetch price history for multiple tickers."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.get_price_history(ticker, period=period)
        return results

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get the latest price for a ticker."""
        try:
            t = yf.Ticker(ticker)
            return t.info.get("regularMarketPrice") or t.info.get("currentPrice")
        except Exception as e:
            logger.error(f"Failed to get current price for {ticker}: {e}")
            return None

    # ── Options Data ────────────────────────────────────────────────────

    def get_options_chain(self, ticker: str, expiry_date: Optional[str] = None) -> Optional[dict]:
        """Fetch options chain from Yahoo Finance.

        Returns dict with 'calls' and 'puts' DataFrames.
        """
        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                logger.warning(f"No options available for {ticker}")
                return None

            if expiry_date and expiry_date in expirations:
                target_expiry = expiry_date
            else:
                # Find expiration closest to 30 DTE
                today = dt.date.today()
                target_dte = 30
                best_expiry = None
                best_diff = float("inf")
                for exp in expirations:
                    exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
                    diff = abs((exp_date - today).days - target_dte)
                    if diff < best_diff:
                        best_diff = diff
                        best_expiry = exp
                target_expiry = best_expiry

            chain = t.option_chain(target_expiry)
            logger.info(f"Fetched options chain for {ticker} exp {target_expiry}: "
                        f"{len(chain.calls)} calls, {len(chain.puts)} puts")
            return {
                "calls": chain.calls,
                "puts": chain.puts,
                "expiry": target_expiry,
                "ticker": ticker,
            }
        except Exception as e:
            logger.error(f"Failed to fetch options for {ticker}: {e}")
            return None

    def get_iv_rank(self, ticker: str, lookback_days: int = 252) -> Optional[float]:
        """Calculate IV Rank (current IV percentile over past year).

        IV Rank = (Current IV - 52w Low IV) / (52w High IV - 52w Low IV) * 100
        Uses historical volatility as proxy when real-time IV unavailable.
        """
        try:
            data = self.get_price_history(ticker, period="2y")
            if data.empty or len(data) < lookback_days:
                return None

            # Calculate 30-day realized volatility as IV proxy
            returns = data["Close"].pct_change().dropna()
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100

            current_vol = rolling_vol.iloc[-1]
            lookback_vol = rolling_vol.iloc[-lookback_days:]
            vol_min = lookback_vol.min()
            vol_max = lookback_vol.max()

            if vol_max == vol_min:
                return 50.0

            iv_rank = ((current_vol - vol_min) / (vol_max - vol_min)) * 100
            return round(float(iv_rank), 1)
        except Exception as e:
            logger.error(f"Failed to calculate IV rank for {ticker}: {e}")
            return None

    # ── Technical Indicators ────────────────────────────────────────────

    def add_technicals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators used by the flywheel strategy."""
        if df.empty:
            return df

        df = df.copy()

        # RSI (14-period)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        sma20 = df["Close"].rolling(window=20).mean()
        std20 = df["Close"].rolling(window=20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_pct"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # EMA Clouds (for crash detection)
        df["ema_8"] = df["Close"].ewm(span=8, adjust=False).mean()
        df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["ema_cloud_bullish"] = df["ema_8"] > df["ema_21"]

        # Moving Averages
        df["sma_50"] = df["Close"].rolling(window=50).mean()
        df["sma_200"] = df["Close"].rolling(window=200).mean()
        df["above_200sma"] = df["Close"] > df["sma_200"]

        # Volume ratio
        df["volume_sma_20"] = df["Volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]

        # Daily return
        df["daily_return"] = df["Close"].pct_change()

        # Average True Range (for volatility-adjusted sizing)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.rolling(window=14).mean()

        return df

    # ── Macro Data (FRED) ───────────────────────────────────────────────

    def get_vix(self) -> Optional[float]:
        """Get current VIX level from Yahoo Finance."""
        try:
            vix = yf.download("^VIX", period="5d", progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            return float(vix["Close"].iloc[-1]) if not vix.empty else None
        except Exception:
            return None

    def get_yield_curve(self) -> Optional[float]:
        """Get 10Y-2Y Treasury spread from FRED."""
        if not self.fred:
            return None
        try:
            spread = self.fred.get_series("T10Y2Y")
            return float(spread.dropna().iloc[-1])
        except Exception as e:
            logger.error(f"Failed to fetch yield curve: {e}")
            return None

    def get_fed_funds_rate(self) -> Optional[float]:
        """Get effective federal funds rate from FRED."""
        if not self.fred:
            return None
        try:
            rate = self.fred.get_series("FEDFUNDS")
            return float(rate.dropna().iloc[-1])
        except Exception as e:
            logger.error(f"Failed to fetch fed funds rate: {e}")
            return None

    def get_macro_dashboard(self) -> dict:
        """Get all macro indicators for regime detection."""
        dashboard = {
            "vix": self.get_vix(),
            "yield_curve_10y2y": self.get_yield_curve(),
            "fed_funds_rate": self.get_fed_funds_rate(),
            "timestamp": dt.datetime.now().isoformat(),
        }

        # SPY above 200 SMA check
        spy = self.get_price_history("SPY", period="1y")
        if not spy.empty:
            spy = self.add_technicals(spy)
            dashboard["spy_above_200sma"] = bool(spy["above_200sma"].iloc[-1])
            dashboard["spy_rsi"] = float(spy["rsi_14"].iloc[-1])

        return dashboard

    # ── Screening ───────────────────────────────────────────────────────

    def screen_for_puts(self, tickers: list[str]) -> list[dict]:
        """Screen tickers for put-selling opportunities.

        Returns sorted list of candidates (biggest red day first).
        """
        candidates = []
        vix = self.get_vix()

        if vix and vix > 25:
            logger.warning(f"VIX at {vix:.1f} — above 25 threshold. Consider sitting out.")

        for ticker in tickers:
            data = self.get_price_history(ticker, period="3mo")
            if data.empty:
                continue

            data = self.add_technicals(data)
            latest = data.iloc[-1]

            daily_change = float(latest["daily_return"]) * 100
            rsi = float(latest["rsi_14"]) if not pd.isna(latest["rsi_14"]) else None
            ema_bullish = bool(latest["ema_cloud_bullish"]) if not pd.isna(latest["ema_cloud_bullish"]) else None
            price = float(latest["Close"])

            # Apply filters
            is_red = daily_change < 0
            rsi_ok = rsi is not None and 30 <= rsi <= 60
            cloud_ok = ema_bullish is not False  # Allow if bullish or unknown

            iv_rank = self.get_iv_rank(ticker)

            candidates.append({
                "ticker": ticker,
                "price": price,
                "daily_change_pct": round(daily_change, 2),
                "rsi": round(rsi, 1) if rsi else None,
                "iv_rank": iv_rank,
                "ema_cloud_bullish": ema_bullish,
                "is_red": is_red,
                "rsi_in_range": rsi_ok,
                "cloud_ok": cloud_ok,
                "passes_all": is_red and rsi_ok and cloud_ok,
            })

        # Sort by daily change (most red first)
        candidates.sort(key=lambda x: x["daily_change_pct"])
        return candidates
