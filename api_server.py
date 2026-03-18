#!/usr/bin/env python3
"""
DalioBot API Server
====================
FastAPI server exposing trading commands as webhook-callable endpoints.
Designed to be triggered by n8n workflows on schedule.

Endpoints:
  POST /webhook/morning     — Run the 5-min morning routine
  POST /webhook/scan        — Scan all tickers for opportunities
  POST /webhook/train       — Run ML self-improvement loop
  POST /webhook/backtest    — Run backtests
  POST /webhook/dashboard   — Get full system status
  POST /webhook/kill        — Emergency shutdown
  GET  /health              — Health check

Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import datetime as dt
import hashlib
import hmac
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from loguru import logger

load_dotenv(Path(__file__).parent / ".env")

app = FastAPI(title="DalioBot Trading API", version="1.0.0")


async def self_ping():
    """Keep-alive: ping own health endpoint every 10 minutes to prevent Render cold starts."""
    import httpx
    render_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not render_url:
        return  # Not on Render, skip
    while True:
        await asyncio.sleep(600)  # 10 minutes
        try:
            async with httpx.AsyncClient() as client:
                await client.get(f"{render_url}/health", timeout=10)
        except Exception:
            pass


@app.on_event("startup")
async def startup():
    asyncio.create_task(self_ping())

WEBHOOK_SECRET = os.getenv("N8N_WEBHOOK_SECRET", "")


def load_config() -> dict:
    with open(Path(__file__).parent / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def is_market_hours() -> bool:
    """Check if US stock market is currently open (or within 30 min of open/close)."""
    from zoneinfo import ZoneInfo
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    # Weekday check (Mon=0, Fri=4)
    if now.weekday() > 4:
        return False
    # Market hours: 9:30 AM - 4:00 PM ET (with 30 min buffer)
    market_open = now.replace(hour=9, minute=0, second=0)
    market_close = now.replace(hour=16, minute=30, second=0)
    return market_open <= now <= market_close


def verify_webhook(request_body: bytes, signature: str) -> bool:
    """Verify webhook signature from n8n."""
    if not WEBHOOK_SECRET:
        return True  # No secret configured, allow all
    expected = hmac.new(WEBHOOK_SECRET.encode(), request_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": dt.datetime.now().isoformat(), "service": "daliobot"}


@app.post("/webhook/morning")
async def webhook_morning():
    """Run the 5-minute morning routine and return results as JSON."""
    if not is_market_hours():
        return {"timestamp": dt.datetime.now().isoformat(), "routine": "morning",
                "action": "MARKET_CLOSED", "message": "Outside market hours, skipping."}

    from core.data_pipeline import MarketDataPipeline
    from core.risk_manager import RiskManager
    from core.broker import AlpacaBroker
    from strategies.macro_regime.detector import MacroRegimeDetector
    from strategies.options_flywheel.engine import CapitalFlywheelEngine

    config = load_config()
    pipeline = MarketDataPipeline()
    risk_mgr = RiskManager(config)
    broker = AlpacaBroker(paper=config["broker"]["paper_trading"])
    regime_detector = MacroRegimeDetector(pipeline)
    flywheel = CapitalFlywheelEngine(config)

    result = {"timestamp": dt.datetime.now().isoformat(), "routine": "morning"}

    # Capital
    if broker.connected:
        account = broker.get_account()
        capital = account["equity"] if account and account["equity"] > 0 else config["account"]["starting_capital"]
        result["broker_connected"] = True
        result["account"] = account
    else:
        capital = config["account"]["starting_capital"]
        result["broker_connected"] = False

    result["capital"] = capital

    # Risk check
    positions = broker.get_positions() if broker.connected else []
    exposures = {p["symbol"]: p["market_value"] for p in positions}
    risk_state = risk_mgr.check_risk(capital, positions, exposures)
    result["risk"] = {
        "can_trade": risk_state.can_trade,
        "reason": risk_state.reason,
        "drawdown_pct": risk_state.drawdown_pct,
        "daily_pnl": risk_state.daily_pnl,
        "exposure_pct": risk_state.exposure_pct,
    }

    if not risk_state.can_trade:
        result["action"] = "HALTED"
        return result

    # Macro regime
    regime = regime_detector.detect_regime()
    result["regime"] = {
        "regime": regime.regime,
        "confidence": regime.confidence,
        "allocation_mult": regime.allocation_mult,
        "rationale": regime.rationale,
        "signals": {k: v for k, v in regime.signals.items() if v is not None},
    }

    if regime.regime == "crisis":
        result["action"] = "CRISIS_HALT"
        return result

    # Scan ALL tickers (use full universe with $100K account)
    stage = flywheel.get_account_stage(capital)
    all_tickers = (
        config["options_flywheel"]["small_account_tickers"] +
        config["options_flywheel"]["tickers"]
    )

    candidates = pipeline.screen_for_puts(all_tickers)

    # Filter out tickers we already hold positions in
    held_tickers = set()
    for p in positions:
        for t in all_tickers:
            if p["symbol"].startswith(t):
                held_tickers.add(t)
    for c in candidates:
        if c["ticker"] in held_tickers:
            c["passes_all"] = False

    result["candidates"] = candidates
    result["stage"] = stage.name
    result["held_tickers"] = list(held_tickers)

    # Generate signals
    signals = flywheel.generate_signals(candidates, capital, regime.regime)
    result["signals"] = []

    for signal in signals:
        sig_dict = {
            "ticker": signal.ticker,
            "action": signal.action,
            "strike": signal.strike,
            "expiry": signal.expiry,
            "premium": signal.premium,
            "max_loss": signal.max_loss,
            "contracts": signal.contracts,
            "confidence": signal.confidence,
            "reason": signal.reason,
        }
        result["signals"].append(sig_dict)

        # Execute if broker connected
        if broker.connected:
            limit_price = signal.premium / signal.contracts / 100 if signal.contracts else 0.50
            order = broker.place_options_order({
                "action": signal.action,
                "ticker": signal.ticker,
                "strike": signal.strike,
                "expiry": signal.expiry,
                "contracts": signal.contracts,
                "spread_width": signal.spread_width or 2,
                "limit_price": round(limit_price, 2),
            })
            sig_dict["order"] = order

    result["action"] = "SIGNALS_GENERATED" if signals else "NO_SIGNALS"
    return result


@app.post("/webhook/scan")
async def webhook_scan():
    """Scan all tickers and return opportunities."""
    from core.data_pipeline import MarketDataPipeline

    config = load_config()
    pipeline = MarketDataPipeline()

    all_tickers = (
        config["options_flywheel"]["tickers"] +
        config["options_flywheel"]["small_account_tickers"]
    )

    candidates = pipeline.screen_for_puts(all_tickers)

    return {
        "timestamp": dt.datetime.now().isoformat(),
        "routine": "scan",
        "candidates": candidates,
        "actionable": [c for c in candidates if c.get("passes_all")],
        "total_scanned": len(all_tickers),
    }


@app.post("/webhook/train")
async def webhook_train():
    """Run ML self-improvement loop."""
    from core.data_pipeline import MarketDataPipeline
    from strategies.ml_signals.signal_engine import MLSignalEngine

    config = load_config()
    pipeline = MarketDataPipeline()
    ml = MLSignalEngine(config)

    tickers = config["options_flywheel"]["small_account_tickers"][:3]
    results = {"timestamp": dt.datetime.now().isoformat(), "routine": "train", "models": []}

    for ticker in tickers:
        data = pipeline.get_price_history(ticker, period="2y")
        if data.empty:
            continue

        data = pipeline.add_technicals(data)

        # Initial train
        metrics = ml.train(ticker, data)

        # Run improvement loop
        improvement = ml.run_improvement_loop(ticker, data, n_experiments=10)

        # Get today's prediction
        pred = ml.predict_entry_quality(data)

        model_result = {
            "ticker": ticker,
            "base_accuracy": metrics.get("avg_accuracy"),
            "base_precision": metrics.get("avg_precision"),
            "improvements_found": improvement.get("improvements_found", 0),
            "best_holdout_score": improvement.get("best_holdout_score"),
            "todays_prediction": pred,
            "feature_importance": dict(list(ml.get_feature_importance().items())[:5]) if ml.get_feature_importance() else None,
        }
        results["models"].append(model_result)

    return results


@app.post("/webhook/backtest")
async def webhook_backtest():
    """Run backtests on all tickers."""
    from core.data_pipeline import MarketDataPipeline
    from backtest.backtester import FlywheelBacktester

    config = load_config()
    pipeline = MarketDataPipeline()
    capital = config["account"]["starting_capital"]

    all_tickers = config["options_flywheel"]["small_account_tickers"][:4]
    results = {"timestamp": dt.datetime.now().isoformat(), "routine": "backtest", "results": []}

    for ticker in all_tickers:
        data = pipeline.get_price_history(ticker, period="2y")
        if data.empty:
            continue

        bt = FlywheelBacktester(starting_capital=capital, config=config)
        result = bt.run(data, ticker, "credit_spread")
        mc = bt.monte_carlo(result)

        results["results"].append({
            "ticker": ticker,
            "total_return_pct": result.total_return_pct,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "monte_carlo": mc if "error" not in mc else None,
        })

    return results


@app.post("/webhook/dashboard")
async def webhook_dashboard():
    """Full system status dashboard."""
    from core.data_pipeline import MarketDataPipeline
    from core.risk_manager import RiskManager
    from core.broker import AlpacaBroker
    from strategies.macro_regime.detector import MacroRegimeDetector

    config = load_config()
    pipeline = MarketDataPipeline()
    risk_mgr = RiskManager(config)
    broker = AlpacaBroker(paper=config["broker"]["paper_trading"])
    regime_detector = MacroRegimeDetector(pipeline)

    capital = config["account"]["starting_capital"]
    account = None
    if broker.connected:
        account = broker.get_account()
        if account and account["equity"] > 0:
            capital = account["equity"]

    positions = broker.get_positions() if broker.connected else []
    exposures = {p["symbol"]: p["market_value"] for p in positions}
    risk_state = risk_mgr.check_risk(capital, positions, exposures)
    regime = regime_detector.detect_regime()

    return {
        "timestamp": dt.datetime.now().isoformat(),
        "routine": "dashboard",
        "capital": capital,
        "broker_connected": broker.connected,
        "account": account,
        "positions": positions,
        "risk": {
            "can_trade": risk_state.can_trade,
            "reason": risk_state.reason,
            "drawdown_pct": risk_state.drawdown_pct,
            "daily_pnl": risk_state.daily_pnl,
        },
        "regime": {
            "regime": regime.regime,
            "confidence": regime.confidence,
            "rationale": regime.rationale,
        },
    }


@app.post("/webhook/kill")
async def webhook_kill():
    """Emergency shutdown."""
    from core.broker import AlpacaBroker
    from core.risk_manager import RiskManager

    config = load_config()
    broker = AlpacaBroker(paper=config["broker"]["paper_trading"])
    risk_mgr = RiskManager(config)

    risk_mgr.activate_kill_switch()
    shutdown_result = None
    if broker.connected:
        shutdown_result = broker.emergency_shutdown()

    return {
        "timestamp": dt.datetime.now().isoformat(),
        "routine": "kill",
        "kill_switch": "ACTIVATED",
        "broker_shutdown": shutdown_result,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
