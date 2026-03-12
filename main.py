#!/usr/bin/env python3
"""
DalioBot Trading Powerhouse
=============================
Main orchestrator that ties together all systems:
- Data Pipeline (free market data)
- Capital Flywheel (options selling engine)
- Macro Regime Detection (Bridgewater-inspired)
- ML Signal Enhancement (Karpathy self-improvement loop)
- Risk Management (Two Sigma-inspired)
- Broker Integration (Alpaca)
- Backtesting Engine (Renaissance-inspired)

Usage:
  python main.py morning      # Run 5-minute morning routine
  python main.py scan         # Scan for opportunities
  python main.py backtest     # Run strategy backtest
  python main.py train        # Train/improve ML models
  python main.py dashboard    # Show full dashboard
  python main.py paper        # Start paper trading mode
  python main.py kill         # Emergency shutdown
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

# File logging only when not on Render (ephemeral filesystem)
if not os.getenv("RENDER"):
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/daliobot_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days", level="DEBUG")


def load_config() -> dict:
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def cmd_morning(config: dict):
    """The 5-minute morning routine — the heart of the system."""
    from core.data_pipeline import MarketDataPipeline
    from core.risk_manager import RiskManager
    from core.broker import AlpacaBroker
    from strategies.macro_regime.detector import MacroRegimeDetector
    from strategies.options_flywheel.engine import CapitalFlywheelEngine

    print("\n" + "=" * 60)
    print("  DALIOBOT MORNING ROUTINE — 5 MINUTES TO MARKET MASTERY")
    print("=" * 60)

    # Initialize systems
    pipeline = MarketDataPipeline()
    risk_mgr = RiskManager(config)
    broker = AlpacaBroker(paper=config["broker"]["paper_trading"])
    regime_detector = MacroRegimeDetector(pipeline)
    flywheel = CapitalFlywheelEngine(config)

    # ── Minute 1: Safety Check ─────────────────────────────────────
    print("\n⏱  MINUTE 1: Safety Check")
    print("-" * 40)

    # Get account value
    if broker.connected:
        account = broker.get_account()
        capital = account["equity"] if account and account["equity"] > 0 else config["account"]["starting_capital"]
        if account and account["equity"] == 0:
            print(f"  [Paper account unfunded. Using config capital: ${capital:,.2f}]")
            print(f"  Fund paper account at https://app.alpaca.markets or use Alpaca's reset endpoint")
    else:
        capital = config["account"]["starting_capital"]
        print(f"  [Broker not connected. Using config capital: ${capital:,.2f}]")

    # Risk check
    positions = broker.get_positions() if broker.connected else []
    exposures = {p["symbol"]: p["market_value"] for p in positions}
    risk_state = risk_mgr.check_risk(capital, positions, exposures)
    print(risk_mgr.get_risk_dashboard(capital, positions, exposures))

    if not risk_state.can_trade:
        print(f"\n  🛑 TRADING HALTED: {risk_state.reason}")
        print("  Close your laptop. No trades today.")
        return

    # ── Minute 2: Macro Regime ─────────────────────────────────────
    print("\n⏱  MINUTE 2: Macro Regime Check")
    print("-" * 40)

    regime = regime_detector.detect_regime()
    print(regime_detector.get_regime_summary())

    if regime.regime == "crisis":
        print("\n  🛑 CRISIS REGIME. Cash is a position. No trades today.")
        return

    # ── Minute 3: Scan Tickers ─────────────────────────────────────
    print("\n⏱  MINUTE 3: Scanning Tickers")
    print("-" * 40)

    # Choose tickers based on account stage
    stage = flywheel.get_account_stage(capital)
    if stage.value <= 2:
        tickers = config["options_flywheel"]["small_account_tickers"]
        print(f"  Stage {stage.value} ({stage.name}): Using small account tickers")
    else:
        tickers = config["options_flywheel"]["tickers"]
        print(f"  Stage {stage.value} ({stage.name}): Using leveraged ETF tickers")

    candidates = pipeline.screen_for_puts(tickers)

    print(f"\n  {'Ticker':<8} {'Price':>8} {'Change':>8} {'RSI':>6} {'IV Rank':>8} {'Passes':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*8} {'─'*8}")
    for c in candidates:
        passes = "✓" if c["passes_all"] else "✗"
        print(f"  {c['ticker']:<8} ${c['price']:>7.2f} {c['daily_change_pct']:>+7.2f}% "
              f"{c['rsi'] or 'N/A':>6} {c['iv_rank'] or 'N/A':>8} {passes:>8}")

    # ── Minute 4: Generate Signal ──────────────────────────────────
    print("\n⏱  MINUTE 4: Signal Generation")
    print("-" * 40)

    signals = flywheel.generate_signals(candidates, capital, regime.regime)

    if not signals:
        print("  No valid signals today. Check back tomorrow.")
        return

    for signal in signals:
        print(f"\n  📊 SIGNAL: {signal.action}")
        print(f"     Ticker:     {signal.ticker}")
        print(f"     Strike:     ${signal.strike:.0f}")
        print(f"     Expiry:     {signal.expiry}")
        print(f"     Premium:    ${signal.premium:.2f}")
        print(f"     Max Loss:   ${signal.max_loss:.2f}" if signal.max_loss else "")
        print(f"     Contracts:  {signal.contracts}")
        print(f"     Confidence: {signal.confidence:.0%}")
        print(f"     Reason:     {signal.reason}")

    # ── Minute 5: Execute (if connected) ───────────────────────────
    print("\n⏱  MINUTE 5: Execution")
    print("-" * 40)

    if broker.connected:
        for signal in signals:
            order = broker.place_options_order({
                "action": signal.action,
                "ticker": signal.ticker,
                "strike": signal.strike,
                "expiry": signal.expiry,
                "contracts": signal.contracts,
                "limit_price": signal.premium / signal.contracts / 100,
            })
            if order:
                print(f"  ✓ Order submitted: {order}")
                # Set the 50% profit GTC exit
                print(f"  ✓ GTC exit set at 50% profit")
    else:
        print("  [Paper mode / Not connected]")
        print("  Execute these trades manually in your broker:")
        for signal in signals:
            print(f"  → {signal.action}: {signal.ticker} {signal.strike} {signal.expiry}")

    print("\n" + "=" * 60)
    print("  MORNING ROUTINE COMPLETE. Close your laptop.")
    print("=" * 60 + "\n")


def cmd_scan(config: dict):
    """Quick scan of all tickers for opportunities."""
    from core.data_pipeline import MarketDataPipeline

    pipeline = MarketDataPipeline()

    all_tickers = (
        config["options_flywheel"]["tickers"] +
        config["options_flywheel"]["small_account_tickers"]
    )

    print("\n📡 Scanning all tickers...\n")
    candidates = pipeline.screen_for_puts(all_tickers)

    print(f"{'Ticker':<8} {'Price':>8} {'Change':>8} {'RSI':>6} {'IV Rank':>8} {'Cloud':>8} {'Signal':>8}")
    print(f"{'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")

    for c in candidates:
        signal = "TRADE" if c["passes_all"] else "WAIT"
        cloud = "Bull" if c.get("ema_cloud_bullish") else "Bear"
        print(f"{c['ticker']:<8} ${c['price']:>7.2f} {c['daily_change_pct']:>+7.2f}% "
              f"{c['rsi'] or 'N/A':>6} {c['iv_rank'] or 'N/A':>8} {cloud:>8} {signal:>8}")


def cmd_backtest(config: dict):
    """Run backtests on the flywheel strategy."""
    from core.data_pipeline import MarketDataPipeline
    from backtest.backtester import FlywheelBacktester

    pipeline = MarketDataPipeline()
    capital = config["account"]["starting_capital"]

    tickers_to_test = config["options_flywheel"]["small_account_tickers"][:3]
    strategy = "credit_spread"

    print(f"\n🔬 Running backtest: {strategy} strategy, ${capital} capital\n")

    for ticker in tickers_to_test:
        print(f"\n{'='*60}")
        print(f"  Backtesting {ticker}...")
        print(f"{'='*60}")

        data = pipeline.get_price_history(ticker, period="2y")
        if data.empty:
            print(f"  No data for {ticker}, skipping.")
            continue

        bt = FlywheelBacktester(starting_capital=capital, config=config)
        result = bt.run(data, ticker, strategy)
        bt.print_results(result)

        # Monte Carlo
        mc = bt.monte_carlo(result)
        if "error" not in mc:
            print(f"  Monte Carlo ({mc['simulations']} simulations):")
            print(f"    Median outcome:  ${mc['median_final_equity']:,.2f}")
            print(f"    95th percentile: ${mc['p95_final_equity']:,.2f}")
            print(f"    5th percentile:  ${mc['p5_final_equity']:,.2f}")
            print(f"    P(profit):       {mc['prob_profit']:.1f}%")
            print(f"    P(2x):           {mc['prob_double']:.1f}%")


def cmd_train(config: dict):
    """Train or improve ML signal models."""
    from core.data_pipeline import MarketDataPipeline
    from strategies.ml_signals.signal_engine import MLSignalEngine

    pipeline = MarketDataPipeline()
    ml = MLSignalEngine(config)

    tickers = config["options_flywheel"]["small_account_tickers"][:3]

    print("\n🧠 ML Signal Training — Karpathy Self-Improvement Loop\n")

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"  Training on {ticker}...")
        print(f"{'='*50}")

        data = pipeline.get_price_history(ticker, period="2y")
        if data.empty:
            print(f"  No data for {ticker}")
            continue

        data = pipeline.add_technicals(data)

        # Initial training
        metrics = ml.train(ticker, data)
        if "error" in metrics:
            print(f"  Error: {metrics['error']}")
            continue

        print(f"  Base model: accuracy={metrics['avg_accuracy']:.3f}, "
              f"precision={metrics['avg_precision']:.3f}")

        # Run improvement loop
        print(f"\n  Running 10 improvement experiments...")
        improvement = ml.run_improvement_loop(ticker, data, n_experiments=10)

        if "error" not in improvement:
            print(f"  Improvements found: {improvement['improvements_found']}/10")
            print(f"  Best holdout score: {improvement['best_holdout_score']:.4f}")

            # Test current prediction
            pred = ml.predict_entry_quality(data)
            if pred:
                print(f"\n  Today's prediction: {pred['signal']} "
                      f"({pred['confidence']:.1%} confidence) — {pred['strength']}")

        # Feature importance
        fi = ml.get_feature_importance()
        if fi:
            print(f"\n  Top 5 features:")
            for i, (feat, imp) in enumerate(list(fi.items())[:5]):
                print(f"    {i+1}. {feat}: {imp:.0f}")


def cmd_dashboard(config: dict):
    """Full system dashboard."""
    from core.data_pipeline import MarketDataPipeline
    from core.risk_manager import RiskManager
    from core.broker import AlpacaBroker
    from strategies.macro_regime.detector import MacroRegimeDetector

    pipeline = MarketDataPipeline()
    risk_mgr = RiskManager(config)
    broker = AlpacaBroker(paper=config["broker"]["paper_trading"])
    regime_detector = MacroRegimeDetector(pipeline)

    capital = config["account"]["starting_capital"]
    if broker.connected:
        account = broker.get_account()
        if account:
            capital = account["equity"]

    positions = broker.get_positions() if broker.connected else []
    exposures = {p["symbol"]: p["market_value"] for p in positions}

    print("\n" + "=" * 60)
    print("  DALIOBOT TRADING POWERHOUSE — FULL DASHBOARD")
    print("=" * 60)

    print(risk_mgr.get_risk_dashboard(capital, positions, exposures))
    print(regime_detector.get_regime_summary())

    if positions:
        print("\n  OPEN POSITIONS:")
        for p in positions:
            print(f"    {p['symbol']}: {p['qty']} shares @ ${p['avg_entry_price']:.2f} "
                  f"(P&L: ${p['unrealized_pl']:.2f})")


def cmd_kill(config: dict):
    """Emergency shutdown."""
    from core.broker import AlpacaBroker
    from core.risk_manager import RiskManager

    broker = AlpacaBroker(paper=config["broker"]["paper_trading"])
    risk_mgr = RiskManager(config)

    print("\n🚨 EMERGENCY SHUTDOWN INITIATED")
    risk_mgr.activate_kill_switch()

    if broker.connected:
        result = broker.emergency_shutdown()
        print(f"  Orders cancelled: {result['orders_cancelled']}")
        print(f"  Positions closed: {result['positions_closed']}")
    else:
        print("  [Broker not connected — cancel orders manually]")

    print("  Kill switch is now ACTIVE. No new trades will be placed.")
    print("  Run 'python main.py dashboard' to review status.\n")


def main():
    parser = argparse.ArgumentParser(description="DalioBot Trading Powerhouse")
    parser.add_argument(
        "command",
        choices=["morning", "scan", "backtest", "train", "dashboard", "paper", "kill"],
        help="Command to run",
    )
    args = parser.parse_args()

    config = load_config()

    commands = {
        "morning": cmd_morning,
        "scan": cmd_scan,
        "backtest": cmd_backtest,
        "train": cmd_train,
        "dashboard": cmd_dashboard,
        "paper": cmd_morning,  # Same as morning but always paper
        "kill": cmd_kill,
    }

    commands[args.command](config)


if __name__ == "__main__":
    main()
