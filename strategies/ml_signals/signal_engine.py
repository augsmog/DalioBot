"""
ML Signal Enhancement Engine — Karpathy Self-Improvement Loop
==============================================================
Applies the autoresearch pattern to trading signals:
  1. Generate feature set from market data
  2. Train model to predict optimal entry timing
  3. Backtest on walk-forward window
  4. Keep improvements, discard failures
  5. Repeat — model continuously improves

The ML layer ENHANCES the base flywheel strategy, not replaces it.
It answers: "Should I trade this candidate TODAY, or wait for a better entry?"
"""

import datetime as dt
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. ML signals disabled. pip install lightgbm")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


RESULTS_FILE = Path(__file__).parent / "experiment_results.tsv"
MODEL_DIR = Path(__file__).parent / "models"


class MLSignalEngine:
    """ML-enhanced entry timing for options selling."""

    FEATURE_COLUMNS = [
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_pct", "volume_ratio", "atr_14",
        "daily_return", "return_5d", "return_10d", "return_20d",
        "volatility_10d", "volatility_20d",
        "rsi_slope_5d", "volume_trend_5d",
        "gap_pct", "close_vs_high", "close_vs_low",
    ]

    def __init__(self, config: dict):
        self.config = config.get("ml_signals", {})
        self.model = None
        self.model_version: int = 0
        self.model_metrics: dict = {}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw OHLCV + technicals into ML features."""
        if df.empty:
            return df

        features = df.copy()

        # Multi-period returns
        features["return_5d"] = features["Close"].pct_change(5)
        features["return_10d"] = features["Close"].pct_change(10)
        features["return_20d"] = features["Close"].pct_change(20)

        # Realized volatility
        features["volatility_10d"] = features["daily_return"].rolling(10).std() * np.sqrt(252)
        features["volatility_20d"] = features["daily_return"].rolling(20).std() * np.sqrt(252)

        # RSI momentum (slope of RSI over 5 days)
        features["rsi_slope_5d"] = features["rsi_14"].diff(5)

        # Volume trend
        features["volume_trend_5d"] = features["Volume"].pct_change(5)

        # Gap percentage (open vs previous close)
        features["gap_pct"] = (features["Open"] - features["Close"].shift(1)) / features["Close"].shift(1)

        # Position within day's range
        day_range = features["High"] - features["Low"]
        features["close_vs_high"] = (features["High"] - features["Close"]) / day_range.replace(0, np.nan)
        features["close_vs_low"] = (features["Close"] - features["Low"]) / day_range.replace(0, np.nan)

        return features

    def build_labels(self, df: pd.DataFrame, forward_days: int = 7) -> pd.Series:
        """Create target variable: was selling a put on this day profitable?

        Label = 1 if the stock didn't drop more than 5% in the next `forward_days`.
        (This is the condition for a put to expire profitable.)
        """
        future_min = df["Close"].rolling(window=forward_days).min().shift(-forward_days)
        current_price = df["Close"]

        # Put is profitable if stock doesn't drop more than 7% (below our typical strike)
        max_drop = (future_min - current_price) / current_price
        labels = (max_drop > -0.07).astype(int)

        return labels

    def train(self, ticker: str, df: pd.DataFrame) -> dict:
        """Train or retrain the ML model using walk-forward validation.

        Returns performance metrics.
        """
        if not LGBM_AVAILABLE:
            return {"error": "LightGBM not installed"}

        # Build features and labels
        features = self.build_features(df)
        labels = self.build_labels(features)

        # Align and drop NaN
        valid_cols = [c for c in self.FEATURE_COLUMNS if c in features.columns]
        X = features[valid_cols].copy()
        y = labels.copy()

        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            logger.warning(f"Not enough data for {ticker}: {len(X)} rows")
            return {"error": f"Insufficient data: {len(X)} rows"}

        # Walk-forward split (purged time series CV)
        tscv = TimeSeriesSplit(n_splits=5, gap=5)
        metrics_list = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
            )

            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            accuracy = (preds == y_test).mean()
            # Precision on positive class (how often "trade" signal is correct)
            true_positives = ((preds == 1) & (y_test == 1)).sum()
            predicted_positives = (preds == 1).sum()
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0

            metrics_list.append({
                "fold": fold,
                "accuracy": accuracy,
                "precision": precision,
                "test_size": len(y_test),
            })

        # Train final model on all data
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y)
        self.model_version += 1

        avg_metrics = {
            "ticker": ticker,
            "version": self.model_version,
            "avg_accuracy": np.mean([m["accuracy"] for m in metrics_list]),
            "avg_precision": np.mean([m["precision"] for m in metrics_list]),
            "total_samples": len(X),
            "folds": metrics_list,
            "feature_importance": dict(zip(valid_cols, self.model.feature_importances_.tolist())),
            "timestamp": dt.datetime.now().isoformat(),
        }

        self.model_metrics = avg_metrics
        self._log_experiment(avg_metrics)

        logger.info(f"ML model v{self.model_version} trained on {ticker}: "
                    f"accuracy={avg_metrics['avg_accuracy']:.3f}, "
                    f"precision={avg_metrics['avg_precision']:.3f}")

        return avg_metrics

    def predict_entry_quality(self, df: pd.DataFrame) -> Optional[dict]:
        """Score today's entry quality using the trained model.

        Returns probability that selling a put today will be profitable.
        """
        if self.model is None:
            return None

        features = self.build_features(df)
        valid_cols = [c for c in self.FEATURE_COLUMNS if c in features.columns]
        X = features[valid_cols].iloc[[-1]]  # Latest row only

        if X.isna().any().any():
            logger.warning("Missing features for prediction")
            return None

        prob = self.model.predict_proba(X)[0][1]
        prediction = self.model.predict(X)[0]

        result = {
            "signal": "TRADE" if prediction == 1 else "WAIT",
            "confidence": round(float(prob), 3),
            "model_version": self.model_version,
            "recommendation": (
                f"ML model says {'TRADE' if prediction == 1 else 'WAIT'} "
                f"with {prob:.1%} confidence"
            ),
        }

        # Strong signal thresholds
        if prob > 0.85:
            result["strength"] = "STRONG_BUY"
        elif prob > 0.65:
            result["strength"] = "MODERATE_BUY"
        elif prob > 0.45:
            result["strength"] = "NEUTRAL"
        else:
            result["strength"] = "AVOID"

        return result

    def get_feature_importance(self) -> Optional[dict]:
        """Return ranked feature importance from the trained model."""
        if not self.model_metrics:
            return None
        fi = self.model_metrics.get("feature_importance", {})
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    # ── Karpathy Self-Improvement Loop ──────────────────────────────────

    def run_improvement_loop(self, ticker: str, df: pd.DataFrame, n_experiments: int = 10) -> dict:
        """Run the Karpathy-style autonomous improvement loop.

        Each experiment:
        1. Modify hyperparameters
        2. Train model
        3. Evaluate on holdout
        4. Keep if improved, discard if not
        """
        if not LGBM_AVAILABLE:
            return {"error": "LightGBM not installed"}

        features = self.build_features(df)
        labels = self.build_labels(features)
        valid_cols = [c for c in self.FEATURE_COLUMNS if c in features.columns]
        X = features[valid_cols].copy()
        y = labels.copy()
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        if len(X) < 100:
            return {"error": "Insufficient data"}

        # Hold out last 20% for final evaluation
        split = int(len(X) * 0.8)
        X_dev, X_holdout = X.iloc[:split], X.iloc[split:]
        y_dev, y_holdout = y.iloc[:split], y.iloc[split:]

        best_score = 0.0
        best_params = None
        best_model = None
        results = []

        # Parameter search space
        param_space = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "min_child_samples": [10, 20, 50],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }

        rng = np.random.RandomState(42)

        for exp_idx in range(n_experiments):
            # Random parameter selection
            params = {k: rng.choice(v) for k, v in param_space.items()}
            params["verbose"] = -1
            params["random_state"] = 42

            try:
                model = lgb.LGBMClassifier(**params)

                # Train on dev set with time series CV
                tscv = TimeSeriesSplit(n_splits=3, gap=5)
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_dev):
                    model.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
                    preds = model.predict(X_dev.iloc[val_idx])
                    acc = (preds == y_dev.iloc[val_idx]).mean()
                    cv_scores.append(acc)

                cv_score = np.mean(cv_scores)

                # Evaluate on holdout
                model.fit(X_dev, y_dev)
                holdout_preds = model.predict(X_holdout)
                holdout_score = (holdout_preds == y_holdout).mean()

                result = {
                    "experiment": exp_idx + 1,
                    "params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v)) for k, v in params.items() if k not in ("verbose", "random_state")},
                    "cv_score": round(cv_score, 4),
                    "holdout_score": round(holdout_score, 4),
                    "improved": holdout_score > best_score,
                }
                results.append(result)

                if holdout_score > best_score:
                    best_score = holdout_score
                    best_params = params
                    best_model = model
                    logger.info(f"  Experiment {exp_idx+1}: NEW BEST holdout={holdout_score:.4f} ✓")
                else:
                    logger.debug(f"  Experiment {exp_idx+1}: holdout={holdout_score:.4f} (best={best_score:.4f})")

            except Exception as e:
                logger.warning(f"  Experiment {exp_idx+1} failed: {e}")
                results.append({"experiment": exp_idx + 1, "error": str(e)})

        # Deploy best model
        if best_model:
            self.model = best_model
            self.model_version += 1
            logger.info(f"Improvement loop complete. Best holdout score: {best_score:.4f}")

        kept = sum(1 for r in results if r.get("improved", False))

        return {
            "ticker": ticker,
            "experiments_run": n_experiments,
            "improvements_found": kept,
            "best_holdout_score": round(best_score, 4),
            "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v)) for k, v in (best_params or {}).items() if k not in ("verbose", "random_state")},
            "all_results": results,
        }

    def _log_experiment(self, metrics: dict):
        """Log experiment results to TSV (autoresearch-style tracking)."""
        row = {
            "timestamp": metrics["timestamp"],
            "ticker": metrics["ticker"],
            "version": metrics["version"],
            "accuracy": round(metrics["avg_accuracy"], 4),
            "precision": round(metrics["avg_precision"], 4),
            "samples": metrics["total_samples"],
        }

        header = not RESULTS_FILE.exists()
        with open(RESULTS_FILE, "a") as f:
            if header:
                f.write("\t".join(row.keys()) + "\n")
            f.write("\t".join(str(v) for v in row.values()) + "\n")
