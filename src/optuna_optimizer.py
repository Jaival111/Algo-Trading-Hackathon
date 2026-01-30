from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd

from src.backtest import backtest
from src.data_loader import load_data
from src.features import (
    add_correlation_features,
    add_return_features,
    add_volatility_features,
    beta,
    zscore,
)
from src.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    macd,
    on_balance_volume,
    roc,
    rsi,
    sma,
    standard_deviation,
    stochastic_oscillator,
    vwap,
)
from src.metrics import compute_metrics
from src.regime import kalman_filter_1d, hmm_regime


def build_features_with_params(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    df["sma_20"] = sma(df["close"], params["sma_period"])
    df["sma_50"] = sma(df["close"], 50)
    df["ema_fast"] = ema(df["close"], params["ema_fast_period"])
    df["ema_slow"] = ema(df["close"], params["ema_slow_period"])
    df["rsi"] = rsi(df["close"], params["rsi_period"])
    df["atr"] = atr(df["high"], df["low"], df["close"], params["atr_period"])

    macd_df = macd(
        df["close"],
        params["macd_fast"],
        params["macd_slow"],
        params["macd_signal"],
    )
    df = pd.concat([df, macd_df], axis=1)

    adx_df = adx(df["high"], df["low"], df["close"], params["adx_period"])
    df = pd.concat([df, adx_df], axis=1)

    stoch_df = stochastic_oscillator(
        df["high"],
        df["low"],
        df["close"],
        params["stoch_k_period"],
        params["stoch_d_period"],
    )
    df = pd.concat([df, stoch_df], axis=1)

    df["roc"] = roc(df["close"], params["roc_period"])

    bb_df = bollinger_bands(df["close"], params["bb_period"], params["bb_std"])
    df = pd.concat([df, bb_df], axis=1)

    df["std_20"] = standard_deviation(df["close"], 20)
    df["obv"] = on_balance_volume(df["close"], df["volume"])
    df["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])

    df = add_return_features(df)
    df = add_volatility_features(df, 20)
    df = add_correlation_features(df, 50)
    df["zscore_close"] = zscore(df["close"], 50)
    df["kalman_close"] = kalman_filter_1d(df["close"], 1e-5, 1e-2)
    df["hmm_regime"] = hmm_regime(df["returns"], 2)

    return df


def generate_signals_with_params(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    long_condition = (
        (df["ema_fast"] > df["ema_slow"])
        & (df["rsi"] > params["rsi_long_threshold"])
        & (df["macd_hist"] > 0)
        & (df["adx"] > params["adx_threshold"])
        & (df["close"] > df["sma_50"])
    )

    short_condition = (
        (df["ema_fast"] < df["ema_slow"])
        & (df["rsi"] < params["rsi_short_threshold"])
        & (df["macd_hist"] < 0)
        & (df["adx"] > params["adx_threshold"])
        & (df["close"] < df["sma_50"])
    )

    df["signal"] = 0
    df.loc[long_condition, "signal"] = 1
    df.loc[short_condition, "signal"] = -1

    return df


def objective(trial: optuna.Trial, data_path: str) -> float:
    try:
        params = {
            "ema_fast_period": trial.suggest_int("ema_fast_period", 5, 20),
            "ema_slow_period": trial.suggest_int("ema_slow_period", 20, 50),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "rsi_long_threshold": trial.suggest_int("rsi_long_threshold", 50, 70),
            "rsi_short_threshold": trial.suggest_int("rsi_short_threshold", 30, 50),
            "atr_period": trial.suggest_int("atr_period", 10, 20),
            "macd_fast": trial.suggest_int("macd_fast", 8, 15),
            "macd_slow": trial.suggest_int("macd_slow", 20, 30),
            "macd_signal": trial.suggest_int("macd_signal", 7, 12),
            "adx_period": trial.suggest_int("adx_period", 10, 20),
            "adx_threshold": trial.suggest_float("adx_threshold", 15.0, 30.0),
            "stoch_k_period": trial.suggest_int("stoch_k_period", 10, 20),
            "stoch_d_period": trial.suggest_int("stoch_d_period", 2, 5),
            "roc_period": trial.suggest_int("roc_period", 8, 15),
            "sma_period": trial.suggest_int("sma_period", 15, 30),
            "bb_period": trial.suggest_int("bb_period", 15, 25),
            "bb_std": trial.suggest_float("bb_std", 1.5, 2.5),
            "initial_capital": 100000.0,
            "risk_per_trade": trial.suggest_float("risk_per_trade", 0.005, 0.02),
            "stop_loss_atr": trial.suggest_float("stop_loss_atr", 1.5, 3.0),
            "take_profit_atr": trial.suggest_float("take_profit_atr", 2.0, 4.0),
            "max_open_trades": trial.suggest_int("max_open_trades", 1, 3),
            "max_drawdown_limit": trial.suggest_float("max_drawdown_limit", 0.2, 0.4),
        }

        df = load_data(data_path)
        df = build_features_with_params(df, params)
        df = generate_signals_with_params(df, params)

        bt_df, trades = backtest(
            df,
            initial_capital=params["initial_capital"],
            risk_per_trade=params["risk_per_trade"],
            stop_loss_atr=params["stop_loss_atr"],
            take_profit_atr=params["take_profit_atr"],
            max_open_trades=int(params["max_open_trades"]),
            max_drawdown_limit=params["max_drawdown_limit"],
        )

        metrics = compute_metrics(bt_df, trades)

        sharpe = metrics.get("sharpe_ratio", -1.0)
        if not isinstance(sharpe, (int, float)) or sharpe != sharpe:
            return -1.0

        return sharpe

    except Exception as e:
        print(f"Trial failed: {e}")
        return -1.0


def optimize_strategy(
    data_path: str, n_trials: int = 100, output_path: str = "optimized_params.json"
) -> Dict[str, Any]:
    sampler = TPESampler(seed=42)
    pruner = MedianPruner()
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    study.optimize(
        lambda trial: objective(trial, data_path),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params["initial_capital"] = 100000.0

    output_file = Path(output_path)
    with open(output_file, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Best Sharpe Ratio: {study.best_value:.4f}")
    print(f"\nBest Parameters:\n{json.dumps(best_params, indent=2)}")
    print(f"\nParameters saved to: {output_file}")
    print(f"{'='*80}\n")

    return best_params


def load_optimized_params(params_file: str = "optimized_params.json") -> Dict[str, Any]:
    path = Path(params_file)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Optimized parameters file not found: {params_file}")
