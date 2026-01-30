from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest import backtest
from src.data_loader import load_data
from src.features import add_correlation_features, add_return_features, add_volatility_features, beta, zscore
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
from src.strategy import generate_signals
from src.optuna_optimizer import (
    build_features_with_params,
    generate_signals_with_params,
    load_optimized_params,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_20"] = sma(df["close"], 20)
    df["sma_50"] = sma(df["close"], 50)
    df["ema_fast"] = ema(df["close"], 12)
    df["ema_slow"] = ema(df["close"], 26)
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df["high"], df["low"], df["close"], 14)

    macd_df = macd(df["close"], 12, 26, 9)
    df = pd.concat([df, macd_df], axis=1)

    adx_df = adx(df["high"], df["low"], df["close"], 14)
    df = pd.concat([df, adx_df], axis=1)

    stoch_df = stochastic_oscillator(df["high"], df["low"], df["close"], 14, 3)
    df = pd.concat([df, stoch_df], axis=1)

    df["roc"] = roc(df["close"], 12)
    bb_df = bollinger_bands(df["close"], 20, 2)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest on provided dataset.")
    parser.add_argument("--data", required=True, help="Path to CSV file or folder with CSVs")
    parser.add_argument("--benchmark", default=None, help="Optional benchmark CSV path")
    parser.add_argument("--output", default="outputs", help="Output folder for results")
    parser.add_argument(
        "--optimized-params",
        default=None,
        help="Path to optimized_params.json for using optimized parameters",
    )
    args = parser.parse_args()

    df = load_data(args.data)

    if args.optimized_params:
        print(f"Loading optimized parameters from {args.optimized_params}")
        params = load_optimized_params(args.optimized_params)
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
    else:
        df = build_features(df)

        if args.benchmark:
            benchmark_df = load_data(args.benchmark)
            benchmark_df = add_return_features(benchmark_df)
            df["beta"] = beta(df["returns"], benchmark_df["returns"], 50)
        else:
            df["beta"] = pd.NA

        df = generate_signals(df)
        bt_df, trades = backtest(df)

    metrics = compute_metrics(bt_df, trades)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    bt_df.to_csv(output_path / "backtest_results.csv")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path / "metrics.csv", index=False)

    print("Backtest complete.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
