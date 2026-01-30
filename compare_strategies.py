from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest import backtest
from src.data_loader import load_data
from src.features import add_correlation_features, add_return_features, add_volatility_features, zscore
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
from src.strategy_improved import generate_signals_improved


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
    parser = argparse.ArgumentParser(description="Compare original vs improved strategy")
    parser.add_argument("--data", required=True, help="Path to CSV file or folder with CSVs")
    parser.add_argument("--output", default="outputs", help="Output folder for results")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("STRATEGY COMPARISON: ORIGINAL vs IMPROVED")
    print("="*80 + "\n")

    df = load_data(args.data)
    df = build_features(df)

    # Run original strategy
    print("Running ORIGINAL strategy...")
    from src.strategy import generate_signals
    df_orig = generate_signals(df.copy())
    bt_orig, trades_orig = backtest(
        df_orig,
        initial_capital=100000.0,
        risk_per_trade=0.01,
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
        max_open_trades=1,
        max_drawdown_limit=0.3,
    )
    metrics_orig = compute_metrics(bt_orig, trades_orig)

    # Run improved strategy
    print("Running IMPROVED strategy...")
    df_improved = generate_signals_improved(df.copy())
    bt_improved, trades_improved = backtest(
        df_improved,
        initial_capital=100000.0,
        risk_per_trade=0.01,
        stop_loss_atr=2.5,  # Widened from 2.0
        take_profit_atr=4.0,  # Increased from 3.0
        max_open_trades=1,
        max_drawdown_limit=0.3,
    )
    metrics_improved = compute_metrics(bt_improved, trades_improved)

    # Comparison table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    comparison = pd.DataFrame({
        "Metric": [
            "Total Return (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Win Rate (%)",
            "Profit Factor",
            "Number of Trades",
            "Final Capital ($)"
        ],
        "Original": [
            f"{metrics_orig['total_return']*100:.2f}",
            f"{metrics_orig['sharpe_ratio']:.4f}",
            f"{metrics_orig['max_drawdown']*100:.2f}",
            f"{metrics_orig['win_rate']*100:.2f}",
            f"{metrics_orig['profit_factor']:.3f}",
            f"{int(metrics_orig['num_trades'])}",
            f"{bt_orig['equity'].iloc[-1]:,.2f}"
        ],
        "Improved": [
            f"{metrics_improved['total_return']*100:.2f}",
            f"{metrics_improved['sharpe_ratio']:.4f}",
            f"{metrics_improved['max_drawdown']*100:.2f}",
            f"{metrics_improved['win_rate']*100:.2f}",
            f"{metrics_improved['profit_factor']:.3f}",
            f"{int(metrics_improved['num_trades'])}",
            f"{bt_improved['equity'].iloc[-1]:,.2f}"
        ],
        "Change": [
            f"{(metrics_improved['total_return'] - metrics_orig['total_return'])*100:+.2f}",
            f"{metrics_improved['sharpe_ratio'] - metrics_orig['sharpe_ratio']:+.4f}",
            f"{(metrics_improved['max_drawdown'] - metrics_orig['max_drawdown'])*100:+.2f}",
            f"{(metrics_improved['win_rate'] - metrics_orig['win_rate'])*100:+.2f}",
            f"{metrics_improved['profit_factor'] - metrics_orig['profit_factor']:+.3f}",
            f"{int(metrics_improved['num_trades'] - metrics_orig['num_trades']):+d}",
            f"{bt_improved['equity'].iloc[-1] - bt_orig['equity'].iloc[-1]:+,.2f}"
        ]
    })
    
    print(comparison.to_string(index=False))

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bt_improved.to_csv(output_path / "improved_backtest_results.csv")
    pd.DataFrame([metrics_improved]).to_csv(output_path / "improved_metrics.csv", index=False)
    comparison.to_csv(output_path / "strategy_comparison.csv", index=False)

    print("\n" + "="*80)
    print("KEY IMPROVEMENTS IMPLEMENTED:")
    print("="*80)
    print("✅ 1. Signal Confirmation: Requires signal to persist 3 bars before entry")
    print("✅ 2. Stricter RSI: Long>60, Short<40 (was 55/45)")
    print("✅ 3. Higher ADX: Minimum 25 (was 20) - avoids choppy markets")
    print("✅ 4. Bollinger Filter: Only trade with price above/below mid-band")
    print("✅ 5. Volatility Filter: Avoid trading during volatility spikes")
    print("✅ 6. Wider Stops: 2.5x ATR (was 2.0x) - reduces premature stop-outs")
    print("✅ 7. Higher Take-Profit: 4.0x ATR (was 3.0x) - better win/loss ratio")
    print("✅ 8. Entry Confirmation: Waits for persistent signal, reduces noise")
    
    print("\n" + "="*80)
    print(f"Results saved to {output_path}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
