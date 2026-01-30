from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest import backtest
from src.data_loader import load_data
from src.optuna_optimizer import (
    build_features_with_params,
    generate_signals_with_params,
    load_optimized_params,
    optimize_strategy,
)
from src.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize trading strategy with Optuna hyperparameter tuning."
    )
    parser.add_argument("--data", required=True, help="Path to CSV file or folder with CSVs")
    parser.add_argument(
        "--trials", type=int, default=100, help="Number of Optuna trials (default 100)"
    )
    parser.add_argument(
        "--output", default="optimized_params.json", help="Output file for optimized parameters"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test optimized parameters on data",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)

    best_params = optimize_strategy(args.data, n_trials=args.trials, output_path=args.output)

    if args.test:
        print("\n" + "=" * 80)
        print("TESTING OPTIMIZED PARAMETERS")
        print("=" * 80)

        df = load_data(args.data)
        df = build_features_with_params(df, best_params)
        df = generate_signals_with_params(df, best_params)

        bt_df, trades = backtest(
            df,
            initial_capital=best_params["initial_capital"],
            risk_per_trade=best_params["risk_per_trade"],
            stop_loss_atr=best_params["stop_loss_atr"],
            take_profit_atr=best_params["take_profit_atr"],
            max_open_trades=int(best_params["max_open_trades"]),
            max_drawdown_limit=best_params["max_drawdown_limit"],
        )

        metrics = compute_metrics(bt_df, trades)

        output_path = Path("outputs")
        output_path.mkdir(parents=True, exist_ok=True)
        bt_df.to_csv(output_path / "optimized_backtest_results.csv")

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_path / "optimized_metrics.csv", index=False)

        print("\nOptimized Strategy Metrics:")
        print(metrics_df.to_string(index=False))
        print(f"\nResults saved to outputs/optimized_backtest_results.csv")
        print(f"Metrics saved to outputs/optimized_metrics.csv")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
