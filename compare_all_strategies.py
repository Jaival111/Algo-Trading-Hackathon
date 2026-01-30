#!/usr/bin/env python3
"""
Compare three strategies side-by-side:
1. Original strategy (strategy.py)
2. Improved strategy (strategy_improved.py)
3. Adaptive regime-based strategy (strategy_adaptive.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_data
from helpers import add_all_indicators, add_all_features
from regime import hmm_regime
from strategy import generate_signals
from strategy_improved import generate_signals_improved
from strategy_adaptive import generate_signals_adaptive, print_regime_distribution
from backtest import backtest, Backtest
from metrics import calculate_metrics

import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def run_strategy(df: pd.DataFrame, strategy_name: str, strategy_func, use_regime: bool = False):
    """Run a single strategy and return metrics."""
    print(f"\nRunning {strategy_name}...")
    
    # Generate signals
    df_strategy = strategy_func(df.copy())
    
    # Run backtest
    if use_regime and "market_regime" in df_strategy.columns:
        backtest_engine = Backtest(
            initial_capital=100_000,
            stop_loss_atr=2.5,
            take_profit_atr=4.0,
            risk_per_trade=0.01,
            max_drawdown_pct=0.20
        )
        result_df, trades = backtest_engine.run(df_strategy, use_regime_params=True)
    else:
        result_df, trades = backtest(
            df_strategy,
            initial_capital=100_000,
            risk_per_trade=0.01,
            stop_loss_atr=2.5,
            take_profit_atr=4.0,
            max_open_trades=1,
            max_drawdown_limit=0.20
        )
    
    # Calculate metrics
    metrics = calculate_metrics(result_df, trades, 100_000)
    
    print(f"  ✓ {strategy_name}: {metrics['num_trades']} trades, "
          f"{metrics['total_return']:.2%} return, "
          f"Sharpe {metrics['sharpe_ratio']:.4f}")
    
    return metrics, result_df, trades


def main():
    """Run comparison of all strategies."""
    
    print("="*80)
    print("STRATEGY COMPARISON: ORIGINAL vs IMPROVED vs ADAPTIVE")
    print("="*80)
    
    # ========================================
    # 1. Load and prepare data
    # ========================================
    print("\nLoading data...")
    data_dir = Path(__file__).parent.parent / "Equity_1min"
    df = load_data(str(data_dir))
    print(f"✓ Loaded {len(df):,} bars of FINNIFTY 1-minute data")
    
    print("\nCalculating indicators and features...")
    df = add_all_indicators(df)
    df = add_all_features(df)
    if "returns" in df.columns:
        df["hmm_regime"] = hmm_regime(df["returns"], 2)
    print("✓ Technical analysis complete")
    
    # ========================================
    # 2. Run all strategies
    # ========================================
    print("\n" + "="*80)
    print("RUNNING BACKTESTS")
    print("="*80)
    
    strategies = []
    
    # Strategy 1: Original
    metrics_orig, df_orig, trades_orig = run_strategy(
        df, "Original Strategy", generate_signals
    )
    strategies.append({
        "Strategy": "Original",
        "metrics": metrics_orig,
        "df": df_orig,
        "trades": trades_orig
    })
    
    # Strategy 2: Improved
    metrics_impr, df_impr, trades_impr = run_strategy(
        df, "Improved Strategy", generate_signals_improved
    )
    strategies.append({
        "Strategy": "Improved",
        "metrics": metrics_impr,
        "df": df_impr,
        "trades": trades_impr
    })
    
    # Strategy 3: Adaptive (with regime-specific params)
    print("\n" + "-"*80)
    metrics_adapt, df_adapt, trades_adapt = run_strategy(
        df, "Adaptive Regime-Based", 
        lambda df: generate_signals_adaptive(df, allow_shorts=True),
        use_regime=True
    )
    strategies.append({
        "Strategy": "Adaptive",
        "metrics": metrics_adapt,
        "df": df_adapt,
        "trades": trades_adapt
    })
    
    # Print regime distribution for adaptive
    print_regime_distribution(df_adapt)
    
    # ========================================
    # 3. Create comparison table
    # ========================================
    print("\n" + "="*80)
    print("STRATEGY COMPARISON RESULTS")
    print("="*80)
    
    comparison_data = []
    for strat in strategies:
        m = strat["metrics"]
        final_capital = strat["df"]["capital" if "capital" in strat["df"].columns else "equity"].iloc[-1]
        pnl = final_capital - 100_000
        
        comparison_data.append({
            "Strategy": strat["Strategy"],
            "Total_Return_%": m["total_return"] * 100,
            "Sharpe_Ratio": m["sharpe_ratio"],
            "Sortino_Ratio": m["sortino_ratio"],
            "Max_Drawdown_%": m["max_drawdown"] * 100,
            "Win_Rate_%": m["win_rate"] * 100,
            "Profit_Factor": m["profit_factor"],
            "Num_Trades": m["num_trades"],
            "Final_Capital": final_capital,
            "PnL": pnl
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # ========================================
    # 4. Highlight improvements
    # ========================================
    print("\n" + "="*80)
    print("KEY IMPROVEMENTS")
    print("="*80)
    
    orig_return = comparison_data[0]["Total_Return_%"]
    impr_return = comparison_data[1]["Total_Return_%"]
    adapt_return = comparison_data[2]["Total_Return_%"]
    
    orig_sharpe = comparison_data[0]["Sharpe_Ratio"]
    impr_sharpe = comparison_data[1]["Sharpe_Ratio"]
    adapt_sharpe = comparison_data[2]["Sharpe_Ratio"]
    
    print(f"\n{'Metric':<30} {'Original':>15} {'Improved':>15} {'Adaptive':>15}")
    print("-" * 77)
    print(f"{'Total Return':<30} {orig_return:>14.2f}% {impr_return:>14.2f}% {adapt_return:>14.2f}%")
    print(f"{'Improvement vs Original':<30} {'---':>15} {impr_return - orig_return:>14.2f}% {adapt_return - orig_return:>14.2f}%")
    print()
    print(f"{'Sharpe Ratio':<30} {orig_sharpe:>15.4f} {impr_sharpe:>15.4f} {adapt_sharpe:>15.4f}")
    print(f"{'Improvement vs Original':<30} {'---':>15} {impr_sharpe - orig_sharpe:>15.4f} {adapt_sharpe - orig_sharpe:>15.4f}")
    print()
    print(f"{'Number of Trades':<30} {comparison_data[0]['Num_Trades']:>15} {comparison_data[1]['Num_Trades']:>15} {comparison_data[2]['Num_Trades']:>15}")
    print(f"{'Profit Factor':<30} {comparison_data[0]['Profit_Factor']:>15.2f} {comparison_data[1]['Profit_Factor']:>15.2f} {comparison_data[2]['Profit_Factor']:>15.2f}")
    
    # ========================================
    # 5. Save results
    # ========================================
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    comparison_df.to_csv(output_dir / "all_strategies_comparison.csv", index=False)
    print(f"\n✓ Comparison saved to {output_dir}/all_strategies_comparison.csv")
    
    # Save individual results
    for strat in strategies:
        name = strat["Strategy"].lower()
        strat["df"].to_csv(output_dir / f"{name}_backtest.csv", index=False)
    
    print("\n" + "="*80)
    print("✓ COMPARISON COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
