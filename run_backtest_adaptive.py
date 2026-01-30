#!/usr/bin/env python3
"""
Run backtest with adaptive regime-based strategy.

This script tests the dynamic strategy that switches between:
- Trend-following in bull markets
- Mean-reversion in choppy/high-volatility markets  
- Defensive/short positions in bear markets
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_data, validate_data
from helpers import add_all_indicators, add_all_features
from regime import hmm_regime
from strategy_adaptive import generate_signals_adaptive, print_regime_distribution
from backtest import Backtest
from metrics import calculate_metrics
from plotting import plot_backtest_results

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    """Run adaptive regime-based backtest."""
    
    print("="*80)
    print("ADAPTIVE REGIME-BASED STRATEGY BACKTEST")
    print("="*80)
    print("\nStrategy Overview:")
    print("  • BULL TREND (Regime 0): Aggressive trend-following with tight stops")
    print("  • HIGH VOLATILITY (Regime 1): Mean-reversion with wide stops")
    print("  • BEAR TREND (Regime 2): Defensive (cash or shorts)")
    print()
    
    # ========================================
    # 1. Load Data
    # ========================================
    print("Loading data...")
    
    # Excel file path (can be changed to CSV directory or single CSV)
    data_path = r"C:\Users\JAIVAL CHAUHAN\Desktop\AlgoTrading\futures_minute.xlsx"
    
    # Alternative paths (uncomment to use):
    # data_path = Path(__file__).parent.parent / "Equity_1min"  # CSV directory
    # data_path = r"C:\path\to\your\file.csv"  # Single CSV
    
    df = load_data(str(data_path))
    print(f"✓ Loaded {len(df):,} bars of data")
    
    # Validate data quality
    validate_data(df)
    
    # ========================================
    # 2. Add Technical Indicators
    # ========================================
    print("\nCalculating technical indicators...")
    df = add_all_indicators(df)
    print("✓ Added 13+ technical indicators")
    
    # ========================================
    # 3. Add Features
    # ========================================
    print("\nCalculating advanced features...")
    df = add_all_features(df)
    print("✓ Added returns, volatility, z-score, correlation, beta")
    
    # ========================================
    # 4. Add HMM Regime Detection
    # ========================================
    print("\nRunning HMM regime detection...")
    if "returns" in df.columns:
        df["hmm_regime"] = hmm_regime(df["returns"], 2)
    else:
        print("  Warning: 'returns' column not found, skipping HMM")
    print("✓ Added HMM-based regime classification")
    
    # ========================================
    # 5. Generate Adaptive Signals
    # ========================================
    print("\nGenerating adaptive regime-based signals...")
    df = generate_signals_adaptive(df, allow_shorts=True)
    print("✓ Generated adaptive signals with regime switching")
    
    # Print regime distribution
    print_regime_distribution(df)
    
    # ========================================
    # 6. Run Backtest
    # ========================================
    print("\nRunning backtest with dynamic risk parameters...")
    
    # Initialize backtest with default params (will be overridden per regime)
    backtest = Backtest(
        initial_capital=100_000,
        stop_loss_atr=2.5,      # These will be overridden
        take_profit_atr=4.0,    # by regime-specific params
        risk_per_trade=0.01,
        max_drawdown_pct=0.20
    )
    
    # Run with regime-aware risk management
    result_df, trades = backtest.run(df, use_regime_params=True)
    
    # ========================================
    # 7. Calculate Metrics
    # ========================================
    print("\nCalculating performance metrics...")
    metrics = calculate_metrics(result_df, trades, backtest.initial_capital)
    
    # ========================================
    # 8. Display Results
    # ========================================
    print("\n" + "="*80)
    print("ADAPTIVE STRATEGY RESULTS")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Value':>20}")
    print("-" * 52)
    print(f"{'Total Return':<30} {metrics['total_return']:>19.2%}")
    print(f"{'Sharpe Ratio':<30} {metrics['sharpe_ratio']:>20.4f}")
    print(f"{'Sortino Ratio':<30} {metrics['sortino_ratio']:>20.4f}")
    print(f"{'Max Drawdown':<30} {metrics['max_drawdown']:>19.2%}")
    print(f"{'Win Rate':<30} {metrics['win_rate']:>19.2%}")
    print(f"{'Profit Factor':<30} {metrics['profit_factor']:>20.2f}")
    print(f"{'Number of Trades':<30} {metrics['num_trades']:>20}")
    
    final_capital = result_df["capital"].iloc[-1]
    pnl = final_capital - backtest.initial_capital
    print(f"\n{'Initial Capital':<30} ${backtest.initial_capital:>18,.2f}")
    print(f"{'Final Capital':<30} ${final_capital:>18,.2f}")
    print(f"{'Total PnL':<30} ${pnl:>18,.2f}")
    
    # ========================================
    # 9. Regime-Specific Performance
    # ========================================
    print("\n" + "="*80)
    print("PERFORMANCE BY REGIME")
    print("="*80)
    
    from strategy_adaptive import analyze_regime_performance
    regime_perf = analyze_regime_performance(result_df, trades)
    
    if not regime_perf.empty:
        print(f"\n{regime_perf.to_string(index=False)}\n")
    else:
        print("\nNo regime-specific performance data available\n")
    
    # ========================================
    # 10. Trade Analysis
    # ========================================
    if trades:
        print("="*80)
        print("TRADE ANALYSIS")
        print("="*80)
        
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl < 0]
        
        print(f"\nWinning Trades: {len(winning)}")
        if winning:
            print(f"  Average: ${sum(t.pnl for t in winning) / len(winning):,.2f}")
            print(f"  Largest: ${max(t.pnl for t in winning):,.2f}")
        
        print(f"\nLosing Trades: {len(losing)}")
        if losing:
            print(f"  Average: ${sum(t.pnl for t in losing) / len(losing):,.2f}")
            print(f"  Largest: ${min(t.pnl for t in losing):,.2f}")
    
    # ========================================
    # 11. Save Results
    # ========================================
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    result_df.to_csv(output_dir / "adaptive_backtest_results.csv", index=False)
    
    # Save metrics
    import pandas as pd
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "adaptive_metrics.csv", index=False)
    
    print(f"\n✓ Results saved to {output_dir}/")
    print(f"  - adaptive_backtest_results.csv")
    print(f"  - adaptive_metrics.csv")
    
    # ========================================
    # 12. Generate Plot
    # ========================================
    print("\nGenerating interactive plot...")
    fig = plot_backtest_results(result_df, trades)
    
    # Save plot
    fig.write_html(str(output_dir / "adaptive_backtest_plot.html"))
    print(f"✓ Plot saved to {output_dir}/adaptive_backtest_plot.html")
    
    print("\n" + "="*80)
    print("✓ ADAPTIVE BACKTEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
