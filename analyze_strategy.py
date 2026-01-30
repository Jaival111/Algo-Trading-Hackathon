from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from src.backtest import backtest, Trade
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
from src.strategy import generate_signals


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


def analyze_signal_noise(df: pd.DataFrame) -> dict:
    """Analyze if indicators are flipping too frequently."""
    analysis = {}
    
    # Signal flips
    signal_changes = (df["signal"].diff() != 0).sum()
    analysis["total_signal_changes"] = int(signal_changes)
    analysis["signal_changes_per_1000_bars"] = float(signal_changes / len(df) * 1000)
    
    # RSI oscillations
    rsi_changes = (df["rsi"].diff().abs() > 10).sum()
    analysis["large_rsi_swings"] = int(rsi_changes)
    analysis["rsi_swings_per_1000_bars"] = float(rsi_changes / len(df) * 1000)
    
    # MACD histogram flips
    macd_flips = ((df["macd_hist"] > 0) != (df["macd_hist"].shift(1) > 0)).sum()
    analysis["macd_histogram_flips"] = int(macd_flips)
    analysis["macd_flips_per_1000_bars"] = float(macd_flips / len(df) * 1000)
    
    # EMA crossovers
    ema_cross = ((df["ema_fast"] > df["ema_slow"]) != (df["ema_fast"].shift(1) > df["ema_slow"].shift(1))).sum()
    analysis["ema_crossovers"] = int(ema_cross)
    analysis["ema_crossovers_per_1000_bars"] = float(ema_cross / len(df) * 1000)
    
    # Signal quality - how long does average signal last?
    signal_durations = []
    current_signal = 0
    duration = 0
    for sig in df["signal"]:
        if sig == current_signal:
            duration += 1
        else:
            if current_signal != 0:
                signal_durations.append(duration)
            current_signal = sig
            duration = 1
    
    analysis["avg_signal_duration_bars"] = float(np.mean(signal_durations)) if signal_durations else 0
    analysis["min_signal_duration_bars"] = float(np.min(signal_durations)) if signal_durations else 0
    
    # Interpretation
    interpretation = []
    if analysis["signal_changes_per_1000_bars"] > 50:
        interpretation.append("⚠️  HIGH NOISE: Signals changing too frequently (>50 per 1000 bars)")
    if analysis["avg_signal_duration_bars"] < 5:
        interpretation.append("⚠️  SHORT SIGNALS: Average signal lasts < 5 bars (likely noise)")
    if analysis["rsi_swings_per_1000_bars"] > 100:
        interpretation.append("⚠️  VOLATILE RSI: Large RSI swings indicate choppy market")
    if analysis["macd_flips_per_1000_bars"] > 30:
        interpretation.append("⚠️  MACD WHIPSAW: MACD histogram flipping too often")
    
    if not interpretation:
        interpretation.append("✅ Signal noise appears reasonable")
    
    analysis["interpretation"] = interpretation
    return analysis


def analyze_overtrading(df: pd.DataFrame, trades: List[Trade], initial_capital: float) -> dict:
    """Check if strategy is taking too many low-quality trades."""
    analysis = {}
    
    # Trade frequency
    analysis["total_trades"] = len(trades)
    analysis["trades_per_day"] = len(trades) / max((df.index[-1] - df.index[0]).days, 1)
    analysis["trades_per_1000_bars"] = len(trades) / len(df) * 1000
    
    # Trade quality
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]
    
    analysis["win_rate"] = len(winning_trades) / len(trades) if trades else 0
    analysis["avg_win"] = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
    analysis["avg_loss"] = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
    analysis["win_loss_ratio"] = abs(analysis["avg_win"] / analysis["avg_loss"]) if analysis["avg_loss"] != 0 else 0
    
    # Commission impact (assuming $2 per trade)
    commission_per_trade = 2.0
    total_commission = len(trades) * commission_per_trade
    analysis["total_commission_cost"] = total_commission
    analysis["commission_as_pct_capital"] = (total_commission / initial_capital) * 100
    
    # Slippage impact (assuming 0.02% per trade)
    slippage_pct = 0.0002
    total_slippage = sum(abs(t.entry_price * t.size * slippage_pct) + abs(t.exit_price * t.size * slippage_pct) for t in trades)
    analysis["total_slippage_cost"] = total_slippage
    analysis["slippage_as_pct_capital"] = (total_slippage / initial_capital) * 100
    
    # Net impact
    total_trading_costs = total_commission + total_slippage
    analysis["total_trading_costs"] = total_trading_costs
    analysis["trading_costs_as_pct_capital"] = (total_trading_costs / initial_capital) * 100
    
    # Small profit trades (unprofitable after costs)
    small_profit_threshold = commission_per_trade + (slippage_pct * 2)
    small_profit_trades = [t for t in trades if 0 < t.pnl < small_profit_threshold]
    analysis["small_profit_trades"] = len(small_profit_trades)
    analysis["small_profit_trades_pct"] = (len(small_profit_trades) / len(trades) * 100) if trades else 0
    
    # Interpretation
    interpretation = []
    if analysis["trades_per_1000_bars"] > 50:
        interpretation.append("⚠️  OVERTRADING: Too many trades per 1000 bars (>50)")
    if analysis["trading_costs_as_pct_capital"] > 5:
        interpretation.append(f"🚨 HIGH COSTS: Trading costs are {analysis['trading_costs_as_pct_capital']:.2f}% of capital")
    if analysis["small_profit_trades_pct"] > 30:
        interpretation.append(f"⚠️  LOW QUALITY: {analysis['small_profit_trades_pct']:.1f}% trades have minimal profit")
    if analysis["win_loss_ratio"] < 1.5:
        interpretation.append(f"⚠️  POOR WIN/LOSS: Win/loss ratio {analysis['win_loss_ratio']:.2f} is below 1.5")
    
    if not interpretation:
        interpretation.append("✅ Trade frequency appears reasonable")
    
    analysis["interpretation"] = interpretation
    return analysis


def analyze_risk_management(df: pd.DataFrame, trades: List[Trade]) -> dict:
    """Analyze if stop-loss is too tight or risk management is suboptimal."""
    analysis = {}
    
    # Stop-loss analysis
    stopped_out = [t for t in trades if (t.direction == 1 and t.exit_price <= t.entry_price) or 
                                       (t.direction == -1 and t.exit_price >= t.entry_price)]
    take_profit_exits = [t for t in trades if (t.direction == 1 and t.exit_price > t.entry_price) or
                                              (t.direction == -1 and t.exit_price < t.entry_price)]
    
    analysis["total_trades"] = len(trades)
    analysis["stopped_out_count"] = len(stopped_out)
    analysis["take_profit_count"] = len(take_profit_exits)
    analysis["stop_out_rate"] = (len(stopped_out) / len(trades) * 100) if trades else 0
    
    # Average trade duration
    trade_durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]  # minutes
    analysis["avg_trade_duration_minutes"] = np.mean(trade_durations) if trade_durations else 0
    analysis["median_trade_duration_minutes"] = np.median(trade_durations) if trade_durations else 0
    
    # Quick stop-outs (< 10 bars)
    quick_stops = [t for t in stopped_out if (t.exit_time - t.entry_time).total_seconds() / 60 < 10]
    analysis["quick_stop_outs"] = len(quick_stops)
    analysis["quick_stop_out_rate"] = (len(quick_stops) / len(trades) * 100) if trades else 0
    
    # Average ATR at entry
    atr_values = df["atr"].dropna()
    analysis["avg_atr"] = float(atr_values.mean())
    analysis["median_atr"] = float(atr_values.median())
    
    # Stop distance analysis
    if stopped_out:
        stop_distances = []
        for t in stopped_out:
            stop_dist = abs(t.exit_price - t.entry_price)
            stop_distances.append(stop_dist)
        analysis["avg_stop_distance"] = np.mean(stop_distances)
        analysis["stop_distance_vs_atr_ratio"] = analysis["avg_stop_distance"] / analysis["avg_atr"] if analysis["avg_atr"] > 0 else 0
    else:
        analysis["avg_stop_distance"] = 0
        analysis["stop_distance_vs_atr_ratio"] = 0
    
    # Drawdown analysis
    equity_curve = df["equity"]
    peak = equity_curve.expanding().max()
    drawdown = (peak - equity_curve) / peak
    analysis["max_drawdown_pct"] = float(drawdown.max() * 100)
    analysis["avg_drawdown_pct"] = float(drawdown[drawdown > 0].mean() * 100) if (drawdown > 0).any() else 0
    
    # Consecutive losses
    loss_streak = 0
    max_loss_streak = 0
    for t in trades:
        if t.pnl < 0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0
    analysis["max_consecutive_losses"] = max_loss_streak
    
    # Interpretation
    interpretation = []
    if analysis["stop_out_rate"] > 70:
        interpretation.append(f"🚨 TIGHT STOPS: {analysis['stop_out_rate']:.1f}% of trades stopped out (target <60%)")
    if analysis["quick_stop_out_rate"] > 30:
        interpretation.append(f"⚠️  PREMATURE STOPS: {analysis['quick_stop_out_rate']:.1f}% stopped out within 10 minutes")
    if analysis["stop_distance_vs_atr_ratio"] < 1.5:
        interpretation.append(f"⚠️  STOP TOO TIGHT: Stop distance is {analysis['stop_distance_vs_atr_ratio']:.2f}x ATR (recommend 2-3x)")
    if analysis["max_consecutive_losses"] > 10:
        interpretation.append(f"⚠️  LONG LOSING STREAK: {analysis['max_consecutive_losses']} consecutive losses")
    if analysis["avg_trade_duration_minutes"] < 15:
        interpretation.append("⚠️  SHORT TRADES: Avg trade duration < 15 min suggests premature exits")
    
    if not interpretation:
        interpretation.append("✅ Risk management appears reasonable")
    
    analysis["interpretation"] = interpretation
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze strategy for flaws")
    parser.add_argument("--data", required=True, help="Path to CSV or folder")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("STRATEGY DIAGNOSTIC ANALYSIS")
    print("="*80 + "\n")
    
    # Load and prepare data
    df = load_data(args.data)
    df = build_features(df)
    df = generate_signals(df)
    
    # Run backtest
    bt_df, trades = backtest(df)
    metrics = compute_metrics(bt_df, trades)
    
    # Display basic metrics
    print("📊 BASIC METRICS")
    print("-" * 80)
    print(f"Initial Capital:  ${100000:,.2f}")
    print(f"Final Capital:    ${bt_df['equity'].iloc[-1]:,.2f}")
    print(f"Total Return:     {metrics['total_return']*100:.2f}%")
    print(f"Max Drawdown:     {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate:         {metrics['win_rate']*100:.2f}%")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.4f}")
    print(f"Total Trades:     {int(metrics['num_trades'])}")
    print()
    
    # 1. Signal Noise Analysis
    print("🔍 1. SIGNAL NOISE ANALYSIS")
    print("-" * 80)
    noise_analysis = analyze_signal_noise(bt_df)
    print(f"Signal changes per 1000 bars:  {noise_analysis['signal_changes_per_1000_bars']:.1f}")
    print(f"Average signal duration:       {noise_analysis['avg_signal_duration_bars']:.1f} bars")
    print(f"MACD flips per 1000 bars:      {noise_analysis['macd_flips_per_1000_bars']:.1f}")
    print(f"RSI swings per 1000 bars:      {noise_analysis['rsi_swings_per_1000_bars']:.1f}")
    print(f"EMA crossovers per 1000 bars:  {noise_analysis['ema_crossovers_per_1000_bars']:.1f}")
    print("\n💡 Interpretation:")
    for interp in noise_analysis["interpretation"]:
        print(f"   {interp}")
    print()
    
    # 2. Overtrading Analysis
    print("📈 2. OVERTRADING ANALYSIS")
    print("-" * 80)
    overtrade_analysis = analyze_overtrading(bt_df, trades, 100000.0)
    print(f"Trades per 1000 bars:          {overtrade_analysis['trades_per_1000_bars']:.1f}")
    print(f"Win rate:                      {overtrade_analysis['win_rate']*100:.1f}%")
    print(f"Avg win:                       ${overtrade_analysis['avg_win']:.2f}")
    print(f"Avg loss:                      ${overtrade_analysis['avg_loss']:.2f}")
    print(f"Win/Loss ratio:                {overtrade_analysis['win_loss_ratio']:.2f}x")
    print(f"\nCommission costs:              ${overtrade_analysis['total_commission_cost']:.2f} ({overtrade_analysis['commission_as_pct_capital']:.2f}%)")
    print(f"Slippage costs:                ${overtrade_analysis['total_slippage_cost']:.2f} ({overtrade_analysis['slippage_as_pct_capital']:.2f}%)")
    print(f"Total trading costs:           ${overtrade_analysis['total_trading_costs']:.2f} ({overtrade_analysis['trading_costs_as_pct_capital']:.2f}%)")
    print(f"Small profit trades:           {overtrade_analysis['small_profit_trades']} ({overtrade_analysis['small_profit_trades_pct']:.1f}%)")
    print("\n💡 Interpretation:")
    for interp in overtrade_analysis["interpretation"]:
        print(f"   {interp}")
    print()
    
    # 3. Risk Management Analysis
    print("🛡️  3. RISK MANAGEMENT ANALYSIS")
    print("-" * 80)
    risk_analysis = analyze_risk_management(bt_df, trades)
    print(f"Stop-out rate:                 {risk_analysis['stop_out_rate']:.1f}%")
    print(f"Take-profit rate:              {risk_analysis['take_profit_count']/risk_analysis['total_trades']*100:.1f}%")
    print(f"Quick stop-outs (<10 min):     {risk_analysis['quick_stop_outs']} ({risk_analysis['quick_stop_out_rate']:.1f}%)")
    print(f"Avg trade duration:            {risk_analysis['avg_trade_duration_minutes']:.1f} minutes")
    print(f"Average ATR:                   ${risk_analysis['avg_atr']:.2f}")
    print(f"Stop distance vs ATR:          {risk_analysis['stop_distance_vs_atr_ratio']:.2f}x")
    print(f"Max consecutive losses:        {risk_analysis['max_consecutive_losses']}")
    print(f"Max drawdown:                  {risk_analysis['max_drawdown_pct']:.2f}%")
    print("\n💡 Interpretation:")
    for interp in risk_analysis["interpretation"]:
        print(f"   {interp}")
    print()
    
    # Overall recommendations
    print("🎯 RECOMMENDED ACTIONS")
    print("-" * 80)
    
    recommendations = []
    
    # Signal noise fixes
    if noise_analysis["signal_changes_per_1000_bars"] > 50:
        recommendations.append("1. Add signal confirmation filter - require signal to persist for 3-5 bars")
        recommendations.append("2. Increase RSI thresholds (e.g., long>60, short<40) for stronger signals")
        recommendations.append("3. Add ADX minimum threshold (e.g., >25) to avoid choppy markets")
    
    # Overtrading fixes
    if overtrade_analysis["trades_per_1000_bars"] > 50 or overtrade_analysis["small_profit_trades_pct"] > 30:
        recommendations.append("4. Reduce position sizing to account for trading costs")
        recommendations.append("5. Add minimum profit target filter (e.g., >1% expected move)")
        recommendations.append("6. Increase take-profit multiplier from 3x to 4-5x ATR")
    
    # Risk management fixes
    if risk_analysis["stop_out_rate"] > 70:
        recommendations.append("7. Widen stop-loss from 2x to 2.5-3x ATR")
        recommendations.append("8. Add volatility filter - avoid trading during high volatility spikes")
    
    if risk_analysis["quick_stop_out_rate"] > 30:
        recommendations.append("9. Add entry confirmation - wait 2-3 bars after signal before entering")
        recommendations.append("10. Use limit orders instead of market orders to reduce slippage")
    
    if not recommendations:
        recommendations.append("✅ Strategy appears well-configured. Consider Optuna optimization for fine-tuning.")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n" + "="*80)
    print("Run 'python optimize_strategy.py --data \"../Equity_1min\" --trials 100' for automated optimization")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
