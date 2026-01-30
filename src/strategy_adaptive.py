from __future__ import annotations

import pandas as pd
import numpy as np

from regime import classify_market_regime, get_regime_name


def generate_signals_adaptive(df: pd.DataFrame, allow_shorts: bool = True) -> pd.DataFrame:
    """
    Adaptive strategy with relaxed filters and variance-based regime classification.
    
    Regimes (sorted by variance):
    - TREND (0): Low variance - Trend-following strategies
    - NORMAL (1): Medium variance - Balanced approach
    - VOLATILE (2): High variance - Mean-reversion strategies
    
    RELAXED FILTERS:
    - ADX threshold: 25 -> 20 (more trend signals)
    - RSI oversold/overbought: 30/70 -> 35/65 (more entries)
    
    Parameters:
        df: DataFrame with all indicators
        allow_shorts: If False, go to cash in bear markets instead of shorting
    
    Returns:
        DataFrame with 'signal' column (1=long, -1=short, 0=cash)
    """
    df = df.copy()
    
    # Classify market regime using HMM with variance sorting
    df["market_regime"] = classify_market_regime(df, use_hmm=True)
    
    # Initialize signals
    df["signal"] = 0
    df["signal_reason"] = ""
    
    # ==========================================
    # REGIME 0: TREND (Low Variance) - Trend Following
    # ==========================================
    # Strategy: Follow strong directional moves
    trend_regime = df["market_regime"] == 0
    
    # Long in uptrends (RELAXED: ADX 22->20)
    trend_long_condition = (
        trend_regime
        & (df["ema_fast"] > df["ema_slow"])  # Uptrend
        & (df["macd_hist"] > 0)  # MACD confirmation
        & (df["rsi"] > 45) & (df["rsi"] < 75)  # Healthy momentum
        & (df["close"] > df["sma_50"])  # Above medium-term trend
        & (df["adx"] > 20)  # RELAXED: Was 22, now 20
        & (df["close"] > df["kalman_close"])  # Above smoothed trend
    )
    df.loc[trend_long_condition, "signal"] = 1
    df.loc[trend_long_condition, "signal_reason"] = "TREND_LONG"
    
    # Short in downtrends (if allowed)
    if allow_shorts:
        trend_short_condition = (
            trend_regime
            & (df["ema_fast"] < df["ema_slow"])  # Downtrend
            & (df["macd_hist"] < 0)  # MACD bearish
            & (df["rsi"] < 55) & (df["rsi"] > 25)  # Not oversold
            & (df["close"] < df["sma_50"])  # Below medium-term trend
            & (df["adx"] > 20)  # RELAXED: Was 22, now 20
            & (df["close"] < df["kalman_close"])  # Below smoothed trend
        )
        df.loc[trend_short_condition, "signal"] = -1
        df.loc[trend_short_condition, "signal_reason"] = "TREND_SHORT"
    
    # ==========================================
    # REGIME 1: NORMAL (Medium Variance) - Balanced
    # ==========================================
    # Strategy: Balanced approach, moderate filters
    normal_regime = df["market_regime"] == 1
    
    # Long on mild oversold with trend support
    normal_long = (
        normal_regime
        & (df["rsi"] < 45)  # Mildly oversold
        & (df["ema_fast"] > df["ema_slow"])  # But in uptrend
        & (df["close"] > df["kalman_close"])  # Above smoothed trend
    )
    df.loc[normal_long, "signal"] = 1
    df.loc[normal_long, "signal_reason"] = "NORMAL_LONG"
    
    # Short on mild overbought with downtrend
    if allow_shorts:
        normal_short = (
            normal_regime
            & (df["rsi"] > 55)  # Mildly overbought
            & (df["ema_fast"] < df["ema_slow"])  # But in downtrend
            & (df["close"] < df["kalman_close"])  # Below smoothed trend
        )
        df.loc[normal_short, "signal"] = -1
        df.loc[normal_short, "signal_reason"] = "NORMAL_SHORT"
    
    # ==========================================
    # REGIME 2: VOLATILE (High Variance) - Mean Reversion
    # ==========================================
    # Strategy: Buy extreme oversold, sell extreme overbought
    # RELAXED: RSI 30/70 -> 35/65 for more entries
    volatile_regime = df["market_regime"] == 2
    
    # Mean reversion long: Buy oversold (RELAXED)
    mean_rev_long = (
        volatile_regime
        & (df["rsi"] < 35)  # RELAXED: Was 30, now 35
        & (df["close"] < df["bb_lower"])  # Below lower Bollinger
        & (df["stoch_k"] < 25)  # RELAXED: Was 20, now 25
    )
    df.loc[mean_rev_long, "signal"] = 1
    df.loc[mean_rev_long, "signal_reason"] = "VOLATILE_MEAN_REV_LONG"
    
    # Mean reversion short: Sell overbought (RELAXED)
    if allow_shorts:
        mean_rev_short = (
            volatile_regime
            & (df["rsi"] > 65)  # RELAXED: Was 70, now 65
            & (df["close"] > df["bb_upper"])  # Above upper Bollinger
            & (df["stoch_k"] > 75)  # RELAXED: Was 80, now 75
        )
        df.loc[mean_rev_short, "signal"] = -1
        df.loc[mean_rev_short, "signal_reason"] = "VOLATILE_MEAN_REV_SHORT"
    
    # ==========================================
    # Signal Confirmation & Filtering
    # ==========================================
    # Require signal to persist for 2 bars (less strict than improved strategy)
    df["signal_confirmed"] = df["signal"]
    for i in range(2, len(df)):
        if (df["signal"].iloc[i] == df["signal"].iloc[i-1] and 
            df["signal"].iloc[i] != 0):
            df.iloc[i, df.columns.get_loc("signal_confirmed")] = df["signal"].iloc[i]
        else:
            df.iloc[i, df.columns.get_loc("signal_confirmed")] = 0
    
    # Only take NEW signals (prevent re-entry on same signal)
    df["signal_final"] = 0
    prev_signal = 0
    for i in range(len(df)):
        confirmed = df["signal_confirmed"].iloc[i]
        if confirmed != 0 and confirmed != prev_signal:
            df.iloc[i, df.columns.get_loc("signal_final")] = confirmed
            prev_signal = confirmed
        elif confirmed == 0:
            prev_signal = 0
    
    # Replace original signal with final filtered signal
    df["signal"] = df["signal_final"]
    
    return df


def get_regime_specific_params(regime: int) -> dict:
    """
    Get risk management parameters specific to each regime.
    
    Regime 0 (TREND): Low variance, strong direction
    Regime 1 (NORMAL): Medium variance, balanced
    Regime 2 (VOLATILE): High variance, mean-reversion
    
    Returns:
        dict: Parameters for stop_loss_atr, take_profit_atr, risk_per_trade
    """
    if regime == 0:  # TREND (Low Variance)
        return {
            "stop_loss_atr": 2.0,      # Tighter stops in trends
            "take_profit_atr": 5.0,    # Trailing for trends
            "risk_per_trade": 0.015,   # Slightly aggressive
            "description": "Trend-following with trailing stops"
        }
    elif regime == 1:  # NORMAL (Medium Variance)
        return {
            "stop_loss_atr": 2.5,      # Moderate stops
            "take_profit_atr": 4.0,    # Moderate targets
            "risk_per_trade": 0.012,   # Balanced risk
            "description": "Balanced approach"
        }
    elif regime == 2:  # VOLATILE (High Variance)
        return {
            "stop_loss_atr": 3.0,      # Wider stops for volatility
            "take_profit_atr": 3.0,    # Quick exits on reversion
            "risk_per_trade": 0.01,    # Conservative in chaos
            "description": "Mean-reversion with room to breathe"
        }
    else:
        return {
            "stop_loss_atr": 2.5,
            "take_profit_atr": 4.0,
            "risk_per_trade": 0.01,
            "description": "Default parameters"
        }


def print_regime_distribution(df: pd.DataFrame) -> None:
    """Print distribution of time spent in each regime."""
    if "market_regime" not in df.columns:
        print("No regime data available")
        return
    
    regime_counts = df["market_regime"].value_counts().sort_index()
    total = len(df)
    
    print("\n" + "="*60)
    print("MARKET REGIME DISTRIBUTION")
    print("="*60)
    for regime_id, count in regime_counts.items():
        pct = (count / total) * 100
        regime_name = get_regime_name(regime_id)
        print(f"{regime_name:20s} (Regime {regime_id}): {count:6d} bars ({pct:5.2f}%)")
    print("="*60 + "\n")


def analyze_regime_performance(df: pd.DataFrame, trades: list) -> pd.DataFrame:
    """
    Analyze strategy performance by regime.
    
    Parameters:
        df: Backtest dataframe with regime column
        trades: List of Trade objects
    
    Returns:
        DataFrame: Performance metrics by regime
    """
    from collections import defaultdict
    
    regime_trades = defaultdict(list)
    
    # Group trades by entry regime
    for trade in trades:
        try:
            entry_idx = df.index.get_loc(trade.entry_time)
            regime = df.iloc[entry_idx]["market_regime"]
            regime_trades[regime].append(trade)
        except:
            continue
    
    # Calculate metrics per regime
    results = []
    for regime_id in sorted(regime_trades.keys()):
        trades_in_regime = regime_trades[regime_id]
        
        if not trades_in_regime:
            continue
        
        winning_trades = [t for t in trades_in_regime if t.pnl > 0]
        losing_trades = [t for t in trades_in_regime if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in trades_in_regime)
        win_rate = len(winning_trades) / len(trades_in_regime) if trades_in_regime else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        results.append({
            "Regime": get_regime_name(regime_id),
            "Trades": len(trades_in_regime),
            "Win_Rate_%": win_rate * 100,
            "Total_PnL": total_pnl,
            "Avg_Win": avg_win,
            "Avg_Loss": avg_loss,
            "Profit_Factor": abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades and sum(t.pnl for t in losing_trades) != 0 else 0
        })
    
    return pd.DataFrame(results)
