from __future__ import annotations

import pandas as pd


def generate_signals_improved(df: pd.DataFrame) -> pd.DataFrame:
    """
    Improved signal generation with fixes for:
    1. Signal noise - requires confirmation over multiple bars
    2. Overtrading - stricter entry conditions
    3. Risk management - better filtering of low-quality setups
    """
    df = df.copy()

    # Stronger trend filters (stricter thresholds)
    long_condition = (
        (df["ema_fast"] > df["ema_slow"])
        & (df["rsi"] > 60)  # Increased from 55 - stronger momentum required
        & (df["macd_hist"] > 0)
        & (df["adx"] > 25)  # Increased from 20 - avoid choppy markets
        & (df["close"] > df["sma_50"])
        & (df["close"] > df["bb_mid"])  # Above Bollinger mid-band
        & (df["volatility"] < df["volatility"].rolling(50).quantile(0.8))  # Avoid high volatility spikes
    )

    short_condition = (
        (df["ema_fast"] < df["ema_slow"])
        & (df["rsi"] < 40)  # Decreased from 45 - stronger bearish momentum
        & (df["macd_hist"] < 0)
        & (df["adx"] > 25)  # Increased from 20
        & (df["close"] < df["sma_50"])
        & (df["close"] < df["bb_mid"])  # Below Bollinger mid-band
        & (df["volatility"] < df["volatility"].rolling(50).quantile(0.8))  # Avoid high volatility spikes
    )

    # Generate raw signals
    df["signal_raw"] = 0
    df.loc[long_condition, "signal_raw"] = 1
    df.loc[short_condition, "signal_raw"] = -1

    # Signal confirmation: require signal to persist for 3 bars
    df["signal_confirmed"] = 0
    for i in range(3, len(df)):
        if (df["signal_raw"].iloc[i] == df["signal_raw"].iloc[i-1] == df["signal_raw"].iloc[i-2] and
            df["signal_raw"].iloc[i] != 0):
            df.iloc[i, df.columns.get_loc("signal_confirmed")] = df["signal_raw"].iloc[i]

    # Only take new signals (not continuing ones) - prevents overtrading
    df["signal"] = 0
    prev_signal = 0
    for i in range(len(df)):
        if df["signal_confirmed"].iloc[i] != 0 and df["signal_confirmed"].iloc[i] != prev_signal:
            df.iloc[i, df.columns.get_loc("signal")] = df["signal_confirmed"].iloc[i]
            prev_signal = df["signal_confirmed"].iloc[i]
        elif df["signal_confirmed"].iloc[i] == 0:
            prev_signal = 0

    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Original signal generation (kept for backward compatibility)."""
    df = df.copy()

    long_condition = (
        (df["ema_fast"] > df["ema_slow"])
        & (df["rsi"] > 55)
        & (df["macd_hist"] > 0)
        & (df["adx"] > 20)
        & (df["close"] > df["sma_50"])
    )

    short_condition = (
        (df["ema_fast"] < df["ema_slow"])
        & (df["rsi"] < 45)
        & (df["macd_hist"] < 0)
        & (df["adx"] > 20)
        & (df["close"] < df["sma_50"])
    )

    df["signal"] = 0
    df.loc[long_condition, "signal"] = 1
    df.loc[short_condition, "signal"] = -1

    return df
