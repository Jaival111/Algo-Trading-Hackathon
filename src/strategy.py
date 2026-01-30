from __future__ import annotations

import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
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
