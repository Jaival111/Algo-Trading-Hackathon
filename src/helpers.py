"""Helper functions to build features and indicators consistently across scripts."""

from __future__ import annotations

import pandas as pd

from indicators import (
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
from features import add_correlation_features, add_return_features, add_volatility_features, beta, zscore
from regime import kalman_filter_1d, hmm_regime


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe."""
    df = df.copy()
    
    # Moving averages
    df["sma_20"] = sma(df["close"], 20)
    df["sma_50"] = sma(df["close"], 50)
    df["ema_fast"] = ema(df["close"], 12)
    df["ema_slow"] = ema(df["close"], 26)
    
    # Momentum indicators
    df["rsi"] = rsi(df["close"], 14)
    
    # Volatility
    df["atr"] = atr(df["high"], df["low"], df["close"], 14)
    
    # MACD
    macd_df = macd(df["close"], 12, 26, 9)
    df = pd.concat([df, macd_df], axis=1)
    
    # Trend strength
    adx_df = adx(df["high"], df["low"], df["close"], 14)
    df = pd.concat([df, adx_df], axis=1)
    
    # Stochastic
    stoch_df = stochastic_oscillator(df["high"], df["low"], df["close"], 14, 3)
    df = pd.concat([df, stoch_df], axis=1)
    
    # Rate of change
    df["roc"] = roc(df["close"], 12)
    
    # Bollinger Bands
    bb_df = bollinger_bands(df["close"], 20, 2)
    df = pd.concat([df, bb_df], axis=1)
    df["std_20"] = standard_deviation(df["close"], 20)
    
    # Volume indicators
    df["obv"] = on_balance_volume(df["close"], df["volume"])
    df["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])
    
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all advanced features to the dataframe."""
    df = df.copy()
    
    # Returns and volatility
    df = add_return_features(df)
    df = add_volatility_features(df, 20)
    
    # Correlation and beta
    df = add_correlation_features(df, 50)
    
    # Z-score
    df["zscore_close"] = zscore(df["close"], 50)
    
    # Kalman filter
    df["kalman_close"] = kalman_filter_1d(df["close"], 1e-5, 1e-2)
    
    return df


def add_regime_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Add HMM-based regime detection."""
    df = df.copy()
    df["hmm_regime"] = hmm_regime(df)
    return df
