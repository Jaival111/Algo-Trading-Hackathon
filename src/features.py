from __future__ import annotations

import numpy as np
import pandas as pd


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"]).diff()
    return df


def add_volatility_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["volatility"] = df["returns"].rolling(window=window, min_periods=window).std() * np.sqrt(window)
    return df


def add_correlation_features(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    df = df.copy()
    volume_returns = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["corr_returns_volume"] = df["returns"].rolling(window=window, min_periods=window).corr(volume_returns)
    return df


def beta(returns: pd.Series, benchmark_returns: pd.Series, window: int = 50) -> pd.Series:
    cov = returns.rolling(window=window, min_periods=window).cov(benchmark_returns)
    var = benchmark_returns.rolling(window=window, min_periods=window).var()
    return cov / var


def zscore(series: pd.Series, window: int = 50) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    return (series - mean) / std
