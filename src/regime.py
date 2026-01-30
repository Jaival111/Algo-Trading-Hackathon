from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_filter_1d(series: pd.Series, process_variance: float = 1e-5, measurement_variance: float = 1e-2) -> pd.Series:
    values = series.values
    n = len(values)
    if n == 0:
        return series
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = values[0]
    P[0] = 1.0
    for k in range(1, n):
        xhat_minus = xhat[k - 1]
        P_minus = P[k - 1] + process_variance
        K = P_minus / (P_minus + measurement_variance)
        xhat[k] = xhat_minus + K * (values[k] - xhat_minus)
        P[k] = (1 - K) * P_minus
    return pd.Series(xhat, index=series.index, name=f"kalman_{series.name}")


def volatility_regime(df: pd.DataFrame, window: int = 50) -> pd.Series:
    vol = df["returns"].rolling(window=window, min_periods=window).std()
    threshold = vol.rolling(window=window, min_periods=window).median()
    regime = (vol > threshold).astype(int)
    return regime.rename("vol_regime")


def hmm_regime(returns: pd.Series, n_states: int = 2) -> pd.Series:
    """Fallback HMM for compatibility - not used in adaptive strategy."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        return volatility_regime(pd.DataFrame({"returns": returns}), window=50)

    valid = returns.dropna()
    if len(valid) < 100:
        return volatility_regime(pd.DataFrame({"returns": returns}), window=50)

    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=42)
    X = valid.values.reshape(-1, 1)
    model.fit(X)
    hidden_states = model.predict(X)
    regime = pd.Series(index=valid.index, data=hidden_states, name="hmm_regime")
    return regime.reindex(returns.index).ffill()


def train_hmm_model(df: pd.DataFrame, n_states: int = 3) -> pd.Series:
    """
    Train HMM model with proper standardization and regime sorting.
    
    Key improvements:
    1. Standardizes inputs (returns + ATR) using StandardScaler
    2. Trains on multiple features for better regime separation
    3. Sorts regimes by variance: 0=Low (Trend), 1=Medium, 2=High (Volatile)
    
    Parameters:
        df: DataFrame with 'returns' and 'atr' columns
        n_states: Number of hidden states (default 3)
    
    Returns:
        pd.Series: Regime labels sorted by variance (0=Trend, 1=Normal, 2=Volatile)
    """
    try:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Warning: hmmlearn or sklearn not available, using volatility fallback")
        return volatility_regime(df, window=50)
    
    # Prepare features: returns and normalized ATR
    returns = df["returns"].fillna(0)
    atr_norm = (df["atr"] / df["close"]).ffill().fillna(0)
    
    # Stack features into 2D array
    X = np.column_stack([returns.values, atr_norm.values])
    
    # Remove any rows with inf or extreme values
    valid_mask = np.isfinite(X).all(axis=1)
    X_clean = X[valid_mask]
    valid_indices = df.index[valid_mask]
    
    if len(X_clean) < 100:
        print("Warning: Insufficient data for HMM, using volatility fallback")
        return volatility_regime(df, window=50)
    
    # CRITICAL: Standardize inputs for HMM convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Train HMM with increased iterations for convergence
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",  # Full covariance for better separation
        n_iter=500,  # More iterations
        random_state=42,
        tol=1e-3
    )
    
    try:
        model.fit(X_scaled)
    except Exception as e:
        print(f"Warning: HMM training failed ({e}), using volatility fallback")
        return volatility_regime(df, window=50)
    
    # Predict hidden states
    hidden_states = model.predict(X_scaled)
    
    # CRITICAL: Sort regimes by variance (state with lowest variance = Regime 0 = Trend)
    # Calculate variance for each state
    state_variances = []
    for state_id in range(n_states):
        state_mask = hidden_states == state_id
        if state_mask.sum() > 0:
            state_returns = X_clean[state_mask, 0]  # Returns column
            state_var = np.var(state_returns)
        else:
            state_var = np.inf  # Empty states get high variance
        state_variances.append((state_id, state_var))
    
    # Sort states by variance (ascending)
    sorted_states = sorted(state_variances, key=lambda x: x[1])
    
    # Create mapping: old_state_id -> new_state_id (0=Low var, 1=Med, 2=High)
    state_mapping = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_states)}
    
    # Remap hidden states
    remapped_states = np.array([state_mapping[s] for s in hidden_states])
    
    # Create series with full index
    regime = pd.Series(index=valid_indices, data=remapped_states, name="hmm_regime")
    regime = regime.reindex(df.index).ffill().bfill().fillna(1)  # Default to medium regime
    
    # Print distribution for debugging
    print("\nHMM Regime Distribution (Sorted by Variance):")
    for regime_id in range(n_states):
        count = (regime == regime_id).sum()
        pct = count / len(regime) * 100
        var_label = ["Low (Trend)", "Medium (Normal)", "High (Volatile)"][regime_id]
        print(f"  Regime {regime_id} ({var_label}): {count:,} bars ({pct:.1f}%)")
    
    return regime.astype(int)


def classify_market_regime(df: pd.DataFrame, use_hmm: bool = True) -> pd.Series:
    """
    Classify market into 3 regimes using HMM with variance-based sorting.
    
    Regimes:
    - 0: TREND (Low Variance) - Strong directional moves, good for trend-following
    - 1: NORMAL (Medium Variance) - Standard market conditions
    - 2: VOLATILE (High Variance) - Choppy, high volatility, mean-reversion opportunities
    
    Parameters:
        df: DataFrame with indicators and features
        use_hmm: If True, use HMM-based classification; else use rule-based
    
    Returns:
        pd.Series: Regime classification (0=Trend, 1=Normal, 2=Volatile)
    """
    df = df.copy()
    
    if use_hmm and "returns" in df.columns and "atr" in df.columns:
        # Use trained HMM model (sorted by variance)
        regime = train_hmm_model(df, n_states=3)
        return regime
    
    else:
        # Fallback: Rule-based classification
        kalman_price = df["kalman_close"]
        price_vs_kalman = (df["close"] - kalman_price) / kalman_price
        
        # Normalized ATR (volatility measure)
        atr_norm = df["atr"] / df["close"]
        atr_threshold = atr_norm.rolling(50, min_periods=50).quantile(0.7)
        
        # ADX for trend strength (RELAXED: 25 -> 20)
        adx_strong = df["adx"] > 20  # CHANGED FROM 25
        adx_weak = df["adx"] < 15    # Even weaker threshold
        
        # Initialize regime array
        regime = pd.Series(1, index=df.index, name="market_regime")  # Default: NORMAL
        
        # TREND (Regime 0): Strong ADX + Clear direction
        trend_condition = (
            (adx_strong)  # Strong trend
            & (atr_norm < atr_threshold)  # Lower volatility
            & (
                ((df["ema_fast"] > df["ema_slow"]) & (df["rsi"] > 40))  # Uptrend
                | ((df["ema_fast"] < df["ema_slow"]) & (df["rsi"] < 60))  # Downtrend
            )
        )
        regime.loc[trend_condition] = 0
        
        # VOLATILE (Regime 2): High ATR or very weak trend
        volatile_condition = (
            (atr_norm > atr_threshold)  # High volatility
            | (adx_weak)  # Very weak trend
        )
        regime.loc[volatile_condition & ~trend_condition] = 2
        
        # Smooth regime changes (require 3 bars confirmation)
        regime_smoothed = regime.rolling(3, min_periods=1).apply(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1], raw=False
        )
        
        return regime_smoothed.astype(int).rename("market_regime")


def get_regime_name(regime_id: int) -> str:
    """Get human-readable regime name."""
    regime_names = {
        0: "TREND (Low Volatility)",
        1: "NORMAL (Medium Volatility)",
        2: "VOLATILE (High Volatility)"
    }
    return regime_names.get(regime_id, "UNKNOWN")

