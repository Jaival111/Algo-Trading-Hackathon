from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd

from src.backtest import Trade


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    drawdown = (peak - equity_curve) / peak
    return drawdown.max()


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / max(len(returns), 1)
    std = excess.std()
    return float(excess.mean() / std) if std != 0 else 0.0


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    downside = returns[returns < 0]
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    return float((returns.mean() - risk_free_rate) / downside_std)


def win_rate(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl > 0)
    return wins / len(trades)


def profit_factor(trades: List[Trade]) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    return gross_profit / gross_loss if gross_loss != 0 else 0.0


def total_return(equity_curve: pd.Series) -> float:
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1


def compute_metrics(df: pd.DataFrame, trades: List[Trade], initial_capital: float = None) -> Dict[str, float]:
    # Use 'capital' or 'equity' column
    equity_col = "capital" if "capital" in df.columns else "equity"
    returns = df[equity_col].pct_change().fillna(0.0)
    metrics = {
        "total_return": total_return(df[equity_col]),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(df[equity_col]),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "num_trades": float(len(trades)),
    }
    return metrics


# Alias for compatibility
calculate_metrics = compute_metrics
