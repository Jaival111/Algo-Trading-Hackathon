from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float


def backtest(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    risk_per_trade: float = 0.01,
    stop_loss_atr: float = 2.0,
    take_profit_atr: float = 3.0,
    max_open_trades: int = 1,
    max_drawdown_limit: float = 0.3,
) -> Tuple[pd.DataFrame, List[Trade]]:
    df = df.copy()
    df["position"] = 0
    df["equity"] = initial_capital
    trades: List[Trade] = []

    open_positions = []
    equity = initial_capital
    peak_equity = initial_capital

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        open_positions = [pos for pos in open_positions if pos is not None]

        for pos in list(open_positions):
            if pos["direction"] == 1:
                stop_hit = row["low"] <= pos["stop_loss"]
                tp_hit = row["high"] >= pos["take_profit"]
                if stop_hit or tp_hit:
                    exit_price = pos["stop_loss"] if stop_hit else pos["take_profit"]
                    pnl = (exit_price - pos["entry_price"]) * pos["size"]
                    equity += pnl
                    trades.append(
                        Trade(
                            entry_time=pos["entry_time"],
                            exit_time=row.name,
                            direction=1,
                            entry_price=pos["entry_price"],
                            exit_price=exit_price,
                            size=pos["size"],
                            pnl=pnl,
                            return_pct=pnl / max(pos["entry_price"], 1e-9),
                        )
                    )
                    open_positions.remove(pos)
            elif pos["direction"] == -1:
                stop_hit = row["high"] >= pos["stop_loss"]
                tp_hit = row["low"] <= pos["take_profit"]
                if stop_hit or tp_hit:
                    exit_price = pos["stop_loss"] if stop_hit else pos["take_profit"]
                    pnl = (pos["entry_price"] - exit_price) * pos["size"]
                    equity += pnl
                    trades.append(
                        Trade(
                            entry_time=pos["entry_time"],
                            exit_time=row.name,
                            direction=-1,
                            entry_price=pos["entry_price"],
                            exit_price=exit_price,
                            size=pos["size"],
                            pnl=pnl,
                            return_pct=pnl / max(pos["entry_price"], 1e-9),
                        )
                    )
                    open_positions.remove(pos)

        drawdown = (peak_equity - equity) / peak_equity
        if drawdown > max_drawdown_limit:
            df.iloc[i:, df.columns.get_loc("position")] = 0
            df.iloc[i:, df.columns.get_loc("equity")] = equity
            break

        signal = row.get("signal", 0)
        if signal != 0 and len(open_positions) < max_open_trades and not np.isnan(row["atr"]):
            risk_amount = equity * risk_per_trade
            stop_distance = row["atr"] * stop_loss_atr
            if stop_distance > 0:
                size = risk_amount / stop_distance
                if signal == 1:
                    entry_price = row["close"]
                    stop_loss = entry_price - stop_distance
                    take_profit = entry_price + row["atr"] * take_profit_atr
                else:
                    entry_price = row["close"]
                    stop_loss = entry_price + stop_distance
                    take_profit = entry_price - row["atr"] * take_profit_atr

                open_positions.append(
                    {
                        "entry_time": row.name,
                        "direction": signal,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "size": size,
                    }
                )

        df.iloc[i, df.columns.get_loc("position")] = sum(p["direction"] for p in open_positions)
        df.iloc[i, df.columns.get_loc("equity")] = equity
        peak_equity = max(peak_equity, equity)

    df["equity"] = df["equity"].ffill().fillna(initial_capital)
    return df, trades


class Backtest:
    """
    Backtesting engine with support for regime-specific risk parameters.
    """
    
    def __init__(
        self,
        initial_capital: float = 100_000,
        stop_loss_atr: float = 2.5,
        take_profit_atr: float = 4.0,
        risk_per_trade: float = 0.01,
        max_drawdown_pct: float = 0.20,
        max_open_trades: int = 1
    ):
        self.initial_capital = initial_capital
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.risk_per_trade = risk_per_trade
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_trades = max_open_trades
    
    def run(
        self, 
        df: pd.DataFrame, 
        use_regime_params: bool = False
    ) -> Tuple[pd.DataFrame, List[Trade]]:
        """
        Run backtest with optional regime-specific parameters.
        
        Parameters:
            df: DataFrame with signals and indicators
            use_regime_params: If True, use regime-specific risk params
        
        Returns:
            Tuple of (result_df, trades)
        """
        if use_regime_params and "market_regime" in df.columns:
            return self._run_with_regime_params(df)
        else:
            return backtest(
                df,
                initial_capital=self.initial_capital,
                risk_per_trade=self.risk_per_trade,
                stop_loss_atr=self.stop_loss_atr,
                take_profit_atr=self.take_profit_atr,
                max_open_trades=self.max_open_trades,
                max_drawdown_limit=self.max_drawdown_pct
            )
    
    def _run_with_regime_params(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[Trade]]:
        """
        Run backtest with dynamic regime-based risk parameters.
        """
        from strategy_adaptive import get_regime_specific_params
        
        df = df.copy()
        df["position"] = 0
        df["capital"] = self.initial_capital
        trades: List[Trade] = []
        
        open_positions = []
        capital = self.initial_capital
        peak_capital = self.initial_capital
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            
            # Exit management for open positions
            for pos in list(open_positions):
                if pos["direction"] == 1:  # Long
                    stop_hit = row["low"] <= pos["stop_loss"]
                    tp_hit = row["high"] >= pos["take_profit"]
                    if stop_hit or tp_hit:
                        exit_price = pos["stop_loss"] if stop_hit else pos["take_profit"]
                        pnl = (exit_price - pos["entry_price"]) * pos["size"]
                        capital += pnl
                        
                        trades.append(Trade(
                            entry_time=pos["entry_time"],
                            exit_time=row.name,
                            direction=1,
                            entry_price=pos["entry_price"],
                            exit_price=exit_price,
                            size=pos["size"],
                            pnl=pnl,
                            return_pct=pnl / max(pos["entry_price"] * pos["size"], 1e-9)
                        ))
                        open_positions.remove(pos)
                        
                elif pos["direction"] == -1:  # Short
                    stop_hit = row["high"] >= pos["stop_loss"]
                    tp_hit = row["low"] <= pos["take_profit"]
                    if stop_hit or tp_hit:
                        exit_price = pos["stop_loss"] if stop_hit else pos["take_profit"]
                        pnl = (pos["entry_price"] - exit_price) * pos["size"]
                        capital += pnl
                        
                        trades.append(Trade(
                            entry_time=pos["entry_time"],
                            exit_time=row.name,
                            direction=-1,
                            entry_price=pos["entry_price"],
                            exit_price=exit_price,
                            size=pos["size"],
                            pnl=pnl,
                            return_pct=pnl / max(pos["entry_price"] * pos["size"], 1e-9)
                        ))
                        open_positions.remove(pos)
            
            # Check drawdown limit
            drawdown = (peak_capital - capital) / peak_capital
            if drawdown > self.max_drawdown_pct:
                df.iloc[i:, df.columns.get_loc("position")] = 0
                df.iloc[i:, df.columns.get_loc("capital")] = capital
                break
            
            # Entry management with regime-specific params
            signal = row.get("signal", 0)
            if signal != 0 and len(open_positions) < self.max_open_trades:
                if not np.isnan(row["atr"]) and row["atr"] > 0:
                    # Get regime-specific parameters
                    regime = int(row.get("market_regime", 0))
                    params = get_regime_specific_params(regime)
                    
                    risk_amount = capital * params["risk_per_trade"]
                    stop_distance = row["atr"] * params["stop_loss_atr"]
                    
                    if stop_distance > 0:
                        size = risk_amount / stop_distance
                        entry_price = row["close"]
                        
                        if signal == 1:  # Long
                            stop_loss = entry_price - stop_distance
                            take_profit = entry_price + row["atr"] * params["take_profit_atr"]
                        else:  # Short
                            stop_loss = entry_price + stop_distance
                            take_profit = entry_price - row["atr"] * params["take_profit_atr"]
                        
                        open_positions.append({
                            "entry_time": row.name,
                            "direction": signal,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "size": size,
                            "regime": regime
                        })
            
            # Update state
            df.iloc[i, df.columns.get_loc("position")] = sum(p["direction"] for p in open_positions)
            df.iloc[i, df.columns.get_loc("capital")] = capital
            peak_capital = max(peak_capital, capital)
        
        # Ensure capital column is filled
        df["capital"] = df["capital"].ffill().fillna(self.initial_capital)
        
        # Add equity column for compatibility
        df["equity"] = df["capital"]
        
        return df, trades
