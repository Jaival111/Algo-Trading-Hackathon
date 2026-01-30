from __future__ import annotations

from typing import List

import pandas as pd
import plotly.graph_objects as go

from src.backtest import Trade


def price_with_trades(df: pd.DataFrame, trades: List[Trade]) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
            )
        ]
    )

    for trade in trades:
        color = "green" if trade.direction == 1 else "red"
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_time, trade.exit_time],
                y=[trade.entry_price, trade.exit_price],
                mode="lines+markers",
                line=dict(color=color),
                name="Trade",
                showlegend=False,
            )
        )

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    return fig


def equity_curve_plot(df: pd.DataFrame) -> go.Figure:
    # Use 'capital' or 'equity' column
    equity_col = "capital" if "capital" in df.columns else "equity"
    fig = go.Figure(
        data=[go.Scatter(x=df.index, y=df[equity_col], mode="lines", name="Equity")]
    )
    fig.update_layout(height=300)
    return fig


def plot_backtest_results(df: pd.DataFrame, trades: List[Trade]) -> go.Figure:
    """
    Create comprehensive backtest visualization with multiple subplots.
    """
    from plotly.subplots import make_subplots
    
    equity_col = "capital" if "capital" in df.columns else "equity"
    
    # Create subplots: price + equity curve
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Price Action with Trades", "Equity Curve"),
        row_heights=[0.7, 0.3]
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add trade markers
    for trade in trades:
        color = "green" if trade.pnl > 0 else "red"
        marker_symbol = "triangle-up" if trade.direction == 1 else "triangle-down"
        
        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_time],
                y=[trade.entry_price],
                mode="markers",
                marker=dict(
                    size=10,
                    color="blue" if trade.direction == 1 else "orange",
                    symbol=marker_symbol
                ),
                name="Entry",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Exit marker
        fig.add_trace(
            go.Scatter(
                x=[trade.exit_time],
                y=[trade.exit_price],
                mode="markers",
                marker=dict(
                    size=10,
                    color=color,
                    symbol="circle"
                ),
                name="Exit",
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[equity_col],
            mode="lines",
            name="Equity",
            line=dict(color="purple", width=2)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Capital ($)", row=2, col=1)
    
    return fig
