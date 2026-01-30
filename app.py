from __future__ import annotations

from pathlib import Path
import json
import tempfile

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest import backtest
from src.data_loader import load_data
from src.features import add_correlation_features, add_return_features, add_volatility_features, beta, zscore
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
from src.plotting import equity_curve_plot, price_with_trades
from src.regime import kalman_filter_1d, hmm_regime
from src.strategy import generate_signals
from src.optuna_optimizer import (
    build_features_with_params,
    generate_signals_with_params,
)


st.set_page_config(
    page_title="QuanTrade Pro | Algorithmic Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/algotrading',
        'Report a bug': 'https://github.com/yourusername/algotrading/issues',
        'About': "QuanTrade Pro - Professional Algorithmic Trading Platform"
    }
)

# Advanced Trading Platform CSS - Professional Grade
st.markdown("""
<style>
    /* Import Professional Trading Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Dark Trading Platform Theme */
    .stApp {
        background: #0a0e1a;
        background-image: 
            radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.05) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.05) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.05) 0px, transparent 50%);
    }
    
    /* Top Navigation Bar */
    .top-nav {
        background: rgba(17, 24, 39, 0.95);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(16, 185, 129, 0.1);
        padding: 16px 24px;
        margin: -80px -100px 20px -100px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
    }
    
    .platform-logo {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .platform-logo h1 {
        margin: 0;
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    .platform-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        background: rgba(16, 185, 129, 0.1);
        border-radius: 6px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Professional Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(31, 41, 55, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(55, 65, 81, 0.5);
        padding: 20px;
        margin: 8px 0;
        box-shadow: 
            0 4px 24px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(16, 185, 129, 0.5) 50%, 
            transparent 100%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .metric-card:hover {
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 
            0 8px 32px rgba(16, 185, 129, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-label {
        color: #9ca3af;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .metric-value-positive {
        color: #10b981;
        font-size: 32px;
        font-weight: 800;
        line-height: 1;
        font-variant-numeric: tabular-nums;
        text-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
        letter-spacing: -1px;
    }
    
    .metric-value-negative {
        color: #ef4444;
        font-size: 32px;
        font-weight: 800;
        line-height: 1;
        font-variant-numeric: tabular-nums;
        text-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
        letter-spacing: -1px;
    }
    
    .metric-value-neutral {
        color: #6366f1;
        font-size: 32px;
        font-weight: 800;
        line-height: 1;
        font-variant-numeric: tabular-nums;
        letter-spacing: -1px;
    }
    
    .metric-change {
        font-size: 13px;
        font-weight: 600;
        margin-top: 8px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .metric-change.positive { color: #10b981; }
    .metric-change.negative { color: #ef4444; }
    
    /* Market Regime Badges */
    .regime-bull {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 13px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        letter-spacing: 0.5px;
    }
    
    .regime-bear {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff;
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 13px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        letter-spacing: 0.5px;
    }
    
    .regime-neutral {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff;
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 13px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        letter-spacing: 0.5px;
    }
    
    /* Typography */
    h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 36px;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }
    
    h2 {
        color: #f3f4f6;
        font-weight: 700;
        font-size: 24px;
        letter-spacing: -0.5px;
        margin-top: 32px;
        margin-bottom: 16px;
    }
    
    h3 {
        color: #e5e7eb;
        font-weight: 600;
        font-size: 18px;
        letter-spacing: -0.3px;
        margin-top: 24px;
        margin-bottom: 12px;
    }
    
    /* Sidebar - Trading Panel Style */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
        border-right: 1px solid rgba(55, 65, 81, 0.5);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #d1d5db;
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #10b981;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 24px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Premium Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 14px 28px;
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        box-shadow: 
            0 4px 20px rgba(16, 185, 129, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 
            0 6px 28px rgba(16, 185, 129, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 
            0 2px 12px rgba(16, 185, 129, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    /* Modern Tab Design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(17, 24, 39, 0.6);
        border-radius: 12px;
        padding: 6px;
        border: 1px solid rgba(55, 65, 81, 0.5);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #9ca3af;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.3px;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(55, 65, 81, 0.3);
        color: #d1d5db;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
    }
    
    /* Enhanced DataFrames */
    .dataframe {
        background: rgba(17, 24, 39, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(55, 65, 81, 0.5);
        font-size: 13px;
    }
    
    .dataframe thead tr th {
        background: rgba(31, 41, 55, 0.9) !important;
        color: #10b981 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 1px;
        padding: 12px 16px !important;
        border-bottom: 2px solid rgba(16, 185, 129, 0.3) !important;
    }
    
    .dataframe tbody tr {
        border-bottom: 1px solid rgba(55, 65, 81, 0.3);
        transition: background 0.15s;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(31, 41, 55, 0.5);
    }
    
    .dataframe tbody tr td {
        padding: 12px 16px !important;
        color: #d1d5db;
        font-variant-numeric: tabular-nums;
    }
    
    /* Input Fields - Trading Terminal Style */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: rgba(17, 24, 39, 0.9) !important;
        border: 1px solid rgba(55, 65, 81, 0.5) !important;
        border-radius: 8px !important;
        color: #f3f4f6 !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
        transition: all 0.2s !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border: 1px solid rgba(16, 185, 129, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
        outline: none !important;
    }
    
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label {
        color: #9ca3af !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 6px !important;
    }
    
    /* Checkbox & Radio */
    .stCheckbox {
        color: #d1d5db;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(17, 24, 39, 0.6);
        border-radius: 10px;
        border: 1px solid rgba(55, 65, 81, 0.5);
        color: #d1d5db;
        font-weight: 600;
        padding: 12px 16px;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(31, 41, 55, 0.6);
        border-color: rgba(16, 185, 129, 0.3);
    }
    
    /* Divider */
    hr {
        border-color: rgba(55, 65, 81, 0.3);
        margin: 24px 0;
    }
    
    /* Alert/Info Boxes */
    .stAlert {
        background: rgba(17, 24, 39, 0.8);
        border-radius: 12px;
        border-left: 4px solid #10b981;
        padding: 16px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #10b981 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(17, 24, 39, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(16, 185, 129, 0.3);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(16, 185, 129, 0.5);
    }
    
    /* Loading Animation */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .loading-shimmer {
        animation: shimmer 2s infinite linear;
        background: linear-gradient(
            to right,
            rgba(17, 24, 39, 0.8) 0%,
            rgba(31, 41, 55, 0.8) 50%,
            rgba(17, 24, 39, 0.8) 100%
        );
        background-size: 1000px 100%;
    }
</style>
""", unsafe_allow_html=True)


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


def create_metric_card(label: str, value: str, is_positive: bool = None, change: str = None, icon: str = ""):
    """Create a professional trading platform metric card"""
    if is_positive is None:
        value_class = "metric-value-neutral"
    else:
        value_class = "metric-value-positive" if is_positive else "metric-value-negative"
    
    change_html = ""
    if change:
        change_class = "positive" if is_positive else "negative"
        change_icon = "▲" if is_positive else "▼"
        change_html = f'<div class="metric-change {change_class}">{change_icon} {change}</div>'
    
    icon_html = f'<span style="font-size: 14px;">{icon}</span>' if icon else ""
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon_html} {label}</div>
        <div class="{value_class}">{value}</div>
        {change_html}
    </div>
    """


def get_regime_badge(regime_value: int) -> str:
    """Get professional regime indicator badge"""
    regime_map = {
        0: ('<span class="regime-bull">🚀 BULLISH TREND</span>', 'Aggressive long positioning'),
        1: ('<span class="regime-neutral">⚡ HIGH VOLATILITY</span>', 'Mean-reversion strategy active'),
        2: ('<span class="regime-bear">🛡️ BEARISH TREND</span>', 'Defensive positioning active')
    }
    return regime_map.get(regime_value, ('<span class="regime-neutral">📊 NEUTRAL MARKET</span>', 'Balanced strategy'))


def create_enhanced_candlestick(df: pd.DataFrame, trades: pd.DataFrame = None):
    """Create professional trading chart with advanced styling"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.65, 0.2, 0.15],
        subplot_titles=('', '', '')
    )
    
    # Professional candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444',
            increasing_fillcolor='rgba(16, 185, 129, 0.4)',
            decreasing_fillcolor='rgba(239, 68, 68, 0.4)',
            increasing_line_width=1.5,
            decreasing_line_width=1.5,
        ),
        row=1, col=1
    )
    
    # Enhanced moving averages
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#3b82f6', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    if 'sma_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['sma_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='#8b5cf6', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Professional buy/sell signals
    if trades is not None and len(trades) > 0:
        entry_times_long = []
        entry_prices_long = []
        entry_times_short = []
        entry_prices_short = []
        exit_times = []
        exit_prices = []
        
        for trade in trades:
            if trade.direction == 1:  # Long
                entry_times_long.append(trade.entry_time)
                entry_prices_long.append(trade.entry_price)
            else:  # Short
                entry_times_short.append(trade.entry_time)
                entry_prices_short.append(trade.entry_price)
            
            exit_times.append(trade.exit_time)
            exit_prices.append(trade.exit_price)
        
        # Long entry signals
        if len(entry_times_long) > 0:
            fig.add_trace(
                go.Scatter(
                    x=entry_times_long,
                    y=entry_prices_long,
                    mode='markers',
                    name='Long Entry',
                    marker=dict(
                        symbol='triangle-up',
                        size=14,
                        color='#10b981',
                        line=dict(color='#ffffff', width=2)
                    ),
                    hovertemplate='<b>Long Entry</b><br>Price: $%{y:,.2f}<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Short entry signals
        if len(entry_times_short) > 0:
            fig.add_trace(
                go.Scatter(
                    x=entry_times_short,
                    y=entry_prices_short,
                    mode='markers',
                    name='Short Entry',
                    marker=dict(
                        symbol='triangle-down',
                        size=14,
                        color='#f59e0b',
                        line=dict(color='#ffffff', width=2)
                    ),
                    hovertemplate='<b>Short Entry</b><br>Price: $%{y:,.2f}<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Exit signals
        if len(exit_times) > 0:
            fig.add_trace(
                go.Scatter(
                    x=exit_times,
                    y=exit_prices,
                    mode='markers',
                    name='Exit',
                    marker=dict(
                        symbol='x',
                        size=12,
                        color='#ef4444',
                        line=dict(color='#ffffff', width=2)
                    ),
                    hovertemplate='<b>Exit</b><br>Price: $%{y:,.2f}<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Professional volume bars
    colors = ['#10b981' if row['close'] >= row['open'] else '#ef4444' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.6,
            showlegend=False,
            hovertemplate='Volume: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # RSI with professional styling
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='#6366f1', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(99, 102, 241, 0.1)',
                hovertemplate='RSI: %{y:.1f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # RSI threshold lines
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", opacity=0.6, line_width=1.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#10b981", opacity=0.6, line_width=1.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#6b7280", opacity=0.4, line_width=1, row=3, col=1)
    
    # Professional layout
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#0a0e1a',
        paper_bgcolor='rgba(17, 24, 39, 0.95)',
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="left",
            x=0,
            bgcolor='rgba(17, 24, 39, 0.9)',
            bordercolor='rgba(55, 65, 81, 0.5)',
            borderwidth=1,
            font=dict(size=11, color='#d1d5db')
        ),
        xaxis_rangeslider_visible=False,
        font=dict(color='#d1d5db', size=12, family='Inter'),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=80, b=10)
    )
    
    # Update axes with professional styling
    fig.update_xaxes(
        gridcolor='rgba(55, 65, 81, 0.2)',
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor='rgba(55, 65, 81, 0.5)',
        linewidth=1
    )
    fig.update_yaxes(
        gridcolor='rgba(55, 65, 81, 0.2)',
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor='rgba(55, 65, 81, 0.5)',
        linewidth=1
    )
    
    # Add titles with professional styling
    fig.add_annotation(
        text="<b>Price Chart with Trading Signals</b>",
        xref="paper", yref="paper",
        x=0, y=1.05,
        xanchor='left', yanchor='bottom',
        showarrow=False,
        font=dict(size=16, color='#f3f4f6', family='Inter'),
        row=1, col=1
    )
    
    return fig


# Platform Header
st.markdown("""
<div style="margin: -80px -100px 40px -100px; padding: 20px 40px; 
     background: linear-gradient(135deg, rgba(17, 24, 39, 0.98) 0%, rgba(31, 41, 55, 0.98) 100%);
     backdrop-filter: blur(20px);
     border-bottom: 1px solid rgba(16, 185, 129, 0.2);
     box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="font-size: 36px;">📈</div>
            <div>
                <h1 style="margin: 0; font-size: 28px; font-weight: 800; 
                    background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    letter-spacing: -0.5px;">QuanTrade Pro</h1>
                <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 13px; font-weight: 500;">
                    Professional Algorithmic Trading Platform
                </p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="padding: 8px 16px; background: rgba(16, 185, 129, 0.15); 
                 border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%;
                         animation: pulse 2s infinite;"></div>
                    <span style="color: #10b981; font-size: 13px; font-weight: 700;">LIVE TRADING</span>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main tabs with enhanced icons
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portfolio Performance Metrics",
    "📈 Price Chart & Trading Signals",
    "💰 Portfolio Equity Curve",
    "📋 Trade Execution History & Performance"
])


with st.sidebar:
    st.markdown("## 📁 Data Configuration")
    st.markdown("*Select your market data source*")
    
    uploaded_data = st.file_uploader(
        "Upload Data File (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload a single market data file"
    )
    data_path = st.text_input(
        "Data Path (optional)",
        value=r"C:\Users\JAIVAL CHAUHAN\Desktop\AlgoTrading\futures_minute.xlsx",
        help="Path to CSV file or folder containing OHLCV data"
    )
    
    uploaded_benchmark = st.file_uploader(
        "Upload Benchmark (Optional)",
        type=["csv", "xlsx", "xls"],
        help="Upload a benchmark file for beta calculation"
    )
    benchmark_path = st.text_input(
        "Benchmark Path (optional)",
        value="",
        help="Benchmark data for beta calculation"
    )
    
    st.markdown("---")
    
    use_optimized = st.checkbox(
        "📊 Use Optimized Parameters", 
        value=False,
        help="Load parameters from optimization results"
    )
    opt_params_file = None
    if use_optimized:
        opt_params_file = st.text_input(
            "Params File", 
            value="optimized_params.json",
            help="JSON file with optimized parameters"
        )

    st.markdown("## 💰 Risk Management")
    st.markdown("*Configure your risk parameters*")
    
    initial_capital = st.number_input(
        "Initial Capital ($)",
        value=100000.0,
        step=10000.0,
        format="%.2f",
        help="Starting portfolio value"
    )
    
    risk_per_trade = st.number_input(
        "Risk Per Trade (%)",
        value=1.0,
        min_value=0.1,
        max_value=5.0,
        step=0.1,
        format="%.1f",
        help="Percentage of equity to risk per trade"
    ) / 100.0
    
    col_sl, col_tp = st.columns(2)
    with col_sl:
        stop_loss_atr = st.number_input(
            "Stop-Loss (ATR)",
            value=2.0,
            min_value=0.5,
            max_value=5.0,
            step=0.5,
            help="ATR multiplier for stop-loss"
        )
    
    with col_tp:
        take_profit_atr = st.number_input(
            "Take-Profit (ATR)",
            value=3.0,
            min_value=1.0,
            max_value=10.0,
            step=0.5,
            help="ATR multiplier for take-profit"
        )
    
    max_open_trades = st.number_input(
        "Max Open Trades",
        value=1,
        min_value=1,
        max_value=10,
        step=1,
        help="Maximum concurrent positions"
    )
    
    max_drawdown_limit = st.number_input(
        "Max Drawdown Limit (%)",
        value=30.0,
        min_value=5.0,
        max_value=50.0,
        step=5.0,
        format="%.0f",
        help="Circuit breaker threshold"
    ) / 100.0

    st.markdown("---")
    run_btn = st.button("🚀 RUN BACKTEST", use_container_width=True, type="primary")

if run_btn:
    with st.spinner("⚙️ Executing backtest... Please wait"):
        try:
            temp_files = []
            
            if uploaded_data is not None:
                suffix = Path(uploaded_data.name).suffix
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(uploaded_data.read())
                temp_file.flush()
                temp_files.append(temp_file.name)
                data_source = temp_file.name
            else:
                data_source = data_path
            
            if uploaded_benchmark is not None:
                bench_suffix = Path(uploaded_benchmark.name).suffix
                bench_temp = tempfile.NamedTemporaryFile(delete=False, suffix=bench_suffix)
                bench_temp.write(uploaded_benchmark.read())
                bench_temp.flush()
                temp_files.append(bench_temp.name)
                benchmark_source = bench_temp.name
            else:
                benchmark_source = benchmark_path
            
            df = load_data(data_source)
            
            if use_optimized and opt_params_file:
                try:
                    with open(opt_params_file) as f:
                        opt_params = json.load(f)
                    st.info(f"✅ Using optimized parameters from {opt_params_file}")
                    df = build_features_with_params(df, opt_params)
                    df = generate_signals_with_params(df, opt_params)
                    bt_df, trades = backtest(
                        df,
                        initial_capital=opt_params.get("initial_capital", initial_capital),
                        risk_per_trade=opt_params.get("risk_per_trade", risk_per_trade),
                        stop_loss_atr=opt_params.get("stop_loss_atr", stop_loss_atr),
                        take_profit_atr=opt_params.get("take_profit_atr", take_profit_atr),
                        max_open_trades=int(opt_params.get("max_open_trades", max_open_trades)),
                        max_drawdown_limit=opt_params.get("max_drawdown_limit", max_drawdown_limit),
                    )
                except FileNotFoundError:
                    st.error(f"❌ Optimized params file not found: {opt_params_file}")
                    st.stop()
            else:
                df = build_features(df)

                if benchmark_source:
                    benchmark_df = load_data(benchmark_source)
                    benchmark_df = add_return_features(benchmark_df)
                    df["beta"] = beta(df["returns"], benchmark_df["returns"], 50)
                else:
                    df["beta"] = pd.NA

                df = generate_signals(df)
                bt_df, trades = backtest(
                    df,
                    initial_capital=initial_capital,
                    risk_per_trade=risk_per_trade,
                    stop_loss_atr=stop_loss_atr,
                    take_profit_atr=take_profit_atr,
                    max_open_trades=int(max_open_trades),
                    max_drawdown_limit=max_drawdown_limit,
                )
            
            metrics = compute_metrics(bt_df, trades)
            current_regime = int(bt_df['hmm_regime'].iloc[-1]) if 'hmm_regime' in bt_df.columns else 0
            regime_badge, regime_desc = get_regime_badge(current_regime)
            
            st.session_state["bt_df"] = bt_df
            st.session_state["trades"] = trades
            st.session_state["metrics"] = metrics
            st.session_state["regime_badge"] = regime_badge
            st.session_state["regime_desc"] = regime_desc
            
            st.success("✅ Backtest completed successfully! Open the tabs to review results.")
        except Exception as e:
            st.error(f"❌ Error during backtest execution: {str(e)}")
            st.exception(e)


def get_backtest_state():
    bt_df = st.session_state.get("bt_df")
    trades = st.session_state.get("trades")
    metrics = st.session_state.get("metrics")
    regime_badge = st.session_state.get("regime_badge")
    regime_desc = st.session_state.get("regime_desc")
    return bt_df, trades, metrics, regime_badge, regime_desc


with tab1:
    st.markdown("### 📊 Portfolio Performance Metrics")
    bt_df, trades, metrics, regime_badge, regime_desc = get_backtest_state()
    if metrics is None:
        st.info("Run a backtest in the Controls tab to view metrics.")
    else:
        total_return = metrics.get('total_return', 0) * 100
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0) * 100
        win_rate = metrics.get('win_rate', 0) * 100
        profit_factor = metrics.get('profit_factor', 0)
        num_trades = metrics.get('num_trades', 0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(create_metric_card("Total Return", f"{total_return:+.2f}%", total_return > 0, icon="💰"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Sharpe Ratio", f"{sharpe:.3f}", sharpe > 1.0, icon="📈"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Max Drawdown", f"{max_dd:.2f}%", max_dd < 15, icon="📉"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("Win Rate", f"{win_rate:.1f}%", win_rate > 50, icon="🎯"), unsafe_allow_html=True)

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.markdown(create_metric_card("Profit Factor", f"{profit_factor:.3f}", profit_factor > 1.5, icon="⚡"), unsafe_allow_html=True)
        with col6:
            st.markdown(create_metric_card("Sortino Ratio", f"{sortino:.3f}", sortino > 1.0, icon="🛡️"), unsafe_allow_html=True)
        with col7:
            st.markdown(create_metric_card("Total Trades", f"{int(num_trades)}", None, icon="🔄"), unsafe_allow_html=True)
        with col8:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🎲 MARKET REGIME</div>
                <div style="margin-top: 12px;">{regime_badge}</div>
                <div style="color: #9ca3af; font-size: 11px; margin-top: 8px; font-weight: 500;">{regime_desc}</div>
            </div>
            """, unsafe_allow_html=True)


with tab2:
    st.markdown("### 📈 Price Chart & Trading Signals")
    bt_df, trades, metrics, _, _ = get_backtest_state()
    if bt_df is None:
        st.info("Run a backtest in the Controls tab to view the chart.")
    else:
        chart = create_enhanced_candlestick(bt_df, trades)
        st.plotly_chart(chart, use_container_width=True)


with tab3:
    st.markdown("### 💰 Portfolio Equity Curve")
    bt_df, trades, metrics, _, _ = get_backtest_state()
    if bt_df is None:
        st.info("Run a backtest in the Controls tab to view the equity curve.")
    else:
        equity_chart = equity_curve_plot(bt_df)
        equity_chart.update_layout(
            template='plotly_dark',
            plot_bgcolor='#0a0e1a',
            paper_bgcolor='rgba(17, 24, 39, 0.95)',
            font=dict(color='#d1d5db', family='Inter'),
            height=450,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        equity_chart.update_xaxes(
            gridcolor='rgba(55, 65, 81, 0.2)',
            showline=True,
            linecolor='rgba(55, 65, 81, 0.5)'
        )
        equity_chart.update_yaxes(
            gridcolor='rgba(55, 65, 81, 0.2)',
            showline=True,
            linecolor='rgba(55, 65, 81, 0.5)'
        )
        st.plotly_chart(equity_chart, use_container_width=True)


with tab4:
    st.markdown("### 📋 Trade Execution History")
    bt_df, trades, metrics, _, _ = get_backtest_state()
    if trades is None:
        st.info("Run a backtest in the Controls tab to view trade history and metrics.")
    else:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("*Detailed trade-by-trade breakdown*")
            if len(trades) > 0:
                trades_data = []
                for i, trade in enumerate(trades, 1):
                    direction_label = "🟢 LONG" if trade.direction == 1 else "🔴 SHORT"
                    pnl_formatted = f"${trade.pnl:,.2f}"
                    return_formatted = f"{trade.return_pct*100:+.2f}%"

                    trades_data.append({
                        '#': i,
                        'Entry': trade.entry_time.strftime('%Y-%m-%d %H:%M'),
                        'Exit': trade.exit_time.strftime('%Y-%m-%d %H:%M'),
                        'Type': direction_label,
                        'Entry $': f"${trade.entry_price:,.2f}",
                        'Exit $': f"${trade.exit_price:,.2f}",
                        'Size': f"{trade.size:.2f}",
                        'P&L': pnl_formatted,
                        'Return': return_formatted
                    })

                trades_df = pd.DataFrame(trades_data)
                st.dataframe(
                    trades_df.tail(25),
                    use_container_width=True,
                    height=500,
                    hide_index=True
                )

                csv = trades_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Trade History",
                    data=csv,
                    file_name="trade_history.csv",
                    mime="text/csv"
                )
            else:
                st.info("ℹ️ No trades executed in this backtest period.")

        with col_right:
            st.markdown("### 📊 Performance Metrics")
            metrics_display = {
                'Total Return': f"{metrics.get('total_return', 0)*100:.2f}%",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
                'Sortino Ratio': f"{metrics.get('sortino_ratio', 0):.3f}",
                'Max Drawdown': f"{metrics.get('max_drawdown', 0)*100:.2f}%",
                'Win Rate': f"{metrics.get('win_rate', 0)*100:.1f}%",
                'Profit Factor': f"{metrics.get('profit_factor', 0):.3f}",
                'Total Trades': f"{int(metrics.get('num_trades', 0))}",
                'Avg Return/Trade': f"{metrics.get('total_return', 0) / max(metrics.get('num_trades', 1), 1) * 100:.3f}%"
            }

            metrics_df = pd.DataFrame([
                {'Metric': k, 'Value': v}
                for k, v in metrics_display.items()
            ])

            st.dataframe(
                metrics_df,
                use_container_width=True,
                height=400,
                hide_index=True
            )

        st.markdown("---")
        with st.expander("🔍 Market Data Snapshot (Latest 50 Bars)", expanded=False):
            display_df = bt_df[[
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'adx', 'atr'
            ]].tail(50)
            st.dataframe(display_df, use_container_width=True, height=400)

