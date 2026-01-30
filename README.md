# Algo-Trading-Hackathon

This project implements a full research-to-backtest pipeline using the provided dataset, including:
- Trend/Momentum/Volatility indicators (EMA, RSI, ATR, SMA, MACD, ADX, Stochastic, ROC, Bollinger Bands, Standard Deviation, Volume, OBV, VWAP)
- Risk management (Stop-Loss, Take-Profit, Risk Per Trade, Max Drawdown Limit, Max Open Trades)
- Returns/Log Returns/Volatility
- Correlation, Beta, Z-score
- Regime detection (volatility regime + Hidden Markov Model)
- Kalman filter smoothing
- Evaluation metrics (Total Return, Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor)

## Setup
1. Create and activate a Python virtual environment (optional but recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run Backtest (CLI)
```
python run_backtest.py --data "../Equity_1min"
```
Optional benchmark beta:
```
python run_backtest.py --data "../Equity_1min" --benchmark "../futures_1_day/BANKNIFTY_active_futures.csv"
```
Outputs are saved in `outputs/`.

## Run Dashboard (Streamlit UI)
```
streamlit run app.py
```
Then update the sidebar data path to point to your dataset folder or file.









