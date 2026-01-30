# Algo-Trading Hackathon Project

## 🎯 Overview
This project is a complete algorithmic trading system developed for the E-Summit 2026 Algo-Trading Hackathon. It implements a comprehensive backtesting framework with:

### ✅ Technical Indicators (13+)
- **Trend:** SMA, EMA, MACD
- **Momentum:** RSI, ROC, Stochastic Oscillator
- **Volatility:** ATR, Bollinger Bands, Standard Deviation
- **Volume:** Volume, OBV (On-Balance Volume), VWAP

### ✅ Risk Management
- Stop-Loss (ATR-based)
- Take-Profit (ATR-based)
- Risk Per Trade (% of capital)
- Max Drawdown Limit
- Max Open Trades

### ✅ Advanced Features
- Returns & Log Returns
- Volatility calculation
- Correlation analysis
- Beta calculation
- Z-score normalization
- **Regime Detection** (Volatility + Hidden Markov Models)
- **Kalman Filters** for price smoothing

### ✅ Performance Metrics
- Total Returns
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- Number of Trades

---

## 📁 Project Structure
```
Algo-Trading-Hackathon/
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load and preprocess CSV data
│   ├── indicators.py         # All technical indicators
│   ├── features.py           # Returns, volatility, correlation, beta, z-score
│   ├── regime.py             # Kalman filter, HMM, regime detection
│   ├── strategy.py           # Signal generation logic
│   ├── backtest.py           # Backtesting engine with risk management
│   ├── metrics.py            # Performance evaluation
│   └── plotting.py           # Visualization functions
├── app.py                    # Streamlit UI Dashboard
├── run_backtest.py           # CLI script for backtesting
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.8+ installed
- Virtual environment (recommended)

### 2. Install Dependencies
Navigate to the project folder and install required packages:
```bash
cd Algo-Trading-Hackathon
pip install -r requirements.txt
```

**Packages installed:**
- pandas (data manipulation)
- numpy (numerical operations)
- plotly (interactive charts)
- streamlit (web dashboard)
- scikit-learn (machine learning utilities)
- hmmlearn (Hidden Markov Models)

---

## 📊 How to Run

### Method 1: Command Line Interface (CLI)
Run backtesting from terminal/command prompt:

```bash
python run_backtest.py --data "../Equity_1min"
```

**Optional: Add benchmark for beta calculation**
```bash
python run_backtest.py --data "../Equity_1min" --benchmark "../futures_1_day/BANKNIFTY_active_futures.csv"
```

**Output:**
- Results saved in `outputs/` folder
- `backtest_results.csv` - Full backtested data with positions
- `metrics.csv` - Performance metrics summary

---

### Method 2: Web Dashboard (Streamlit UI) ⭐ RECOMMENDED
Launch the interactive dashboard:

```bash
streamlit run app.py
```

**Features:**
- 📈 Price chart with trade markers (buy/sell points)
- 📉 Equity curve visualization
- ⚙️ Adjustable risk parameters:
  - Initial Capital
  - Risk Per Trade
  - Stop-Loss ATR multiplier
  - Take-Profit ATR multiplier
  - Max Open Trades
  - Max Drawdown Limit
- 📊 Real-time performance metrics display
- 🔍 Latest feature snapshot table

**Usage:**
1. Update data path in sidebar (default: `../Equity_1min`)
2. Adjust risk parameters as needed
3. Click "Run Backtest"
4. View results and charts

---

## 🧪 Testing Your Setup

After installation, run a quick test:
```bash
python run_backtest.py --data "../Equity_1min" --output "test_outputs"
```

You should see:
- Loading CSV files (warnings about date parsing are normal)
- "Backtest complete." message
- Performance metrics table printed

---

## 🎓 Strategy Logic

The strategy uses a **multi-indicator confirmation approach**:

**LONG Entry:**
- EMA Fast > EMA Slow (trend)
- RSI > 55 (momentum)
- MACD Histogram > 0 (momentum)
- ADX > 20 (strong trend)
- Close > SMA 50 (price above moving average)

**SHORT Entry:**
- EMA Fast < EMA Slow (downtrend)
- RSI < 45 (weak momentum)
- MACD Histogram < 0 (bearish momentum)
- ADX > 20 (strong trend)
- Close < SMA 50 (price below moving average)

**Risk Management:**
- Stop-loss: Entry ± (ATR × stop_loss_multiplier)
- Take-profit: Entry ± (ATR × take_profit_multiplier)
- Position sizing based on risk per trade

---

## 📈 Data Format

The system expects CSV files with these columns:
- `date` - Trading date
- `time` - Trading time
- `symbol` - Instrument symbol
- `open` - Open price
- `high` - High price
- `low` - Low price
- `close` - Close price
- `volume` - Trading volume

---

## 🔧 Customization

### Modify Strategy Logic
Edit [src/strategy.py](src/strategy.py) to change entry/exit conditions.

### Add New Indicators
Add functions to [src/indicators.py](src/indicators.py) and call them in `build_features()`.

### Adjust Risk Parameters
Modify defaults in [app.py](app.py) or [run_backtest.py](run_backtest.py).

---

## 📝 Deliverables for Hackathon

✅ **Trading Strategy Documentation** - See "Strategy Logic" section above  
✅ **Source Code** - All files in `src/` folder  
✅ **Backtesting Results** - Generated in `outputs/` folder  
✅ **UI/UX Interface** - Streamlit dashboard (`app.py`)  

---

## 🏆 Key Features Implemented

| Requirement | Status | Implementation |
|------------|--------|----------------|
| EMA, SMA, RSI, MACD, ADX | ✅ | [indicators.py](src/indicators.py) |
| Stochastic, ROC, Bollinger | ✅ | [indicators.py](src/indicators.py) |
| ATR, Volume, OBV, VWAP | ✅ | [indicators.py](src/indicators.py) |
| Stop-Loss & Take-Profit | ✅ | [backtest.py](src/backtest.py) |
| Risk Management | ✅ | [backtest.py](src/backtest.py) |
| Returns & Volatility | ✅ | [features.py](src/features.py) |
| Correlation & Beta | ✅ | [features.py](src/features.py) |
| Z-score | ✅ | [features.py](src/features.py) |
| Regime Detection | ✅ | [regime.py](src/regime.py) |
| Kalman Filter | ✅ | [regime.py](src/regime.py) |
| HMM | ✅ | [regime.py](src/regime.py) |
| All Metrics | ✅ | [metrics.py](src/metrics.py) |
| Interactive UI | ✅ | [app.py](app.py) |

---

## 🐛 Troubleshooting

**Warning about date parsing:**
- This is normal and doesn't affect results
- Data is correctly parsed using dateutil

**HMM not converging:**
- This is normal for some datasets
- Falls back to volatility-based regime detection

**Module import errors:**
- Run `pip install -r requirements.txt` again
- Ensure you're in the correct directory

---

## 👥 Team & Submission

This project satisfies all requirements of the Algo-Trading Hackathon:
- ✅ Uses provided dataset exclusively
- ✅ Original implementation
- ✅ Clean, modular, well-documented code
- ✅ Functional UI/UX interface
- ✅ Complete backtesting results
- ✅ Performance metrics included

---

## 📧 Support

For questions about the code, refer to inline comments in each module.

---

**Good luck with your hackathon submission! 🚀**
