# Quick Start Guide - Algo-Trading Hackathon

## ⚡ 3-Step Setup

### Step 1: Install Dependencies (1 minute)
```bash
cd Algo-Trading-Hackathon
pip install -r requirements.txt
```

### Step 2: Run Backtest (2 minutes)
```bash
python run_backtest.py --data "../Equity_1min"
```

### Step 3: Launch Dashboard (Recommended)
```bash
streamlit run app.py
```
Then open browser to http://localhost:8501

---

## 📊 Using the Dashboard

1. **Update Data Path** (left sidebar)
   - Default: `../Equity_1min`
   - Or use full path: `C:/Users/.../Equity_1min`

2. **Adjust Risk Parameters** (left sidebar)
   - Initial Capital: Starting portfolio value
   - Risk Per Trade: % of capital risked per trade (default 1%)
   - Stop-Loss ATR: Multiplier for stop distance (default 2.0)
   - Take-Profit ATR: Multiplier for profit target (default 3.0)
   - Max Open Trades: Concurrent positions limit (default 1)
   - Max Drawdown Limit: Circuit breaker threshold (default 30%)

3. **Click "Run Backtest"**

4. **View Results**
   - Price chart with buy/sell markers
   - Performance metrics table
   - Equity curve graph
   - Latest feature values

---

## 🎯 What You Get

### Files Created
- `outputs/backtest_results.csv` - Full trade history
- `outputs/metrics.csv` - Performance summary

### Metrics Explained
- **Total Return:** Overall % gain/loss
- **Sharpe Ratio:** Risk-adjusted return (higher is better)
- **Sortino Ratio:** Downside risk-adjusted return
- **Max Drawdown:** Worst peak-to-trough decline
- **Win Rate:** % of profitable trades
- **Profit Factor:** Gross profit ÷ gross loss
- **Num Trades:** Total trades executed

---

## 🔍 What's Included

✅ **13+ Technical Indicators**  
✅ **Advanced Risk Management**  
✅ **Regime Detection (HMM)**  
✅ **Kalman Filter Smoothing**  
✅ **6 Performance Metrics**  
✅ **Interactive UI Dashboard**  

---

## 🆘 Quick Troubleshooting

**Problem:** Module not found errors  
**Solution:** Run `pip install -r requirements.txt` again

**Problem:** Date parsing warnings  
**Solution:** These are normal and don't affect results

**Problem:** Streamlit won't open  
**Solution:** Manually go to http://localhost:8501 in browser

**Problem:** No trades generated  
**Solution:** Try adjusting RSI thresholds or ADX minimum in strategy.py

---

## 📁 File Structure
```
Algo-Trading-Hackathon/
├── src/              # Core modules
├── app.py            # Dashboard (START HERE)
├── run_backtest.py   # CLI script
├── outputs/          # Results saved here
├── GUIDE.md          # Full documentation
└── STRATEGY_DOCUMENTATION.md  # Strategy details
```

---

## 🚀 Next Steps

1. ✅ Review results in dashboard
2. ✅ Read `STRATEGY_DOCUMENTATION.md` for strategy details
3. ✅ Customize parameters and re-run
4. ✅ Modify `src/strategy.py` for different entry rules
5. ✅ Prepare presentation for Stage 2

---

**Ready for hackathon submission! 🏆**

Need help? Check `GUIDE.md` for detailed documentation.
