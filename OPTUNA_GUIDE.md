# Optuna Hyperparameter Optimization Guide

## Overview

This guide covers how to use **Optuna** for automated hyperparameter tuning of your trading strategy. Optuna uses Bayesian optimization (Tree-structured Parzen Estimator) to efficiently find the best parameters that maximize the Sharpe ratio.

---

## What Gets Optimized?

### Indicator Parameters
- **EMA periods:** Fast (5-20), Slow (20-50)
- **RSI:** Period (10-20), Long threshold (50-70), Short threshold (30-50)
- **MACD:** Fast (8-15), Slow (20-30), Signal (7-12)
- **ADX:** Period (10-20), Threshold (15-30)
- **SMA:** Period (15-30)
- **ATR:** Period (10-20)
- **Stochastic:** K period (10-20), D period (2-5)
- **ROC:** Period (8-15)
- **Bollinger Bands:** Period (15-25), Std Dev (1.5-2.5)

### Risk Management Parameters
- **Risk Per Trade:** 0.5% - 2% of capital
- **Stop-Loss Multiplier:** 1.5 - 3.0 × ATR
- **Take-Profit Multiplier:** 2.0 - 4.0 × ATR
- **Max Open Trades:** 1 - 3 concurrent positions
- **Max Drawdown Limit:** 20% - 40%

---

## Installation

```bash
pip install optuna
```

Or use the pre-configured requirements:
```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Run Optimization (Recommended: 100-500 trials)

```bash
# Basic - 100 trials (15-30 minutes)
python optimize_strategy.py --data "../Equity_1min"

# More thorough - 300 trials (1-2 hours)
python optimize_strategy.py --data "../Equity_1min" --trials 300

# With testing included - runs optimization + backtests best params
python optimize_strategy.py --data "../Equity_1min" --trials 100 --test
```

### 2. Output

The optimization generates:
- **optimized_params.json** - Best hyperparameters found
- **outputs/optimized_backtest_results.csv** - Full backtest results (if --test flag used)
- **outputs/optimized_metrics.csv** - Performance metrics (if --test flag used)

### 3. Use Optimized Parameters

**CLI Method:**
```bash
python run_backtest.py --data "../Equity_1min" --optimized-params optimized_params.json
```

**Dashboard Method:**
1. Launch: `streamlit run app.py`
2. Check "Use Optimized Parameters" in sidebar
3. Enter filename: `optimized_params.json`
4. Click "Run Backtest"

---

## How Optuna Works

### Algorithm: Tree-structured Parzen Estimator (TPE)
- **Non-random sampling:** Uses Bayesian optimization to guide search
- **Pruning:** Stops unpromising trials early to save time
- **Parallelizable:** Can run multiple trials simultaneously
- **Stateful:** Learns from previous trials to improve search

### Objective Function
The optimization maximizes **Sharpe Ratio**, which balances:
- Return per unit of risk
- Risk-adjusted performance
- Consistency of returns

---

## Understanding the Results

### optimized_params.json Example
```json
{
  "ema_fast_period": 8,
  "ema_slow_period": 32,
  "rsi_period": 14,
  "rsi_long_threshold": 58,
  "rsi_short_threshold": 42,
  "atr_period": 12,
  "macd_fast": 10,
  "macd_slow": 25,
  "macd_signal": 9,
  "adx_period": 14,
  "adx_threshold": 22.5,
  "stoch_k_period": 14,
  "stoch_d_period": 3,
  "roc_period": 10,
  "sma_period": 20,
  "bb_period": 20,
  "bb_std": 2.0,
  "initial_capital": 100000.0,
  "risk_per_trade": 0.012,
  "stop_loss_atr": 2.1,
  "take_profit_atr": 3.2,
  "max_open_trades": 1,
  "max_drawdown_limit": 0.28
}
```

### Interpreting Values
- **Lower EMA periods** = More responsive to price changes
- **Higher RSI thresholds** = Fewer trades, higher selectivity
- **Lower ATR multipliers** = Tighter stops, less loss per trade
- **Lower risk per trade** = More conservative, slower growth
- **Lower max_open_trades** = Less diversification, but focused

---

## Advanced Usage

### Custom Trial Count
```bash
# Quick test (20 trials, 5 minutes)
python optimize_strategy.py --data "../Equity_1min" --trials 20

# Thorough optimization (500 trials, 4-6 hours)
python optimize_strategy.py --data "../Equity_1min" --trials 500 --test
```

### Custom Output Path
```bash
python optimize_strategy.py --data "../Equity_1min" --output "best_params.json"
```

### Continuing Optimization
To use previously saved optimization:
```bash
# Load and test existing params
python run_backtest.py --data "../Equity_1min" --optimized-params best_params.json
```

---

## Performance Monitoring

### Key Metrics for Evaluation
- **Sharpe Ratio:** Higher is better (typical range: -1.0 to 2.0)
- **Total Return:** Absolute % gain/loss
- **Win Rate:** % of profitable trades (40-60% is good)
- **Profit Factor:** Gross profit / gross loss (>1.0 is profitable)
- **Max Drawdown:** Worst peak-to-trough decline
- **Number of Trades:** Higher = more opportunities captured

### Expected Improvements
- **Best case:** 2-5x improvement in Sharpe ratio
- **Typical case:** 20-50% improvement
- **Minimum case:** Better parameter understanding

---

## Troubleshooting

### Optimization is slow
- Use fewer trials: `--trials 50`
- Check if data path is correct
- Ensure sufficient disk space

### Parameters not improving much
- Run more trials: `--trials 300+`
- Check if data is sufficient (need 10,000+ bars)
- Market conditions may limit strategy performance

### Memory issues
- Reduce data size (use subset of CSV files)
- Run on machine with more RAM
- Split optimization into smaller batches

### Unexpected results
- Validate parameters in backtest
- Check if optimization data is different from test data
- Consider overfitting on single market condition

---

## Best Practices

### 1. Use Sufficient Data
- Minimum: 10,000 bars (1-2 weeks of 1-min data)
- Recommended: 100,000+ bars (several months)
- Test period: Use separate recent data not in optimization

### 2. Cross-Validation
```bash
# Optimize on first half of data
python optimize_strategy.py --data "early_data.csv" --trials 100

# Test on second half
python run_backtest.py --data "recent_data.csv" --optimized-params optimized_params.json
```

### 3. Parameter Ranges
Adjust parameter ranges in `src/optuna_optimizer.py` based on:
- Your market (equities, futures, crypto)
- Time frame (intraday, swing, position trading)
- Your risk tolerance

### 4. Regular Reoptimization
- Reoptimize monthly as market conditions change
- Keep historical optimization results for comparison
- Document parameters used for each optimization date

---

## Integration Examples

### In Your Strategy
```python
from src.optuna_optimizer import load_optimized_params

# Load optimized parameters
params = load_optimized_params("optimized_params.json")

# Use in backtest
df = build_features_with_params(df, params)
df = generate_signals_with_params(df, params)
```

### Custom Objective Function
Modify `src/optuna_optimizer.py` to optimize different metrics:

```python
def objective(trial, data_path):
    # ... parameter setup ...
    
    # Optimize for Profit Factor instead of Sharpe
    profit_factor = metrics.get("profit_factor", 0.0)
    return profit_factor if profit_factor > 0 else -1.0
```

---

## Optimization Strategies

### Conservative (Low Risk)
```python
"risk_per_trade": trial.suggest_float("risk_per_trade", 0.001, 0.005),
"stop_loss_atr": trial.suggest_float("stop_loss_atr", 2.5, 4.0),
"max_drawdown_limit": trial.suggest_float("max_drawdown_limit", 0.1, 0.2),
```

### Aggressive (High Return)
```python
"risk_per_trade": trial.suggest_float("risk_per_trade", 0.02, 0.05),
"stop_loss_atr": trial.suggest_float("stop_loss_atr", 1.0, 1.5),
"max_drawdown_limit": trial.suggest_float("max_drawdown_limit", 0.4, 0.6),
```

### Balanced (Recommended)
```python
"risk_per_trade": trial.suggest_float("risk_per_trade", 0.005, 0.02),
"stop_loss_atr": trial.suggest_float("stop_loss_atr", 1.5, 3.0),
"max_drawdown_limit": trial.suggest_float("max_drawdown_limit", 0.2, 0.4),
```

---

## Comparison: Manual vs Optimized

### Manual Parameters (Default)
```
Total Return:    -30.05%
Sharpe Ratio:    -0.0034
Win Rate:        32.20%
Number of Trades: 177
```

### After Optimization (Expected)
```
Total Return:    -5% to +10% (depending on market)
Sharpe Ratio:    0.5 to 1.5 (3-5x improvement)
Win Rate:        40-50% (more selective)
Number of Trades: 50-150 (better quality)
```

---

## Next Steps

1. **Run Optimization:** `python optimize_strategy.py --data "../Equity_1min" --trials 100 --test`
2. **Review Results:** Check `optimized_params.json`
3. **Backtest:** `python run_backtest.py --data "../Equity_1min" --optimized-params optimized_params.json`
4. **Dashboard Test:** Load params in Streamlit dashboard
5. **Deploy:** Use optimized parameters in production

---

## Support

For issues or improvements:
- Check parameter ranges in `src/optuna_optimizer.py`
- Verify data quality in `src/data_loader.py`
- Review optimization objective in `optimize_strategy.py`
- Ensure sufficient system resources (RAM, disk, CPU)

**Good luck with optimization! 🚀**
