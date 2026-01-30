# Adaptive Regime-Based Strategy - Quick Start

## What Was Built

A dynamic trading strategy that automatically switches between **3 trading modes** based on market conditions:

1. **BULL TREND** (8% of time) → Aggressive trend-following
2. **HIGH_VOLATILITY** (79% of time) → Mean-reversion (MOST PROFITABLE ✅)
3. **BEAR_TREND** (13% of time) → Defensive/short positions

## How to Use

### Run Adaptive Strategy
```bash
cd "Algo-Trading-Hackathon"
python run_backtest_adaptive.py
```

**Output:**
- Regime distribution report
- Performance metrics by regime
- Interactive plot: `outputs/adaptive_backtest_plot.html`

### Compare All Strategies
```bash
python compare_all_strategies.py
```

**Compares:**
- Original strategy (from initial build)
- Improved strategy (from bug fixes)
- Adaptive strategy (NEW)

## Key Results

### Overall Performance
- **88 trades** executed
- **32.95% win rate** (best of all strategies)
- **-20.43% return** (challenging dataset/period)
- **Sharpe -0.0024** (best risk-adjusted return)

### Best Performing: HIGH_VOLATILITY Regime
- **51.52% win rate** 🎯
- **Profit Factor 1.04** (profitable!)
- **33 trades**, only regime making money (+$596)
- **Strategy**: Buy oversold (RSI<30), sell overbought (RSI>70)

### Underperforming: Trending Regimes
- BULL_TREND: 18% win rate, -$10,953 PnL
- BEAR_TREND: 24% win rate, -$10,072 PnL
- **Issue**: Trends are weak/short-lived in this dataset

## Files Created

| File | Purpose |
|------|---------|
| [src/strategy_adaptive.py](src/strategy_adaptive.py) | Core adaptive strategy logic |
| [src/helpers.py](src/helpers.py) | Feature building utilities |
| [run_backtest_adaptive.py](run_backtest_adaptive.py) | Run adaptive backtest |
| [compare_all_strategies.py](compare_all_strategies.py) | Compare 3 strategies |
| [ADAPTIVE_STRATEGY_SUMMARY.md](ADAPTIVE_STRATEGY_SUMMARY.md) | Full documentation |

## Understanding the Regimes

### Regime 0: BULL_TREND
```
Identification:
- Price > Kalman filtered price
- Low volatility (ATR)
- Strong uptrend (ADX>25, EMA bullish)

Trading Approach:
- Trend-following with EMA crossovers
- Tight stops (2×ATR) + trailing TP (5×ATR)
- Aggressive risk (1.5% per trade)
```

### Regime 1: HIGH_VOLATILITY ⭐ PROFITABLE
```
Identification:
- High volatility (ATR) OR weak trend (ADX<20)
- 79% of market time

Trading Approach:
- Mean reversion: Buy RSI<30, Sell RSI>70
- Wide stops (3×ATR) for breathing room
- Conservative risk (1% per trade)
```

### Regime 2: BEAR_TREND
```
Identification:
- Price < Kalman filtered price
- Strong downtrend (ADX>25, EMA bearish)

Trading Approach:
- Short positions OR cash
- Tight stops (2×ATR) + wider TP (4×ATR)
- Moderate risk (1.2% per trade)
```

## Next Steps to Improve

### Priority 1: Enhance Mean Reversion (Already Profitable)
- Fine-tune RSI thresholds (30/70 → maybe 25/75?)
- Add Bollinger Band %B confirmation
- Implement partial profit-taking
- Test different take-profit levels

### Priority 2: Fix or Disable Trend-Following
Current trending regimes lose money. Options:
1. **Tighten entry filters** (require more confirmation)
2. **Switch to cash** in trending regimes (preserve capital)
3. **Only trade WITH high-vol regime** (focus on what works)
4. **Use higher timeframes** for trend detection (1min may be too noisy)

### Priority 3: Add Risk Controls
- Maximum daily loss limit
- Correlation filters (don't overtrade same pattern)
- Time-based filters (avoid low liquidity hours)
- Volatility-based position sizing

## Comparison Summary

| Metric | Original | Improved | Adaptive | Winner |
|--------|----------|----------|----------|--------|
| **Return** | -19.91% | -19.75% | -20.43% | Improved |
| **Sharpe** | -0.0031 | -0.0038 | **-0.0024** | **Adaptive** |
| **Win Rate** | 28.57% | 24.14% | **32.95%** | **Adaptive** |
| **Profit Factor** | 0.64 | 0.50 | **0.69** | **Adaptive** |
| **Trades** | 84 | 58 | 88 | Varies |

**Key Takeaway**: Adaptive strategy has best risk-adjusted performance and highest quality trades, but all strategies struggle with absolute returns in this dataset/period.

## Technical Details

### Regime Classification Logic
Located in [src/regime.py](src/regime.py) - `classify_market_regime()`:
- Uses Kalman filter for price smoothing
- ATR normalization (ATR / 20-period SMA)
- ADX for trend strength
- EMA crossovers for direction
- 3-bar smoothing to prevent whipsaw

### Dynamic Risk Parameters
Located in [src/strategy_adaptive.py](src/strategy_adaptive.py) - `get_regime_specific_params()`:
- Stop-loss, take-profit, and position size adjusted per regime
- Applied automatically in backtest with `use_regime_params=True`

### Backtesting Engine
Enhanced [src/backtest.py](src/backtest.py) - `Backtest` class:
- New `_run_with_regime_params()` method
- Dynamically queries regime-specific parameters each trade
- Maintains regime information with each position

## Troubleshooting

### "ImportError: cannot import name..."
Make sure you're in the project directory:
```bash
cd "c:\Users\JAIVAL CHAUHAN\Desktop\AlgoTrading\Algo-Trading-Hackathon"
```

### "ValueError: could not convert string to float"
The data format issue - should be fixed in current version with `format="%d-%m-%y %H:%M:%S"` in data_loader.py

### "FutureWarning: Setting an item of incompatible dtype"
Cosmetic warning from pandas, doesn't affect results. Can be safely ignored.

### "Model is not converging" (HMM warning)
Normal behavior - HMM may not converge perfectly but still produces useful regime classification.

## Contact & Support

- Full documentation: [ADAPTIVE_STRATEGY_SUMMARY.md](ADAPTIVE_STRATEGY_SUMMARY.md)
- Original strategy docs: [STRATEGY_DOCUMENTATION.md](STRATEGY_DOCUMENTATION.md)
- Project overview: [README.md](README.md)

## Disclaimer

This is a research/educational implementation. Past performance does not guarantee future results. The adaptive strategy shows improved risk-adjusted metrics but negative absolute returns in backtests. Transaction costs, slippage, and market impact not fully modeled. Always test thoroughly before live trading.
