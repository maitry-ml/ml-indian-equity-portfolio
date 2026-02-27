# Multi-Dimensional Return Forecasting & Portfolio Construction  
### DA6701 – Data Science & AI in Finance (Assignment 2)

## Project Overview

This project develops a multi-factor machine learning framework to forecast daily equity returns and construct an out-of-sample portfolio across six Indian large-cap stocks:

- RELIANCE  
- HDFCBANK  
- INFY  
- M&M  
- BHARTIARTL  
- HINDUNILVR  

The objective was to integrate structured financial data, macroeconomic indicators, and transformer-based sentiment signals into a robust time-series ML pipeline while eliminating look-ahead bias.

---

## Data Sources

**Market Data (OHLCV)**  
- Daily adjusted prices (Jan 2020 – Dec 2025)  
- Source: Yahoo Finance  

**Fundamental Data**  
- EPS, ROE, Debt/Equity, P/E  
- Quarterly alignment with daily timestamps  
- Source: MoneyControl  

**Macroeconomic Indicators**  
- Gold
- USD/INR  
- India 10Y Bond Yield  
- Crude Oil  
- CPI (lagged)  
- Sources: RBI, Yahoo Finance  

**Sentiment Data**  
- Financial news headlines  
- Transformer-based polarity scoring (FinBERT)  
- Sector index proxy for historical periods  

All macro and sentiment variables were lagged appropriately to eliminate look-ahead bias.

---

## Feature Engineering

- Log returns (stationarity transformation)
- RSI(14)
- Price/SMA20 ratio
- Rolling 20-day volatility
- Return lags (1, 2, 5)
- Log volume change
- Macro & sentiment features

RobustScaler (median/IQR) was fit only on training data to prevent leakage.

---

## Model Architecture

- LightGBM with L1 and L2 regularization  
- Recursive Feature Elimination (Ridge base)  
- 9-fold expanding walk-forward validation  
- No shuffled K-Fold (time-series preserved)

---

## Out-of-Sample Results  
**Forward Test: Oct 2025 – Dec 2025 (60 Trading Days)**

| Strategy | Annual Return | Sharpe | Max Drawdown | Hit Ratio |
|-----------|--------------|--------|--------------|------------|
| Dynamic | 42.75% | 3.87 | -2.04% | 55.0% |
| Static | 20.76% | 1.74 | -2.66% | 51.67% |
| Hybrid | 42.31% | 3.73 | -1.98% | 58.33% |
| Nifty50 | 18.89% | 1.64 | -2.16% | 52.54% |

> Sharpe ratios are elevated due to short evaluation window and favorable market regime.

---

## 🔍 Key Insights

- Macro variables dominate short-term return prediction.
- Fundamentals add limited incremental signal at daily frequency.
- Sentiment improves directional accuracy.
- Excluding low-accuracy stocks improves portfolio stability.
- Walk-forward validation is critical in financial ML.

---

## ⚠️ Disclaimer

This project was developed as part of the course *Data Science & AI in Finance*.  
Results reflect out-of-sample performance over a single quarter and should not be interpreted as long-run expected returns.
