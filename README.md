# 📈 Multi-Dimensional Return Forecasting & Portfolio Construction  
**DA6701 – Data Science & AI in Finance (Assignment 2)**

---

## 🚀 Project Overview

This project builds a **multi-factor machine learning framework** to forecast daily equity returns and construct an out-of-sample portfolio across six Indian large-cap stocks:

- RELIANCE  
- HDFCBANK  
- INFY  
- M&M  
- BHARTIARTL  
- HINDUNILVR  

The objective was to integrate:

- Structured financial data  
- Macroeconomic indicators  
- Transformer-based sentiment signals  

into a **robust time-series ML pipeline**, while strictly eliminating look-ahead bias.

---

## 📊 Data Sources

### 1️⃣ Market Data (OHLCV)
- Daily adjusted prices (Jan 2020 – Dec 2025)  
- Source: Yahoo Finance  

### 2️⃣ Fundamental Data
- EPS  
- ROE  
- Debt/Equity  
- P/E  
- Quarterly data aligned to daily timestamps  
- Source: MoneyControl  

### 3️⃣ Macroeconomic Indicators
- Gold  
- USD/INR  
- India 10Y Bond Yield  
- Crude Oil  
- CPI (lagged)  
- Sources: RBI, Yahoo Finance  

### 4️⃣ Sentiment Data
- Financial news headlines  
- Transformer-based polarity scoring (FinBERT)  
- Sector index proxy for historical sentiment periods  

All macro and sentiment variables were **lagged appropriately** to eliminate look-ahead bias.

---

## 🛠 Feature Engineering

- Log returns (stationarity transformation)  
- RSI(14)  
- Price/SMA20 ratio  
- Rolling 20-day volatility  
- Return lags (1, 2, 5)  
- Log volume change  
- Macro & sentiment features  

### Scaling
- RobustScaler (median/IQR)  
- Fit only on training data to prevent leakage  

---

## 🤖 Model Architecture

- LightGBM with L1 and L2 regularization  
- Recursive Feature Elimination (Ridge base estimator)  
- 9-fold expanding walk-forward validation  
- No shuffled K-Fold (time-series structure preserved)  

---

## 📈 Out-of-Sample Performance

**Forward Test Window:** Oct 2025 – Dec 2025 (60 trading days)

| Strategy  | Annual Return | Sharpe | Max Drawdown | Hit Ratio |
|-----------|--------------|--------|--------------|----------|
| Dynamic   | 42.75%      | 3.87   | -2.04%       | 55.0%    |
| Static    | 20.76%      | 1.74   | -2.66%       | 51.67%   |
| Hybrid    | 42.31%      | 3.73   | -1.98%       | 58.33%   |
| Nifty50   | 18.89%      | 1.64   | -2.16%       | 52.54%   |

> Sharpe ratios are elevated due to short evaluation window and favorable market regime.

---

## 🔎 Key Insights

- Macroeconomic variables dominate short-term return prediction  
- Fundamentals add limited incremental signal at daily frequency  
- Sentiment improves directional accuracy  
- Excluding low-accuracy stocks improves portfolio stability  
- Walk-forward validation is critical in financial ML  

---

## 🖥️ How to Run

```bash
pip install -r requirements.txt

indian_largecap_portfolio.ipynb

##Disclaimer

This project was developed as part of course Data Science & AI in Finance.
Results reflect out-of-sample performance over a single quarter and should not be interpreted as long-run expected returns.
 


 
