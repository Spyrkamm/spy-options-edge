SPY 0DTE Options Edge Analyzer

The SPY 0DTE Options Edge Analyzer is a Python script that fetches live SPY options data and compares theoretical fair value pricing against market prices, optimized specifically for same-day expiration (0DTE) options.

It is designed for intraday 0DTE trading, helping traders quantify edge — the premium you are paying or receiving relative to model-based fair value when options have less than 8 hours until expiration.

The script incorporates live SPY prices, Treasury yields, dividend yield adjustments, and real-time time-to-expiration, automatically switching between Black-Scholes and intrinsic value pricing for expiration day.

✨ Features

📈 Live SPY price tracking with 3M Treasury (risk-free) rate and dividend yield adjustments

⚠️ 0DTE Detection: Automatically switches to intrinsic value pricing for same-day expiration

📍 Pin Risk Analysis: Highlights high open interest strikes likely to experience "pinning"

⏱️ Real-time Expiration Clock: Updates dynamically with hour/minute precision

🧮 Dual Pricing Models:

Black-Scholes for options with >8 hours remaining

Intrinsic value pricing for options with <8 hours remaining

💰 Enhanced IV Analysis: Volume-weighted implied volatility with liquidity filters

🔎 Edge Detection for 0DTE:

Negative edge → Option overpriced (sell candidate)

Positive edge → Option underpriced (buy candidate)

📊 Greeks Calculation: Includes 0DTE adjustments (binary deltas, accelerated theta)

📌 Position Tracker: 0DTE-aware P&L calculations

🚀 Intraday Opportunity Scanner: Identifies mispricings in final trading hours

📋 Example Output
