spy-options-edge
This Python script fetches live SPY options data and calculates theoretical vs market pricing to identify intraday edges — optimized specifically for 0DTE (same-day expiration) options trading.

SPY 0DTE Options Edge Analyzer
The SPY 0DTE Options Edge Analyzer is a Python script that fetches live SPY options data and compares theoretical fair value pricing against market prices, with specialized handling for same-day expiration (0DTE) options.
It is designed for intraday 0DTE trading, helping traders quantify edge — the premium you are paying (or receiving) relative to model-based fair value when options have less than 8 hours until expiration.
The script incorporates live data, Treasury yields, dividend yield adjustments, and real-time time-to-expiration (down to the minute) with automatic switching between Black-Scholes and intrinsic value pricing for expiration day.

✨ Features

📈 Live SPY price tracking with 3M Treasury (risk-free) rate and dividend yield adjustments
⚠️ 0DTE Detection: Automatically detects same-day expiration and switches to intrinsic value pricing
📍 Pin Risk Analysis: Identifies high open interest strikes that may cause "pinning" behavior
⏱️ Real-time expiration clock: time-to-expiration updates dynamically with hour/minute precision
🧮 Dual Pricing Models:

Black-Scholes for options with >8 hours remaining
Intrinsic value pricing for 0DTE options with <8 hours remaining


💰 Enhanced IV analysis: Volume-weighted implied volatility with liquidity filters
🔎 Edge detection optimized for 0DTE: Lower thresholds and faster price updates

Negative edge → Option overpriced (potential sell candidate)
Positive edge → Option underpriced (potential buy candidate)


📊 Greeks calculation with 0DTE adjustments (binary deltas, accelerated theta)
📌 Position tracker with 0DTE-aware P&L calculation
🚀 Intraday opportunity scanner: identifies call/put mispricings during final trading hours


📋 Example Output
