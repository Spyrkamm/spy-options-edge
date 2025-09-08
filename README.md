SPY 0DTE Options Edge Analyzer

The SPY 0DTE Options Edge Analyzer is a Python script that fetches live SPY options data and compares theoretical fair value pricing against market prices, optimized specifically for same-day expiration (0DTE) options.

It is designed for intraday 0DTE trading and helps traders quantify edge — the premium being paid or received relative to model-based fair value when options have less than 8 hours until expiration.

The script incorporates live SPY prices, Treasury yields, dividend yield adjustments, and real-time time-to-expiration, automatically switching between Black-Scholes and intrinsic value pricing on expiration day.

✨ Features

Live SPY price tracking with 3M Treasury (risk-free) rate and dividend yield adjustments

Automatic 0DTE detection with a switch to intrinsic value pricing

Pin risk analysis highlighting high open interest strikes likely to experience “pinning”

Real-time expiration clock with dynamic hour/minute updates

Dual pricing models: Black-Scholes for >8 hours to expiration and intrinsic value pricing for <8 hours

Volume-weighted implied volatility analysis with liquidity filters

Edge detection tuned for 0DTE options

Negative edge → overpriced (sell candidate)

Positive edge → underpriced (buy candidate)

Greeks calculation with 0DTE-specific adjustments

Position tracker with 0DTE-aware P&L calculations

Intraday opportunity scanner to highlight mispricings in the final hours of trading

📋 Example Output

Getting live SPY 0DTE options data…
Current SPY Price: 647.84
Risk-free Rate: 3.93%
Dividend Yield: 1.20%
Last Updated: 15:17:31
Expiration Date: 2025-09-08
0DTE Warning: Only 0.7 hours remaining — using intrinsic value pricing
Pin Risk: Strike 648 (OI: 11,545) | Distance: 0.16 | Risk: HIGH

Calls (Intrinsic Value Pricing)

Strike 647.0 | Market: 1.57 | Theoretical: 0.84 | Edge: -0.73 | Delta: 0.964 | Volume: 78,021 | OI: 10,453

Strike 648.0 | Market: 0.72 | Theoretical: 0.00 | Edge: -0.72 | Delta: 0.321 | Volume: 319,137 | OI: 11,545

Opportunities

Sell 647C → Edge -0.73, Delta 0.964, Volume 78,021

Sell 648C → Edge -0.72, Delta 0.321, Volume 319,137

Buy 649P → Edge +0.44, Delta -1.000, Volume 381,235

Buy 650P → Edge +0.61, Delta -1.000, Volume 92,801

📖 What is "Edge" in 0DTE Context?

Edge is defined as:

Market Price – Theoretical Price

For 0DTE options:

Negative edge → the option trades above intrinsic value (potential sell)

Positive edge → the option trades below intrinsic value (potential buy)

Key 0DTE dynamics:

Time decay accelerates rapidly in the final hours

Implied volatility loses relevance

Delta approaches 1.0 (calls) or -1.0 (puts) for ITM options

Pin risk becomes significant near strikes with high open interest

⚙️ Installation

Clone this repository:
git clone https://github.com/Spyrkamm/spy-options-edge.git

Enter the project folder:
cd spy-options-edge

Install dependencies:
pip install -r requirements.txt

Run the script:
python OG1.py

📌 Requirements

Python 3.8+

Dependencies: numpy, pandas, scipy, yfinance, pytz (installed via requirements.txt)

🎯 0DTE Trading Considerations

How 0DTE differs from regular options:

Time decay accelerates in the final 2–4 hours

Liquidity can dry up quickly, widening bid-ask spreads

SPY often “pins” to high open interest strikes

ITM options will be assigned at 4 PM ET

Gamma risk is amplified in the final hour

Best practices:

Monitor pin risk and distance to high OI strikes

Focus on high-volume options for better liquidity

Account for accelerating time decay after 2 PM ET

Consider closing positions before the final 30 minutes

⚠️ Disclaimer

This tool is for educational and research purposes only.
It is not financial advice.

0DTE options trading carries extremely high risk due to:

Rapid time decay

High volatility

Assignment risk

Liquidity issues

Potential for total loss within minutes

Use at your own discretion and never risk more than you can afford to lose.
