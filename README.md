# spy-options-edge
This Python script fetches live SPY options data and calculates theoretical vs market pricing to identify intraday edges — the premium you are paying (or receiving) relative to model-based fair value.

--------
SPY Options Edge Analyzer

This Python script fetches live SPY options data and calculates theoretical vs market pricing to identify intraday edges — the premium you are paying (or receiving) relative to model-based fair value.

It uses live data along with treasury yields and dividend adjustments to compute implied volatility (IV), theoretical pricing, and potential opportunities for both calls and puts.

Features

Live SPY price tracking with risk-free rate (3M Treasury) and dividend yield adjustments

Calculation of individual implied volatility and weighted average IV

Theoretical pricing of calls and puts

Edge detection: difference between market price and theoretical fair value

Position tracker to monitor active trades

Identification of potential buy/sell opportunities intraday

Example Output
Getting live SPY options data with enhanced pricing...
=== SPY Live Options Analysis (Enhanced) ===
Current SPY Price: $645.53
Risk-free Rate: 3.90%
Dividend Yield: 1.20%
Last Updated: 13:40:22

=== CALLS (Individual IV Pricing) ===
 Strike  Market  Bid   Ask  Theo  Edge  IV%  Delta  Gamma  Theta      Vol    OI
  645.0    1.61 1.60  1.62  0.73 -0.88  9.7  0.703  0.341 -1.851 142557.0 11276
  646.0    0.96 0.95  0.96  0.17 -0.78  8.5  0.305  0.389 -1.638 266137.0 16016
...

=== POTENTIAL OPPORTUNITIES (Individual IV) ===
CALL OPPORTUNITIES:
  SELL 645C - Edge: $-0.88, Delta: 0.703, IV: 9.7%
  SELL 646C - Edge: $-0.78, Delta: 0.305, IV: 8.5%

PUT OPPORTUNITIES:
  BUY 650P - Edge: $0.45, Delta: -1.000, IV: 0.0%
  BUY 651P - Edge: $0.54, Delta: -1.000, IV: 0.0%

What is “Edge”?

In this context, Edge = Market Price – Theoretical Price.

Negative edge: The option is overpriced relative to fair value → better to sell.

Positive edge: The option is underpriced relative to fair value → better to buy.

This helps traders quantify whether they are paying a premium or capturing value when entering intraday trades.

Installation

Clone this repo:

git clone https://github.com/your-Spyrkamm/spy-options-edge.git
cd spy-options-edge


Install dependencies:

pip install -r requirements.txt


Run the script:

python OG.py
