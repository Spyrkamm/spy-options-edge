# spy-options-edge
This Python script fetches live SPY options data and calculates theoretical vs market pricing to identify intraday edges — the premium you are paying (or receiving) relative to model-based fair value.

--------

---

```markdown
# SPY Options Edge Analyzer

The **SPY Options Edge Analyzer** is a Python script that fetches live SPY options data and compares **theoretical fair value pricing** against **market prices**.  
It is designed for **intraday trading**, helping traders quantify **edge** — the premium you are paying (or receiving) relative to model-based fair value.

The script incorporates live data, Treasury yields, dividend yield adjustments, and **real-time time-to-expiration** (down to the minute) to compute:
- Implied volatility (IV)
- Theoretical option pricing (Black–Scholes)
- Position Greeks
- Potential buy/sell opportunities

---

## ✨ Features

- 📈 **Live SPY price tracking** with 3M Treasury (risk-free) rate and dividend yield adjustments  
- ⏱️ **Real-time expiration clock**: time-to-expiration updates dynamically from the moment you run the script  
- 🧮 **Implied Volatility (IV) analysis**: individual strike IV and weighted averages  
- 💰 **Theoretical pricing** for calls and puts vs. market quotes  
- 🔎 **Edge detection**: Market Price – Theoretical Price  
  - Negative edge → Option overpriced (potential sell candidate)  
  - Positive edge → Option underpriced (potential buy candidate)  
- 📊 **Greeks calculation** (Delta, Gamma, Theta)  
- 📌 **Position tracker** to monitor active trades and P&L  
- 🚀 **Intraday opportunity scanner**: identifies call/put mispricings during the trading session  

---

## 📋 Example Output

```

Getting live SPY options data with enhanced pricing...
\=== SPY Live Options Analysis (Enhanced) ===
Current SPY Price: \$645.53
Risk-free Rate: 3.90%
Dividend Yield: 1.20%
Time to Expiration: 0.04 days (0.9 hours)
Last Updated: 13:40:22

\=== CALLS (Individual IV Pricing) ===
Strike  Market  Bid   Ask  Theo  Edge  IV%  Delta  Gamma  Theta      Vol    OI
645.0    1.61 1.60  1.62  0.73 -0.88  9.7  0.703  0.341 -1.851 142557.0 11276
646.0    0.96 0.95  0.96  0.17 -0.78  8.5  0.305  0.389 -1.638 266137.0 16016
...

\=== POTENTIAL OPPORTUNITIES (Individual IV) ===
CALL OPPORTUNITIES:
SELL 645C - Edge: \$-0.88, Delta: 0.703, IV: 9.7%
SELL 646C - Edge: \$-0.78, Delta: 0.305, IV: 8.5%

PUT OPPORTUNITIES:
BUY 650P - Edge: \$0.45, Delta: -1.000, IV: 0.0%
BUY 651P - Edge: \$0.54, Delta: -1.000, IV: 0.0%

````

---

## 📖 What is “Edge”?

**Edge = Market Price – Theoretical Price**

- **Negative edge** → The option is *overpriced* vs. fair value → better to **sell**.  
- **Positive edge** → The option is *underpriced* vs. fair value → better to **buy**.  

This is especially useful for **intraday traders** looking to capture small, short-term mispricings.

---

## ⚙️ Installation

Clone this repo:

```bash
git clone https://github.com/Spyrkamm/spy-options-edge.git
cd spy-options-edge
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python OG.py
```

---

## 📌 Requirements

* Python 3.8+
* Dependencies: `numpy`, `pandas`, `scipy`, `yfinance`, `pytz`
  (installed automatically with `requirements.txt`)

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**.
It is **not financial advice**. Options trading carries significant risk.
Use at your own discretion.

```

---




