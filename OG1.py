import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
import math
import yfinance as yf
import warnings
import pytz
warnings.filterwarnings('ignore')

class IntradayOptionsCalculator:
    """
    Enhanced 0DTE Options Calculator optimized for same-day expiration
    """
    
    def __init__(self):
        self.risk_free_rate = 0.048  # Updated to approximate current 3-month Treasury
        self.dividend_yield = 0.012  # SPY dividend yield (~1.2%)
        self.spy_ticker = yf.Ticker("SPY")
        self.current_price = None
        self.last_update = None
        self.debug_mode = True  # Enable debug prints
        
    def get_current_risk_free_rate(self):
        """Get current 3-month Treasury rate with better error handling"""
        try:
            treasury = yf.Ticker("^IRX")  # 3-month Treasury
            data = treasury.history(period="5d")
            if not data.empty:
                rate = data['Close'].iloc[-1] / 100  # Convert percentage to decimal
                # Validate rate is reasonable (between 0% and 10%)
                if 0 <= rate <= 0.10:
                    if self.debug_mode:
                        print(f"DEBUG: Current 3M Treasury rate: {rate*100:.2f}%")
                    return rate
                else:
                    if self.debug_mode:
                        print(f"DEBUG: Treasury rate {rate*100:.2f}% seems unreasonable, using default")
        except Exception as e:
            if self.debug_mode:
                print(f"DEBUG: Error fetching Treasury rate: {e}, using default {self.risk_free_rate*100:.2f}%")
        
        return self.risk_free_rate
        
    def get_live_spy_price(self):
        """Get current SPY price with caching for performance"""
        current_time = datetime.now()
        
        # Cache price for 15 seconds for 0DTE (more frequent updates)
        if (self.current_price is None or 
            self.last_update is None or 
            (current_time - self.last_update).seconds > 15):
            
            try:
                # Get real-time price data
                spy_data = self.spy_ticker.history(period="1d", interval="1m")
                if not spy_data.empty:
                    self.current_price = spy_data['Close'].iloc[-1]
                    self.last_update = current_time
                    if self.debug_mode:
                        print(f"DEBUG: Updated SPY price: ${self.current_price:.2f}")
                else:
                    # Fallback to daily data if intraday fails
                    spy_data = self.spy_ticker.history(period="2d")
                    self.current_price = spy_data['Close'].iloc[-1]
                    self.last_update = current_time
                    if self.debug_mode:
                        print(f"DEBUG: Using daily SPY price: ${self.current_price:.2f}")
                    
            except Exception as e:
                print(f"Error fetching SPY price: {e}")
                # Use a default price if all else fails
                if self.current_price is None:
                    self.current_price = 450.0  # Approximate SPY level
                    
        return self.current_price
    
    def get_spy_options_chain(self, expiration_date=None):
        """Get live SPY options chain for given expiration"""
        try:
            # Get available expiration dates
            expirations = self.spy_ticker.options
            
            if not expirations:
                raise ValueError("No options data available")
                
            # Use next expiration if not specified
            if expiration_date is None:
                expiration_date = expirations[0]
            elif expiration_date not in expirations:
                print(f"Available expirations: {list(expirations)}")
                expiration_date = expirations[0]
                
            # Get options chain
            options_chain = self.spy_ticker.option_chain(expiration_date)
            
            return {
                'calls': options_chain.calls,
                'puts': options_chain.puts,
                'expiration': expiration_date
            }
            
        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return None
    
    def calculate_implied_volatility_from_market(self, options_df, current_price):
        """Calculate weighted average implied volatility from market options"""
        try:
            # Focus on near-the-money options for better IV estimate
            atm_range = 0.05  # Within 5% of current price
            lower_bound = current_price * (1 - atm_range)
            upper_bound = current_price * (1 + atm_range)
            
            near_money = options_df[
                (options_df['strike'] >= lower_bound) & 
                (options_df['strike'] <= upper_bound) &
                (options_df['impliedVolatility'] > 0) &
                (options_df['bid'] > 0) &  # Ensure liquid options
                (options_df['ask'] > 0) &
                (options_df['volume'] > 10)  # Minimum volume for 0DTE
            ]
            
            if not near_money.empty:
                # Weight by volume and open interest
                weights = near_money['volume'].fillna(0) + near_money['openInterest'].fillna(0)
                if weights.sum() > 0:
                    weighted_iv = (near_money['impliedVolatility'] * weights).sum() / weights.sum()
                else:
                    weighted_iv = near_money['impliedVolatility'].mean()
                
                if self.debug_mode:
                    print(f"DEBUG: Calculated weighted IV: {weighted_iv*100:.1f}% from {len(near_money)} options")
                
                return weighted_iv
            else:
                # Fallback to all options average
                fallback_iv = options_df[
                    (options_df['impliedVolatility'] > 0) &
                    (options_df['volume'] > 0)
                ]['impliedVolatility'].mean()
                if self.debug_mode:
                    print(f"DEBUG: Using fallback IV: {fallback_iv*100:.1f}%")
                return fallback_iv
                
        except Exception as e:
            print(f"Error calculating IV: {e}")
            return 0.2  # Default 20% volatility
    
    def expiration_day_value(self, S, K, option_type):
        """Calculate intrinsic value for expiration day options"""
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    def black_scholes_call_with_div(self, S, K, T, r, sigma, q):
        """Calculate Black-Scholes call option price with dividend yield"""
        # For very short time to expiration, use intrinsic value
        if T <= 0.001:  # Less than ~9 hours
            return max(S - K, 0)
        
        if sigma <= 0:
            return max(S - K, 0)
            
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put_with_div(self, S, K, T, r, sigma, q):
        """Calculate Black-Scholes put option price with dividend yield"""
        # For very short time to expiration, use intrinsic value
        if T <= 0.001:  # Less than ~9 hours
            return max(K - S, 0)
        
        if sigma <= 0:
            return max(K - S, 0)
            
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return put_price
    
    # Keep original methods for backward compatibility
    def black_scholes_call(self, S, K, T, r, sigma):
        return self.black_scholes_call_with_div(S, K, T, r, sigma, 0)
    
    def black_scholes_put(self, S, K, T, r, sigma):
        return self.black_scholes_put_with_div(S, K, T, r, sigma, 0)
    
    def calculate_greeks_with_div(self, S, K, T, r, sigma, q, option_type='call'):
        """Calculate option Greeks with dividend yield - optimized for 0DTE"""
        # For very short expiration, Greeks become binary
        if T <= 0.001:
            if option_type == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
                
            return {
                'delta': delta,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        if sigma <= 0:
            if option_type == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
                
            return {
                'delta': delta,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
            
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta (adjusted for dividends)
        if option_type == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        
        # Gamma (same for calls and puts, adjusted for dividends)
        gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
        
        # Theta (per day, adjusted for dividends) - accelerated for 0DTE
        theta_multiplier = max(1.0, 5.0 * (1 - T * 365))  # Accelerate theta decay on expiration day
        if option_type == 'call':
            theta = ((-S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    + q * S * np.exp(-q * T) * norm.cdf(d1)
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365 * theta_multiplier
        else:
            theta = ((-S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    - q * S * np.exp(-q * T) * norm.cdf(-d1)
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365 * theta_multiplier
        
        # Vega (per 1% volatility change) - reduced for 0DTE
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho (per 1% interest rate change) - minimal for 0DTE
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    # Keep original method for backward compatibility
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        return self.calculate_greeks_with_div(S, K, T, r, sigma, 0, option_type)
    
    def time_to_expiration_calendar(self, expiration_date, current_time=None):
        """Calculate time to expiration using calendar days (standard approach)"""
        if current_time is None:
            current_time = datetime.now()
        
        # Convert to Eastern Time (market timezone)
        eastern = pytz.timezone('US/Eastern')
        
        if isinstance(expiration_date, str):
            # Options expire at 4:00 PM ET on expiration date
            exp_naive = datetime.strptime(expiration_date + " 16:00:00", '%Y-%m-%d %H:%M:%S')
            exp_datetime = eastern.localize(exp_naive)
        else:
            exp_datetime = expiration_date
        
        # Ensure current time is timezone aware
        if current_time.tzinfo is None:
            current_time = eastern.localize(current_time)
        else:
            current_time = current_time.astimezone(eastern)
        
        time_diff = exp_datetime - current_time
        total_seconds = max(time_diff.total_seconds(), 0)
        
        # Convert to years using calendar days
        years = total_seconds / (365.25 * 24 * 3600)
        
        if self.debug_mode:
            hours_remaining = total_seconds / 3600
            print(f"DEBUG: Time to expiration: {years*365:.2f} days ({hours_remaining:.1f} hours)")
        
        return years
    
    def time_to_expiration_intraday(self, expiration_date, current_time=None):
        """Original method - keeping for compatibility but now calls calendar method"""
        return self.time_to_expiration_calendar(expiration_date, current_time)
    
    def get_market_price(self, bid, ask, last_price):
        """Get best estimate of current market price"""
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            # If last price is reasonable relative to bid/ask, average it with mid
            if bid <= last_price <= ask:
                return (mid + last_price) / 2
            else:
                return mid
        else:
            return last_price
    
    def calculate_pin_risk(self, current_price, options_df):
        """Calculate pin risk for 0DTE options"""
        # Find strikes with highest open interest
        oi_by_strike = {}
        for _, row in options_df.iterrows():
            strike = row['strike']
            oi = row.get('openInterest', 0)
            if strike in oi_by_strike:
                oi_by_strike[strike] += oi
            else:
                oi_by_strike[strike] = oi
        
        # Find the strike with maximum OI
        if oi_by_strike:
            max_oi_strike = max(oi_by_strike.items(), key=lambda x: x[1])
            distance = abs(current_price - max_oi_strike[0])
            
            return {
                'pin_strike': max_oi_strike[0],
                'total_oi': max_oi_strike[1],
                'distance': distance,
                'risk_level': 'HIGH' if distance < 1.0 else 'MEDIUM' if distance < 2.0 else 'LOW'
            }
        
        return None

    def live_options_analysis(self, expiration_date=None, strike_range=None):
        """Enhanced 0DTE options analysis for SPY"""
        print("=== SPY 0DTE Options Analysis ===")
        
        # Get current parameters
        current_price = self.get_live_spy_price()
        current_rf_rate = self.get_current_risk_free_rate()
        
        print(f"Current SPY Price: ${current_price:.2f}")
        print(f"Risk-free Rate: {current_rf_rate*100:.2f}%")
        print(f"Dividend Yield: {self.dividend_yield*100:.2f}%")
        print(f"Last Updated: {self.last_update.strftime('%H:%M:%S')}")
        
        # Get options chain
        options_data = self.get_spy_options_chain(expiration_date)
        if not options_data:
            print("Failed to fetch options data")
            return None
            
        calls_df = options_data['calls']
        puts_df = options_data['puts']
        exp_date = options_data['expiration']
        
        print(f"Expiration Date: {exp_date}")
        
        # Calculate time to expiration
        T = self.time_to_expiration_calendar(exp_date)
        
        # Check if it's truly 0DTE
        hours_remaining = T * 365 * 24
        is_0dte = hours_remaining < 8  # Less than 8 hours
        
        if is_0dte:
            print(f"⚠️  0DTE WARNING: Only {hours_remaining:.1f} hours remaining - using intrinsic value pricing")
        
        # Calculate pin risk
        all_options = pd.concat([calls_df, puts_df])
        pin_info = self.calculate_pin_risk(current_price, all_options)
        if pin_info:
            print(f"📍 PIN RISK: Strike ${pin_info['pin_strike']:.0f} (OI: {pin_info['total_oi']:,}) - "
                  f"Distance: ${pin_info['distance']:.2f} - Risk: {pin_info['risk_level']}")
        
        # Get market implied volatility
        call_iv = self.calculate_implied_volatility_from_market(calls_df, current_price)
        put_iv = self.calculate_implied_volatility_from_market(puts_df, current_price)
        avg_iv = (call_iv + put_iv) / 2
        
        print(f"Call IV: {call_iv*100:.1f}%, Put IV: {put_iv*100:.1f}%, Average: {avg_iv*100:.1f}%")
        
        # Filter strikes around current price
        if strike_range is None:
            strike_range = 5 if is_0dte else 20  # Smaller range for 0DTE
            
        min_strike = current_price - strike_range
        max_strike = current_price + strike_range
        
        # Process calls and puts
        relevant_calls = calls_df[
            (calls_df['strike'] >= min_strike) & 
            (calls_df['strike'] <= max_strike)
        ].copy()
        
        relevant_puts = puts_df[
            (puts_df['strike'] >= min_strike) & 
            (puts_df['strike'] <= max_strike)
        ].copy()
        
        # Calculate theoretical prices and compare to market
        analysis_data = []
        
        for _, call_row in relevant_calls.iterrows():
            strike = call_row['strike']
            call_bid = call_row['bid']
            call_ask = call_row['ask']
            call_last = call_row['lastPrice']
            call_volume = call_row['volume']
            call_oi = call_row['openInterest']
            call_iv_individual = call_row['impliedVolatility'] if call_row['impliedVolatility'] > 0 else call_iv
            
            call_market_price = self.get_market_price(call_bid, call_ask, call_last)
            
            # Find corresponding put
            put_row = relevant_puts[relevant_puts['strike'] == strike]
            if put_row.empty:
                continue
                
            put_row = put_row.iloc[0]
            put_bid = put_row['bid']
            put_ask = put_row['ask']
            put_last = put_row['lastPrice']
            put_volume = put_row['volume']
            put_oi = put_row['openInterest']
            put_iv_individual = put_row['impliedVolatility'] if put_row['impliedVolatility'] > 0 else put_iv
            
            put_market_price = self.get_market_price(put_bid, put_ask, put_last)
            
            # For 0DTE, use intrinsic value, otherwise Black-Scholes
            if is_0dte:
                theo_call = self.expiration_day_value(current_price, strike, 'call')
                theo_put = self.expiration_day_value(current_price, strike, 'put')
            else:
                theo_call = self.black_scholes_call_with_div(current_price, strike, T, current_rf_rate, call_iv_individual, self.dividend_yield)
                theo_put = self.black_scholes_put_with_div(current_price, strike, T, current_rf_rate, put_iv_individual, self.dividend_yield)
            
            # Calculate Greeks
            call_greeks = self.calculate_greeks_with_div(current_price, strike, T, current_rf_rate, call_iv_individual, self.dividend_yield, 'call')
            put_greeks = self.calculate_greeks_with_div(current_price, strike, T, current_rf_rate, put_iv_individual, self.dividend_yield, 'put')
            
            # Calculate edge (theoretical vs market)
            call_edge = theo_call - call_market_price
            put_edge = theo_put - put_market_price
            
            analysis_data.append({
                'strike': strike,
                'call_last': call_last,
                'call_bid': call_bid,
                'call_ask': call_ask,
                'call_market': call_market_price,
                'call_theo': theo_call,
                'call_edge': call_edge,
                'call_iv': call_iv_individual,
                'call_delta': call_greeks['delta'],
                'call_gamma': call_greeks['gamma'],
                'call_theta': call_greeks['theta'],
                'call_volume': call_volume,
                'call_oi': call_oi,
                'put_last': put_last,
                'put_bid': put_bid,
                'put_ask': put_ask,
                'put_market': put_market_price,
                'put_theo': theo_put,
                'put_edge': put_edge,
                'put_iv': put_iv_individual,
                'put_delta': put_greeks['delta'],
                'put_gamma': put_greeks['gamma'],
                'put_theta': put_greeks['theta'],
                'put_volume': put_volume,
                'put_oi': put_oi
            })
        
        if not analysis_data:
            print("No options data found in the specified range")
            return None
            
        df = pd.DataFrame(analysis_data)
        df = df.sort_values('strike')
        
        # Display formatted results - CALLS
        print(f"\n=== CALLS ({'Intrinsic Value' if is_0dte else 'Black-Scholes'} Pricing) ===")
        call_cols = ['strike', 'call_market', 'call_bid', 'call_ask', 'call_theo', 'call_edge', 
                    'call_iv', 'call_delta', 'call_gamma', 'call_volume', 'call_oi']
        call_display = df[call_cols].copy()
        call_display.columns = ['Strike', 'Market', 'Bid', 'Ask', 'Theo', 'Edge', 'IV%', 'Delta', 'Gamma', 'Vol', 'OI']
        
        for col in ['Market', 'Bid', 'Ask', 'Theo', 'Edge']:
            call_display[col] = call_display[col].round(2)
        call_display['IV%'] = (call_display['IV%'] * 100).round(1)
        for col in ['Delta', 'Gamma']:
            call_display[col] = call_display[col].round(3)
            
        print(call_display.to_string(index=False))
        
        # Display formatted results - PUTS
        print(f"\n=== PUTS ({'Intrinsic Value' if is_0dte else 'Black-Scholes'} Pricing) ===")
        put_cols = ['strike', 'put_market', 'put_bid', 'put_ask', 'put_theo', 'put_edge', 
                   'put_iv', 'put_delta', 'put_gamma', 'put_volume', 'put_oi']
        put_display = df[put_cols].copy()
        put_display.columns = ['Strike', 'Market', 'Bid', 'Ask', 'Theo', 'Edge', 'IV%', 'Delta', 'Gamma', 'Vol', 'OI']
        
        for col in ['Market', 'Bid', 'Ask', 'Theo', 'Edge']:
            put_display[col] = put_display[col].round(2)
        put_display['IV%'] = (put_display['IV%'] * 100).round(1)
        for col in ['Delta', 'Gamma']:
            put_display[col] = put_display[col].round(3)
            
        print(put_display.to_string(index=False))
        
        # Highlight potential trades for 0DTE
        print(f"\n=== 0DTE OPPORTUNITIES ===")
        
        edge_threshold = 0.05 if is_0dte else 0.10  # Lower threshold for 0DTE
        
        call_opportunities = df[(abs(df['call_edge']) > edge_threshold) & (df['call_volume'] > 0)]
        put_opportunities = df[(abs(df['put_edge']) > edge_threshold) & (df['put_volume'] > 0)]
        
        if not call_opportunities.empty:
            print("CALL OPPORTUNITIES:")
            for _, row in call_opportunities.iterrows():
                direction = "BUY" if row['call_edge'] > 0 else "SELL"
                print(f"  {direction} {row['strike']:.0f}C - Edge: ${row['call_edge']:+.2f}, "
                      f"Delta: {row['call_delta']:.3f}, Vol: {row['call_volume']:,.0f}")
        
        if not put_opportunities.empty:
            print("PUT OPPORTUNITIES:")
            for _, row in put_opportunities.iterrows():
                direction = "BUY" if row['put_edge'] > 0 else "SELL"
                print(f"  {direction} {row['strike']:.0f}P - Edge: ${row['put_edge']:+.2f}, "
                      f"Delta: {row['put_delta']:.3f}, Vol: {row['put_volume']:,.0f}")
        
        if call_opportunities.empty and put_opportunities.empty:
            print("No significant opportunities found with current thresholds")
        
        return df
    
    def intraday_position_tracker(self, positions):
        """Enhanced position tracker optimized for 0DTE"""
        current_price = self.get_live_spy_price()
        current_rf_rate = self.get_current_risk_free_rate()
        total_pnl = 0
        total_delta = 0
        
        print(f"\n=== 0DTE POSITION TRACKER (SPY: ${current_price:.2f}) ===")
        print(f"{'Position':<15} {'Qty':<5} {'Entry':<8} {'Current':<8} {'P&L':<8} {'Delta':<8} {'Gamma':<8}")
        print("-" * 70)
        
        for pos in positions:
            option_type = pos['type']  # 'call' or 'put'
            strike = pos['strike']
            quantity = pos['quantity']
            entry_price = pos['entry_price']
            expiration = pos['expiration']
            
            # Calculate time to expiration
            T = self.time_to_expiration_calendar(expiration)
            hours_remaining = T * 365 * 24
            is_0dte = hours_remaining < 8
            
            # Use market IV for current valuation
            current_iv = pos.get('current_iv', 0.1)  # Lower default for 0DTE
            
            # Calculate current theoretical value
            if is_0dte:
                # Use intrinsic value for 0DTE
                current_value = self.expiration_day_value(current_price, strike, option_type)
                # Greeks become binary for 0DTE
                if option_type == 'call':
                    delta = 1.0 if current_price > strike else 0.0
                else:
                    delta = -1.0 if current_price < strike else 0.0
                gamma = 0.0
                theta = 0.0
            else:
                # Use Black-Scholes for longer-dated
                if option_type == 'call':
                    current_value = self.black_scholes_call_with_div(current_price, strike, T, current_rf_rate, current_iv, self.dividend_yield)
                    greeks = self.calculate_greeks_with_div(current_price, strike, T, current_rf_rate, current_iv, self.dividend_yield, 'call')
                else:
                    current_value = self.black_scholes_put_with_div(current_price, strike, T, current_rf_rate, current_iv, self.dividend_yield)
                    greeks = self.calculate_greeks_with_div(current_price, strike, T, current_rf_rate, current_iv, self.dividend_yield, 'put')
                
                delta = greeks['delta']
                gamma = greeks['gamma']
                theta = greeks['theta']
            
            # Calculate P&L
            position_pnl = (current_value - entry_price) * quantity * 100  # Options are per 100 shares
            position_delta = delta * quantity * 100
            position_gamma = gamma * quantity * 100
            
            total_pnl += position_pnl
            total_delta += position_delta
            
            position_name = f"{strike:.0f}{option_type[0].upper()}"
            expiry_note = " [0DTE]" if is_0dte else f" [{hours_remaining:.1f}h]"
            
            print(f"{position_name:<15} {quantity:<5} {entry_price:<8.2f} {current_value:<8.2f} "
                  f"{position_pnl:<8.0f} {position_delta:<8.0f} {position_gamma:<8.3f}")
        
        print("-" * 70)
        print(f"{'TOTAL':<15} {'':<5} {'':<8} {'':<8} {total_pnl:<8.0f} {total_delta:<8.0f}")
        
        return {
            'total_pnl': total_pnl,
            'total_delta': total_delta,
            'current_spy_price': current_price
        }

# Enhanced example usage optimized for 0DTE
def main():
    # Initialize enhanced 0DTE calculator
    calc = IntradayOptionsCalculator()
    
    # Live SPY 0DTE options analysis
    print("Getting live SPY 0DTE options data...")
    try:
        analysis = calc.live_options_analysis(strike_range=5)  # Smaller range for 0DTE
    except Exception as e:
        print(f"Error in analysis: {e}")
        print("Make sure you have yfinance and pytz installed: pip install yfinance pytz")
        return
    
    # Example 0DTE position tracking with current date
    print("\n" + "="*80)
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    sample_positions = [
        {
            'type': 'call',
            'strike': 648,
            'quantity': 5,
            'entry_price': 0.50,
            'expiration': current_date,  # Same day expiration
            'current_iv': 0.08
        },
        {
            'type': 'put',
            'strike': 647,
            'quantity': -3,
            'entry_price': 0.25,
            'expiration': current_date,  # Same day expiration
            'current_iv': 0.06
        }
    ]
    
    try:
        calc.intraday_position_tracker(sample_positions)
    except Exception as e:
        print(f"Error in position tracking: {e}")

if __name__ == "__main__":
    main()