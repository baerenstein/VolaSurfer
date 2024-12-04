import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from polygon import WebSocketClient, RESTClient
from polygon.websocket.models import WebSocketMessage, Market
from datetime import datetime, timedelta
import plotly.graph_objects as go
from collections import deque
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx
import os
from dataclasses import dataclass
from typing import Dict, List, Set
from models.BlackScholesSolver import BlackScholesIV

@dataclass
class OptionData:
    strike: float
    maturity: datetime
    contract_type: str
    ticker: str
    bid_prices: deque
    ask_prices: deque
    timestamps: deque
    implied_vols: deque

class VolatilitySurfaceStreamer:
    def __init__(
        self,
        underlying: str,
        num_strikes: int = 2,  # strikes above and below ATM
        days_to_expiration: List[int] = [7, 14, 30, 45],  # target maturities
        max_streaming_len: int = 1000,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0
    ):
        self.underlying = underlying
        self.num_strikes = num_strikes
        self.target_maturities = days_to_expiration
        self.max_streaming_len = max_streaming_len
        self.rf_rate = risk_free_rate
        self.div_yield = dividend_yield
        self.api_key = os.getenv("API_KEY")
        self.rest_client = RESTClient(self.api_key)
        self.contracts: Dict[str, OptionData] = {}
        self.bs_solver = BlackScholesIV(
        rate=self.rf_rate,
        dividend=self.div_yield
    )
        
    def fetch_last_price(self):
        """Fetch previous day's closing price"""
        try:
            prev_close_url = f"https://api.polygon.io/v2/aggs/ticker/{self.underlying}/prev?adjusted=true&apiKey={self.api_key}"
            
            response = requests.get(prev_close_url, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            
            data = response.json()
            if "results" not in data:
                raise Exception("No results found in stock data")
                
            prev_day_data = data["results"][0]  # Access first element of results array
            self.last_price = prev_day_data['c']
            
            st.sidebar.metric("Ticker: ", self.underlying)
            st.sidebar.metric("Last Close", f"${self.last_price:.2f}")
            st.sidebar.write(f"Date: {datetime.fromtimestamp(prev_day_data['t']/1000).strftime('%Y-%m-%d')}")
            
            return self.last_price
        except Exception as e:
            st.error(f"Error fetching last price: {str(e)}")
            return None

    def select_contracts(self):
        """Select option contracts around ATM for each target maturity"""
        try:
            selected_contracts = []
            now = datetime.now()
            
            for target_days in self.target_maturities:
                expiry_date = (now + timedelta(days=target_days)).strftime('%Y-%m-%d')
                
                # Fetch contracts for this expiry
                contracts = list(self.rest_client.list_options_contracts(
                    self.underlying,
                    expiration_date_gte=now.strftime('%Y-%m-%d'),
                    expiration_date_lte=expiry_date
                ))
                
                if not contracts:
                    continue
                
                # Group by strike
                strikes = sorted(set(c.strike_price for c in contracts))
                atm_strike = min(strikes, key=lambda x: abs(x - self.last_price))
                atm_index = strikes.index(atm_strike)
                
                # Select strikes around ATM
                selected_indices = range(
                    max(0, atm_index - self.num_strikes),
                    min(len(strikes), atm_index + self.num_strikes + 1)
                )
                selected_strikes = [strikes[i] for i in selected_indices]
                
                # Add both calls and puts
                for c in contracts:
                    if c.strike_price in selected_strikes:
                        selected_contracts.append(c)
                        
                        self.contracts[c.ticker] = OptionData(
                            strike=c.strike_price,
                            maturity=datetime.strptime(c.expiration_date, '%Y-%m-%d'),
                            contract_type=c.contract_type,
                            ticker=c.ticker,
                            bid_prices=deque(maxlen=self.max_streaming_len),
                            ask_prices=deque(maxlen=self.max_streaming_len),
                            timestamps=deque(maxlen=self.max_streaming_len),
                            implied_vols=deque(maxlen=self.max_streaming_len)
                        )
            
            return selected_contracts
            
        except Exception as e:
            st.error(f"Error selecting contracts: {str(e)}")
            return []

    def calculate_implied_vol(self, mid_price: float, strike: float, 
                            maturity: datetime, contract_type: str) -> float:
        """Calculate implied volatility using custom Black-Scholes solver"""
        try:
            days_to_expiry = (maturity - datetime.now()).days
            T = days_to_expiry / 365.0
            
            if T <= 0:
                return None
                
            impl_vol = self.bs_solver.implied_volatility(
                price=mid_price,
                S=self.last_price,
                K=strike,
                T=T,
                option_type=contract_type
            )
            
            return impl_vol
            
        except Exception as e:
            return None

    def handle_msg(self, msgs: List[WebSocketMessage]):
        """Handle incoming websocket messages"""
        for m in msgs:
            contract_symbol = str(m.symbol)
            if contract_symbol in self.contracts:
                current_time = datetime.now()
                contract = self.contracts[contract_symbol]
                
                # Store prices
                contract.bid_prices.append(m.bid_price)
                contract.ask_prices.append(m.ask_price)
                contract.timestamps.append(current_time)
                
                # Calculate mid price and IV
                mid_price = (m.bid_price + m.ask_price) / 2
                impl_vol = self.calculate_implied_vol(
                    mid_price, contract.strike, contract.maturity, 
                    contract.contract_type
                )
                contract.implied_vols.append(impl_vol)

    def create_vol_surface(self) -> go.Figure:
        """Create volatility surface plot"""
        # Collect latest IVs
        data = []
        for contract in self.contracts.values():
            if contract.implied_vols and contract.implied_vols[-1] is not None:
                days_to_expiry = (contract.maturity - datetime.now()).days
                data.append({
                    'strike': contract.strike,
                    'maturity': days_to_expiry,
                    'implied_vol': contract.implied_vols[-1]
                })
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        
        # Create surface grid
        strikes = sorted(df['strike'].unique(), reverse=True)
        maturities = sorted(df['maturity'].unique(), reverse=True)
        
        # Swap K and T in meshgrid to swap axes
        T, K = np.meshgrid(maturities, strikes)
        vol_matrix = np.zeros_like(K)
        
        # Update matrix filling to match swapped axes
        for i, k in enumerate(strikes):
            for j, t in enumerate(maturities):
                matching = df[(df['strike'] == k) & (df['maturity'] == t)]
                if not matching.empty:
                    vol_matrix[i,j] = matching['implied_vol'].iloc[0]
        
        # Create surface plot with swapped axes
        fig = go.Figure(data=[go.Surface(x=T, y=K, z=vol_matrix)])
        
        # Update layout with bird's eye view and swapped axis labels
        fig.update_layout(
            title='Real-time Implied Volatility Surface',
            scene = dict(
                xaxis_title='Time to Maturity (Days)',
                yaxis_title='Strike Price',
                zaxis_title='Implied Volatility',
                # Set camera to bird's eye view
                camera=dict(
                    # eye=dict(x=0, y=0, z=2),
                    # center=dict(x=0, y=0, z=0),
                    # up=dict(x=0, y=1, z=0)
                eye=dict(x=1.5, y=1.5, z=0.8),  # Lowered z value for lower viewing angle
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
                ),
                xaxis=dict(autorange='reversed'),
                yaxis=dict(autorange='reversed')
            ),
            width=800,
            height=800
        )
        
        return fig

st.set_page_config(layout="wide")

def main(ticker):
    # st.set_page_config(layout="wide")
    st.title("Live Option Volatility Surface")
    
    # Initialize volatility surface streamer
    if 'vss' not in st.session_state:
        st.session_state.vss = VolatilitySurfaceStreamer(
            underlying=ticker,
            num_strikes=20,
            days_to_expiration=[7, 14, 30,]
        )
        
        # Fetch initial data
        if st.session_state.vss.fetch_last_price():
            selected_contracts = st.session_state.vss.select_contracts()
            st.sidebar.write(f"Selected {len(selected_contracts)} contracts")
            
            # # Display contract details
            # for c in selected_contracts:
            #     st.sidebar.write(f"{c.ticker}: {c.contract_type.upper()} ${c.strike_price}")
    
    # Create plot placeholders
    surface_plot = st.empty()
    # metrics_cols = st.columns(len(st.session_state.vss.contracts))

    def update_display():
        # Update surface plot
        fig = st.session_state.vss.create_vol_surface()
        if fig:
            surface_plot.plotly_chart(fig)

    if 'websocket_running' not in st.session_state:
        st.session_state.websocket_running = False
    
    if st.sidebar.button("Start Streaming") and not st.session_state.websocket_running:
        st.session_state.websocket_running = True
        
        # Initialize WebSocket
        client = WebSocketClient(
            api_key=st.session_state.vss.api_key,
            market=Market.Options
        )
        
        # Subscribe to contracts
        subscription_string = ",".join(
            f"Q.{ticker}" for ticker in st.session_state.vss.contracts.keys()
        )
        client.subscribe(subscription_string)
        
        # Start WebSocket in a thread
        def run_websocket():
            try:
                client.run(st.session_state.vss.handle_msg)
            except Exception as e:
                st.error(f"WebSocket error: {str(e)}")
                st.session_state.websocket_running = False
        
        websocket_thread = threading.Thread(target=run_websocket)
        websocket_thread.daemon = True
        add_script_run_ctx(websocket_thread)
        websocket_thread.start()
    
    # Add a delay between updates
    if st.session_state.websocket_running:
        update_display()
        time.sleep(10)  # Add a x-second delay between updates
        st.rerun()

if __name__ == "__main__":
    main(ticker="TSLA")
