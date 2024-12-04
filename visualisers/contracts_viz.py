import streamlit as st
import pandas as pd
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Market
from datetime import datetime
import os
import plotly.graph_objects as go
from collections import deque
import asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading

st.set_page_config(layout="wide")
st.title("Live Options Price Tracker")

CONTRACTS = [
    "O:NKE241220P00071000",
    "O:NKE241220P00072000", 
    "O:NKE241220C00088000",
    "O:NKE241220C00087000"
]

max_streaming_len = 100000

if 'contract_data' not in st.session_state:
    st.session_state.contract_data = {
        contract: {
            'bid_prices': deque(maxlen=max_streaming_len),
            'ask_prices': deque(maxlen=max_streaming_len),
            'timestamps': deque(maxlen=max_streaming_len)
        } for contract in CONTRACTS
    }

chart_placeholder = st.empty()

# Split columns by put and call options
put_col, call_col = st.columns(2)
with put_col:
    st.subheader("Put Options")
put_prices = {contract: put_col.empty() for contract in CONTRACTS if 'P' in contract}
with call_col:
    st.subheader("Call Options")
call_prices = {contract: call_col.empty() for contract in CONTRACTS if 'C' in contract}

def update_chart():
    fig = go.Figure()
    
    colors = {
        CONTRACTS[0]: {'bid': 'red', 'ask': 'lightcoral'},      # Put 71
        CONTRACTS[1]: {'bid': 'darkred', 'ask': 'pink'},        # Put 72
        CONTRACTS[2]: {'bid': 'blue', 'ask': 'lightblue'},      # Call 88
        CONTRACTS[3]: {'bid': 'darkblue', 'ask': 'skyblue'}     # Call 87
    }
    
    for contract, data in st.session_state.contract_data.items():
        option_type = 'Put' if 'P' in contract else 'Call'
        strike = contract[-8:-4]
        if len(data['bid_prices']) > 0:
            fig.add_trace(go.Scatter(
                x=list(data['timestamps']),
                y=list(data['bid_prices']),
                name=f'{option_type} {strike} Bid',
                line=dict(color=colors[contract]['bid'])
            ))
            fig.add_trace(go.Scatter(
                x=list(data['timestamps']),
                y=list(data['ask_prices']),
                name=f'{option_type} {strike} Ask',
                line=dict(color=colors[contract]['ask'])
            ))
    
    fig.update_layout(
        title='Live Options Prices',
        xaxis_title='Time',
        yaxis_title='Price',
        height=600
    )
    
    chart_placeholder.plotly_chart(fig, use_container_width=True)

def handle_msg(msgs):
    for m in msgs:
        contract_symbol = str(m.symbol)
        if contract_symbol in st.session_state.contract_data:
            current_time = datetime.now()
            
            data = st.session_state.contract_data[contract_symbol]
            data['bid_prices'].append(m.bid_price)
            data['ask_prices'].append(m.ask_price)
            data['timestamps'].append(current_time)
            
            strike = contract_symbol[-8:-4]
            if 'P' in contract_symbol:
                put_prices[contract_symbol].metric(
                    f"Strike ${strike}",
                    f"Bid: ${m.bid_price:.2f}",
                    f"Ask: ${m.ask_price:.2f}"
                )
            else:
                call_prices[contract_symbol].metric(
                    f"Strike ${strike}",
                    f"Bid: ${m.bid_price:.2f}",
                    f"Ask: ${m.ask_price:.2f}"
                )
    
    update_chart()

def start_websocket():
    api_key = os.getenv("API_KEY")
    client = WebSocketClient(api_key=api_key, market=Market.Options)
    
    subscription_string = ",".join(f"Q.{contract}" for contract in CONTRACTS)
    client.subscribe(subscription_string)
    
    try:
        client.run(handle_msg)
    except Exception as e:
        st.error(f"WebSocket error: {str(e)}")

if st.sidebar.button("Start Streaming"):
    websocket_thread = threading.Thread(target=start_websocket)
    websocket_thread.daemon = True
    add_script_run_ctx(websocket_thread)
    websocket_thread.start()

update_chart()