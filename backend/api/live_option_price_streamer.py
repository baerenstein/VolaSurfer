from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Market
from typing import List
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import os

print("Starting program...")

# Define the option contracts we want to track (without Q prefix for matching)
OPTION_CONTRACTS = [
    "O:NVDA241129C00138000",
    "O:NVDA241129C00137000",
    # "O:NKE241220P00070000"
]

print(f"Monitoring contracts: {OPTION_CONTRACTS}")

api_key = os.getenv("API_KEY")

# Initialize WebSocket client and subscribe to defined contracts
client = WebSocketClient(
    api_key=api_key,
    market=Market.Options
)

# Add Q. prefix only for subscription
subscription_string = ",".join(f"Q.{contract}" for contract in OPTION_CONTRACTS)
print(f"Subscription string: {subscription_string}")
client.subscribe(subscription_string)

# Initialize data structures for each contract (without Q prefix)
contract_data = {
    contract: {
        'bid_prices': [],
        'ask_prices': [],
        'timestamps': []
    } for contract in OPTION_CONTRACTS
}

print("Initializing plot...")

# Create single figure and axis
plt.ion()  # Enable interactive mode
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('NKE Options Prices - Call vs Put (Strike $78)', pad=10)
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.grid(True)

# Configure date formatting for x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

# Define colors for better distinction between contracts
colors = {
    OPTION_CONTRACTS[0]: {'bid': 'blue', 'ask': 'lightblue'},     # Call option
    OPTION_CONTRACTS[1]: {'bid': 'red', 'ask': 'lightcoral'}      # Put option
}

def handle_msg(msgs: List[WebSocketMessage]):
    print(f"Received message batch. Number of messages: {len(msgs)}")
    
    global contract_data
    
    for m in msgs:
        contract_symbol = str(m.symbol)
        print(f"Processing message for symbol: {contract_symbol}")
        
        if contract_symbol in contract_data:
            current_time = datetime.now()
            
            # Print detailed message data
            print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} - Contract: {contract_symbol}, Bid: {m.bid_price}, Ask: {m.ask_price}")
            
            # Update data
            contract_data[contract_symbol]['bid_prices'].append(m.bid_price)
            contract_data[contract_symbol]['ask_prices'].append(m.ask_price)
            contract_data[contract_symbol]['timestamps'].append(current_time)

            # Keep only the latest 100 data points
            if len(contract_data[contract_symbol]['bid_prices']) > 200:
                contract_data[contract_symbol]['bid_prices'].pop(0)
                contract_data[contract_symbol]['ask_prices'].pop(0)
                contract_data[contract_symbol]['timestamps'].pop(0)

            print(f"Data points for {contract_symbol}: {len(contract_data[contract_symbol]['bid_prices'])}")

    try:
        # Clear and redraw the plot
        ax.clear()
        
        # Plot data for each contract
        for contract in OPTION_CONTRACTS:
            data = contract_data[contract]
            option_type = 'Call' if 'C' in contract else 'Put'
            
            if data['bid_prices']:  # Only plot if we have data
                print(f"Plotting {option_type} option data. Points: {len(data['bid_prices'])}")
                
                ax.plot(data['timestamps'], data['bid_prices'],
                        label=f'{option_type} Bid', 
                        color=colors[contract]['bid'],
                        linewidth=2)
                ax.plot(data['timestamps'], data['ask_prices'],
                        label=f'{option_type} Ask', 
                        color=colors[contract]['ask'],
                        linewidth=2)
        
        # Restore plot formatting
        ax.set_title('Live Options Prices', pad=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.legend(loc='upper right')
        ax.grid(True)
        
        # Add some padding to the y-axis
        ymin, ymax = ax.get_ylim()
        padding = (ymax - ymin) * 0.1 if ymin != ymax else 1
        ax.set_ylim(ymin - padding, ymax + padding)

        # Rotate date labels for better readability
        plt.gcf().autofmt_xdate()

        # Draw the plot
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)  # Even shorter pause time
        
        # print("Plot updated successfully")
        
    except Exception as e:
        print(f"Error updating plot: {str(e)}")

print("Starting WebSocket client...")

try:
    client.run(handle_msg)
except Exception as e:
    print(f"Error in main loop: {str(e)}")
    raise