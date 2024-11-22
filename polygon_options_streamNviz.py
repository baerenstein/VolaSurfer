from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Market
from typing import List
import matplotlib.pyplot as plt
from datetime import datetime

api_key = 'UgOS5zZnpBQfCjbXRTLsontpgPB7FnLr'

client = WebSocketClient(
    api_key=api_key,
    market=Market.Options
) # POLYGON_API_KEY environment variable is used

# aggregates
# client.subscribe("AM.*") # aggregates (per minute)
# client.subscribe("A.*") # aggregates (per minute)
# client.subscribe(
#     "A.O:GOOG241220C00165000, \
#     A.O:GOOG241220C00167500, \
#     A.O:GOOG241220C00170000, \
#     A.O:GOOG241220C00172500, \
#     A.O:GOOG241220P00165000, \
#     A.O:GOOG241220P00167500, \
#     A.O:GOOG241220P00170000, \
#     A.O:GOOG241220P00172500")  # aggregates (per second)

# quotes (1,000 option contracts per connection)
# client.subscribe("Q.O:SPY241220P00720000", "Q.O:SPY251219C00650000") # limited quotes
# client.subscribe("Q.O:NKE241129P00076000", "Q.O:NKE241129C00076000") # limited quotes
client.subscribe("Q.O:NKE241129C00076000") # limited quotes
  
# def handle_msg(msgs: List[WebSocketMessage]):
#     for m in msgs:
#         print(m)
#         print(f"Bid price: {m.bid_price}")

# Initialize empty lists to store bid and ask prices
bid_prices = []
ask_prices = []
  
# Create a Matplotlib figure
plt.ion() # Turn on interactive mode
fig, ax = plt.subplots()

ax.set_title('Bid and Ask Prices')
ax.set_xlabel('Time')
ax.set_ylabel('Price')

def handle_msg(msgs: List[WebSocketMessage]):
    global bid_prices, ask_prices
    for m in msgs:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Get current timestamp
        print(f"{current_time} - Symbol: {m.symbol}, Bid price: {m.bid_price}, Ask price: {m.ask_price}")
        bid_prices.append(m.bid_price)
        ask_prices.append(m.ask_price)

        # Keep only the latest 50 data points
        if len(bid_prices) > 150:
            bid_prices.pop(0)
        if len(ask_prices) > 150:
            ask_prices.pop(0)

        # Clear the axes and plot the updated data
        ax.clear()
        ax.set_title('Bid and Ask Prices')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.plot(list(range(len(bid_prices))), bid_prices, label='Bid Price', color='blue')
        ax.plot(list(range(len(ask_prices))), ask_prices, label='Ask Price', color='orange')
        ax.legend()

        # Show the updated figure
        plt.pause(0.1) # Pause to allow the plot to update

# print messages
client.run(handle_msg)