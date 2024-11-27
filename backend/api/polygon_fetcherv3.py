import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
api_key = os.getenv("API_KEY")

ticker_list = [
    'O:NKE241129P00075000',
    'O:NKE241129P00080000', 
    'O:NKE241129C00080000', 
    'O:NKE241206C00075000', 

]

stock_url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/hour/2024-11-19/2024-11-24?adjusted=true&sort=asc&apiKey={api_key}"
response = requests.get(stock_url)

if response.status_code == 200:
    data = response.json()
    print("Ticker:", data.get("ticker", "N/A"))
    print("Query Count:", data.get("queryCount", "N/A"))
    print("Results Count:", data.get("resultsCount", "N/A"))
    
    # Check if 'results' key exists
    if "results" in data:
        results_df = pd.DataFrame(data["results"])
        # Convert timestamp to datetime
        results_df['date'] = pd.to_datetime(results_df['t'], unit='ms')
        results_df.set_index('date', inplace=True)

        # Print the DataFrame in the terminal
        print(results_df[['o', 'c', 'h', 'l', 'v']])
    else:
        print("No results found for ticker:", data.get("ticker", "N/A"))
else:
    print("Error fetching data for ticker AAPL:", response.status_code, response.text)

def collect_data(ticker):
    # Construct the URL for the specific ticker
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/2024-10-29/2024-11-11?adjusted=true&sort=asc&apiKey={api_key}"

    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        print("Ticker:", data.get("ticker", "N/A"))
        print("Query Count:", data.get("queryCount", "N/A"))
        print("Results Count:", data.get("resultsCount", "N/A"))
        
        # Check if 'results' key exists
        if "results" in data:
            results_df = pd.DataFrame(data["results"])
            # Convert timestamp to datetime
            results_df['date'] = pd.to_datetime(results_df['t'], unit='ms')
            results_df.set_index('date', inplace=True)

            # Display the DataFrame
            print(results_df[['o', 'c', 'h', 'l', 'v']])

            # Store closing prices for aggregation
            return results_df['c']  # Return closing prices for aggregation
        else:
            print("No results found for ticker:", ticker)
            return None
    else:
        print("Error fetching data for ticker", ticker, ":", response.status_code, response.text)
        return None

# Aggregate closing prices for all tickers
all_closing_prices = {}
for ticker in ticker_list:
    closing_prices = collect_data(ticker)
    if closing_prices is not None:
        all_closing_prices[ticker] = closing_prices

# Plotting all closing prices in one graph
plt.figure(figsize=(12, 6))
for ticker, prices in all_closing_prices.items():
    plt.plot(prices.index, prices, marker='o', linestyle='-', label=ticker)

plt.title('Contract Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

