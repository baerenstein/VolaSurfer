import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from polygon import RESTClient
import os

api_key = os.getenv("API_KEY")

ticker = "NKE"

client = RESTClient(api_key)  # POLYGON_API_KEY

stock_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/2024-11-19/2024-11-24?adjusted=true&sort=asc&apiKey={api_key}"

response = requests.get(stock_url)

if response.status_code == 200:
    data = response.json()
    print("Ticker:", data.get("ticker", "N/A"))
    print("Query Count:", data.get("queryCount", "N/A"))
    print("Results Count:", data.get("resultsCount", "N/A"))
    
    # Check if 'results' key existsx
    if "results" in data:
        results_df = pd.DataFrame(data["results"])
        # Convert timestamp to datetime
        results_df['date'] = pd.to_datetime(results_df['t'], unit='ms')
        results_df.set_index('date', inplace=True)

        # Print the DataFrame in the terminal
        print(results_df[['o', 'c', 'h', 'l', 'v']])
        
        # Create a variable for the last closing price
        last_closing_price = results_df['c'].iloc[-1]  # Get the last closing price
        print("Last Closing Price:", last_closing_price)  # Print the last closing price
    else:
        print("No results found for ticker:", data.get("ticker", "N/A"))
else:
    print(f"Error fetching data for ticker {ticker}:", response.status_code, response.text)

######################## option data

days_to_expiration = 15  # Define the number of days to add
expiration_date_lte = (datetime.now() + timedelta(days=days_to_expiration)).strftime('%Y-%m-%d')

print(f"fetching contracts for {ticker} with {days_to_expiration} days to expiration...")
contracts = []
for c in client.list_options_contracts(
    ticker, 
    limit=500, 
    sort="strike_price", 
    order="desc",
    expired=False,
    expiration_date_lte=expiration_date_lte  # Use the calculated expiration date
):
    contracts.append(c)
print(f"number of maturities: {pd.DataFrame(contracts)['expiration_date'].nunique()}")
# print each unique maturities
for maturity in pd.DataFrame(contracts)['expiration_date'].unique():
    print(maturity)

# Sort contracts by strike price
sorted_contracts = sorted(contracts, key=lambda x: x.strike_price)
# Create a DataFrame from the sorted contracts
contracts_df = pd.DataFrame([{
    'ticker': c.ticker,
    'underlying_ticker': c.underlying_ticker,
    'contract_type': c.contract_type,
    'strike_price': c.strike_price,
    'expiration_date': c.expiration_date,
} for c in sorted_contracts])
print(f"printing contract dataframe with {len(contracts_df)} rows and {len(contracts_df.columns)} columns") 
print(contracts_df)

# Statistics: Number of contracts at each unique expiration date
expiration_stats = contracts_df.groupby('expiration_date').size()
print("\nNumber of contracts at each unique expiration date:")
print(expiration_stats)

# # Statistics: Number of contracts by strike price
# strike_price_stats = contracts_df.groupby('strike_price').size()
# print("\nNumber of contracts by strike price:")
# print(strike_price_stats)

# Define the number of days for the date range
days_begin = 7  # Number of days to look back from today
days_end = 1  # Number of days to look forward (yesterday)

# Calculate the timestamps for the date range
end_timestamp = int((datetime.now() - timedelta(days=days_end)).timestamp() * 1_000_000_000)  # Today's timestamp minus days_end
start_timestamp = int((datetime.now() - timedelta(days=days_begin)).timestamp() * 1_000_000_000)  # Start timestamp

# Define the maximum number of contract tickers to fetch
max_contract_tickers = 5  # Change this value as needed

# Extract unique contract tickers from contracts_df
contract_tickers = contracts_df['ticker'].unique()[:max_contract_tickers]  # Limit to max_contract_tickers
all_quotes = []  # Initialize a list to store all quotes

# Loop through each contract ticker to fetch quotes
for contract_ticker in contract_tickers:
    print(f"Fetching quotes for contract ticker: {contract_ticker}")
    quotes = []
    for t in client.list_quotes(
        contract_ticker,
        limit=5000,
        order="desc",
        # timestamp_lte=end_timestamp,
        # timestamp_gte=start_timestamp,
        sort="timestamp",
    ):
        quotes.append({
            'ask_price': t.ask_price,
            'ask_size': t.ask_size,
            'bid_price': t.bid_price,
            'bid_size': t.bid_size,
            'sip_timestamp': t.sip_timestamp,
            'ticker': contract_ticker,  # Assign the ticker here
        })

    print(f"Number of quotes fetched for {contract_ticker}: {len(quotes)}")
    if quotes:
        # print("First quote fetched:", quotes[0])
        all_quotes.extend(quotes)  # Add the fetched quotes to the all_quotes list

# Optionally, convert all_quotes to a DataFrame if needed
if all_quotes:
    quotes_df = pd.DataFrame(all_quotes)  # Create DataFrame directly from the list of dictionaries

    print(f"Total quotes fetched: {len(quotes_df)}")
    print(quotes_df.head())  # Display the first few rows of the quotes DataFrame

# Debugging: Print columns of both DataFrames
print("Contracts DataFrame columns:", contracts_df.columns)
print("Quotes DataFrame columns:", quotes_df.columns)

try:
    # Merge contracts and quotes DataFrames
    merged_df = contracts_df.merge(quotes_df, on='ticker', how='inner')
    
    # Select relevant columns and set datetime as index
    final_df = merged_df.set_index('timestamp')[[
        'ticker', 'underlying_ticker', 'contract_type', 'strike_price', 
        'expiration_date', 'ask_price', 'ask_size', 'bid_price', 'bid_size'
    ]]

    # Print the final DataFrame
    print(final_df)
    print(f"Final DataFrame has {len(final_df)} rows and {len(final_df.columns)} columns")
    print(f"Columns: {final_df.columns}")
    print(final_df.head())  # Display the first few rows for inspection

    # Construct filename based on the contract ticker and current time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    contract_ticker = final_df['ticker'].iloc[0]  # Get the ticker from the final DataFrame
    filename = f"{contract_ticker}_quotes_{current_time}.csv"

    # Save the final DataFrame as a .csv file
    final_df.reset_index().to_csv(filename, index=False)
    print(f"Final DataFrame saved as {filename}")

except Exception as e:
    print(f"Error merging DataFrames: {str(e)}")