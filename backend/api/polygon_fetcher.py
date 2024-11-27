from polygon import RESTClient
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
api_key = os.getenv("API_KEY")

client = RESTClient(api_key)  # POLYGON_API_KEY

ticker = "NKE"

days_to_expiration = 30  # Define the number of days to add
expiration_date_lte = (datetime.datetime.now() + datetime.timedelta(days=days_to_expiration)).strftime('%Y-%m-%d')

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

# Statistics: Number of contracts by strike price
strike_price_stats = contracts_df.groupby('strike_price').size()
print("\nNumber of contracts by strike price:")
print(strike_price_stats)

# Define the number of days for the date range
days_begin = 7  # Number of days to look back from today
days_end = 1  # Number of days to look forward (yesterday)

# Calculate the timestamps for the date range
end_timestamp = int((datetime.datetime.now() - datetime.timedelta(days=days_end)).timestamp() * 1_000_000_000)  # Today's timestamp minus days_end
start_timestamp = int((datetime.datetime.now() - datetime.timedelta(days=days_begin)).timestamp() * 1_000_000_000)  # Start timestamp


contract_ticker = contracts_df.iloc[0]['ticker']
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
    quotes.append(t)

print(f"Number of quotes fetched: {len(quotes)}")
if quotes:
    print("First quote fetched:", quotes[0])

quotes_df = pd.DataFrame([{
    'ask_price': q.ask_price,
    'ask_size': q.ask_size,
    'bid_price': q.bid_price,
    'bid_size': q.bid_size,
    'sip_timestamp': q.sip_timestamp,
} for q in quotes])

# Convert sip_timestamp to datetime and rename to 'timestamp'
quotes_df['timestamp'] = pd.to_datetime(quotes_df['sip_timestamp'], unit='ns')

# Sort quotes_df by the new 'timestamp' column
quotes_df.sort_values(by='timestamp', ascending=True, inplace=True)

print(f"Quotes DataFrame has {len(quotes_df)} rows and {len(quotes_df.columns)} columns")
print("Quotes DataFrame (sorted by timestamp):")
print(quotes_df)

# Add ticker column to quotes_df based on the contract we're fetching
quotes_df['ticker'] = contract_ticker  # Add the ticker we're querying

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
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    contract_ticker = final_df['ticker'].iloc[0]  # Get the ticker from the final DataFrame
    filename = f"{contract_ticker}_quotes_{current_time}.csv"

    # Save the final DataFrame as a .csv file
    final_df.reset_index().to_csv(filename, index=False)
    print(f"Final DataFrame saved as {filename}")

except Exception as e:
    print(f"Error merging DataFrames: {str(e)}")