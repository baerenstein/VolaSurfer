import pandas as pd
import numpy as np
import re
from datetime import datetime

file_path = "C:/Users/mbeckhusen/Documents/VolaSurfer/SPY_Options_log.txt"

price_data = []
option_data = []

with open(file_path, 'r') as file:
    for line in file:
        if 'SPY:' in line:
            parts = line.split()
            timestamp = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S")
            ticker = 'SPY'

            try:
                open_price = float(parts[4])
                high = float(parts[6])
                low = float(parts[8])
                close = float(parts[10])
                volume = float(parts[12])

                price_data.append({
                    'timestamp': timestamp,
                    'ticker': ticker,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                })

            except:
                print("we have problem")
            
        if 'SPY' in line and ('Bid:' in line and 'Ask:' in line):
            parts = line.split()
            timestamp = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S")
            ticker = 'SPY'

            contract = parts[3].rstrip(':')
            
            match = re.match(r'(\d{6})([CP])(\d+)', contract)
            if match:
                maturity = datetime.strptime(match.group(1), "%y%m%d").date()
                option_type = 'Call' if match.group(2) == 'C' else 'Put'
                strike = float(match.group(3)) / 1000
            
            # Find indices for bid and ask data
            bid_index = parts.index('Bid:')
            ask_index = parts.index('Ask:')

            # Extract bid data
            try:
                bid_open = float(parts[bid_index + 2])
                bid_high = float(parts[bid_index + 5])
                bid_low = float(parts[bid_index + 8])
                bid_close = float(parts[bid_index + 11])

                # Extract ask data
                ask_open = float(parts[ask_index + 2])
                ask_high = float(parts[ask_index + 5])
                ask_low = float(parts[ask_index + 8])
                ask_close = float(parts[ask_index + 11])
                
                option_data.append({
                    'timestamp': timestamp,
                    'ticker': ticker,
                    'maturity': maturity,
                    'strike': strike,
                    'option_type': option_type,
                    'bid_open': bid_open,
                    'bid_high': bid_high,
                    'bid_low': bid_low,
                    'bid_close': bid_close,
                    'ask_open': ask_open,
                    'ask_high': ask_high,
                    'ask_low': ask_low,
                    'ask_close': ask_close
                })       

            except:
                print(f"Bid index: {bid_index}")
                print(f"Attempting to convert: {parts[bid_index + 2]}")
                print(line)

# Create pandas DataFrames
df1 = pd.DataFrame(price_data)
df2 = pd.DataFrame(option_data)

# Set 'timestamp' as index for both dataframes
df1 = df1.set_index('timestamp')
df2 = df2.set_index('timestamp')

# print(df1)
# print(df2)

# Merge the dataframes
merged_df = df2.merge(df1, left_index=True, right_index=True, how='left', suffixes=('_option', '_price'))

# Reset index if you want 'timestamp' as a regular column
merged_df = merged_df.reset_index()

# Calculate log strike
merged_df['log_strike'] = 1 + np.log(merged_df['strike'] / merged_df['close'])

# print(merged_df.head())

# If you want to see some statistics of the new column:
print(merged_df['log_strike'].describe())

# If you want to see the dataframe with only certain columns:
columns_to_display = ['timestamp', 'bid_close', 'close', 'log_strike', 'option_type']
print(merged_df[columns_to_display])