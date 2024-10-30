import pandas as pd
import numpy as np
import re
from datetime import datetime
from Utils import Utils, ModelUtils
from ModelUtils import bsinv, find_vol

class DataHandler:
	def __init__(self):
		self.util = Utils()
		self.modelutils = ModelUtils()

	def parse_file(self, file_path=int):

		price_data = []
		option_data = []

		# Try different encodings
		encodings = ['utf-8', 'iso-8859-1', 'cp1252']
		
		for encoding in encodings:
			try:
				with open(file_path, 'r', encoding=encoding) as file:
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
								# print(f"error")
								pass
				break
			except UnicodeDecodeError:
				continue
		else:
			raise ValueError(f"Unable to decode the file with any of the attempted encodings: {encodings}")
				
		df1 = pd.DataFrame(price_data)
		df2 = pd.DataFrame(option_data)
		
		df1 = df1.set_index('timestamp')
		df2 = df2.set_index('timestamp')
		
		merged_df = df2.merge(df1, left_index=True, right_index=True, how='left', suffixes=('_option', '_price'))
		merged_df = merged_df.reset_index()
		
		return merged_df
	
	def get_basic_data(self, df=pd.DataFrame):
		"""
		Create dataframe with essential features, namely mid prices, time to maturity and implied volatility.

		Args:
			df (pd.DataFrame): pandas dataframe with option prices.

		Returns:
			df
		"""
		df['bid_mid'] = (df['bid_open'] + df['bid_close']) / 2
		df['ask_mid'] = (df['ask_open'] + df['ask_close']) / 2
		df['mid_price'] = (df['bid_mid'] + df['ask_mid']) / 2
		df['maturity'] = pd.to_datetime(df['maturity'])
		df['T'] = (df['maturity'] - df['timestamp']).dt.total_seconds() / (365 * 24 * 60 * 60)
		df['days_to_expiry'] = (df['maturity'] - df['timestamp']).dt.total_seconds() / (24 * 60 * 60)
	
		df['implied_volatility'] = df.apply(lambda row: self.util.calc_implied_volatility(
				row['option_type'],
				row['mid_price'], 
				row['strike'], 
				row['T'], 
				row['close'], 
				r=0.05), 
				axis=1)
		
		# df['vol_of_iv']

		df['log_strike'] = 1 + np.log(df['strike'] / df['close'])
		
		df['vega'] = df.apply(lambda row: self.modelutils.bs_vega(
				row['close'], 
				row['strike'], 
				row['T'], 
				0.05,
				row['implied_volatility']), 
				axis=1)
		
		return df

	def get_advanced_features(
			self, 
            df=pd.DataFrame,
            strike_bins=int, 
            maturity_bins=int, 
            index_lag=int, 
            weighting_method=str):
		"""_summary_

		Args:
			df (pd.DataFrame): pandas dataframe with option prices and basic features.
			bins (_type_, optional): _description_. Defaults to tuple.
			index_lag (_type_, optional): _description_. Defaults to int.
			weighting_method (_type_, optional): _description_. Defaults to str.
		"""

		df = self.util.binning(df, strike_bins) #, weighting_method
		print(df)

		# df[f'iv_index{index_lag}'] = iv_index(df, lag, weighting_method)
		return df	
	
	def calculate_greeks(self, df):
		df_greeks = self.util.greeks(df['mid_price'])
		return df_greeks

