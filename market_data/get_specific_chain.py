import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from polygon import RESTClient
from dataclasses import dataclass
from typing import List, Dict, Set, Union
import os
from pathlib import Path

@dataclass
class StrikeSelection:
    atm_strike: float
    selected_strikes: Set[float]
    maturity: str
    contract_types: List[str]

class EnhancedOptionFetcher:
    def __init__(
        self,
        ticker: str,
        num_strikes: int = 2,
        days_to_expiration: int = 15,
        max_contract_tickers: int = 5,
    ):
        self.ticker = ticker
        self.num_strikes = num_strikes
        self.days_to_expiration = days_to_expiration
        self.max_contract_tickers = max_contract_tickers
        self.api_key = os.getenv("API_KEY")
        self.client = RESTClient(self.api_key)

    def fetch_stock_data(self):
        """Fetch previous day's stock data"""
        prev_close_url = f"https://api.polygon.io/v2/aggs/ticker/{self.ticker}/prev?adjusted=true&apiKey={self.api_key}"
        
        try:
            response = requests.get(prev_close_url, timeout=10)  # Add timeout
            response.raise_for_status()  # Raise exception for bad status codes
            
            data = response.json()
            if "results" not in data:
                raise Exception("No results found in stock data")
                
            # Create a single-row DataFrame with previous day's data
            prev_day_data = data["results"][0]  # Access first element of results array
            results_df = pd.DataFrame([{
                'date': pd.to_datetime(prev_day_data['t'], unit='ms'),
                # 'o': prev_day_data['o'],
                # 'h': prev_day_data['h'],
                # 'l': prev_day_data['l'],
                'c': prev_day_data['c'],
                # 'v': prev_day_data['v']
            }])
            results_df.set_index('date', inplace=True)
            
            self.last_closing_price = prev_day_data['c']
            print("Previous day's data retrieved successfully:")
            # print(f"Open: ${prev_day_data['o']:.2f}")
            # print(f"High: ${prev_day_data['h']:.2f}")
            # print(f"Low: ${prev_day_data['l']:.2f}")
            print(f"Date: {results_df.index[0].strftime('%Y-%m-%d')}")  # Fixed date printing
            print(f"Close: ${prev_day_data['c']:.2f}")
            print(results_df)
            # print(f"Volume: {prev_day_data['v']:,}")
            
            return results_df
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching stock data: {str(e)}")

    def select_strike_range(self, contracts_df: pd.DataFrame) -> pd.DataFrame:
        """Select strikes around ATM for each maturity"""
        if contracts_df.empty:
            print("options contract dataframe is empty!")
            return pd.DataFrame()
            
        selected_contracts = []
        
        # Group by expiration date and contract type to handle calls and puts separately
        for (expiry_date, contract_type), group in contracts_df.groupby(['expiration_date', 'contract_type']):
            strikes = sorted(group['strike_price'].unique())
            if not strikes:
                continue
                
            # Find ATM strike
            atm_strike = min(strikes, key=lambda x: abs(x - self.last_closing_price))
            atm_index = strikes.index(atm_strike)
            
            # Select strikes around ATM
            selected_indices = range(
                max(0, atm_index - self.num_strikes),
                min(len(strikes), atm_index + self.num_strikes + 1)
            )
            
            selected_strikes = [strikes[i] for i in selected_indices]
            
            # Filter contracts for selected strikes
            selected_group = group[group['strike_price'].isin(selected_strikes)]
            selected_contracts.extend(selected_group.to_dict('records'))
            
            print(f"\nSelected strikes for {expiry_date} {contract_type}:")
            print(f"ATM strike: ${atm_strike:.2f}")
            print(f"Strike range: ${min(selected_strikes):.2f} - ${max(selected_strikes):.2f}")
        
        return pd.DataFrame(selected_contracts)

    def fetch_option_data(self):
        """Fetch option contract data and quotes"""
        expiration_date_lte = (datetime.now() + timedelta(days=self.days_to_expiration)).strftime('%Y-%m-%d')
        
        print(f"\nFetching option contracts expiring before {expiration_date_lte}...")
        
        try:
            contracts = []
            for c in self.client.list_options_contracts(
                self.ticker,
                limit=500,
                sort="strike_price",
                order="desc",
                expired=False,
                expiration_date_lte=expiration_date_lte
            ):
                contracts.append(c)
                
            if not contracts:
                raise Exception("No option contracts found")
                
            print(f"Found {len(contracts)} total contracts")
            
            contracts_df = pd.DataFrame([{
                'ticker': c.ticker,
                'underlying_ticker': c.underlying_ticker,
                'contract_type': c.contract_type,
                'strike_price': c.strike_price,
                'expiration_date': c.expiration_date,
            } for c in contracts])
            
            # Select strikes around ATM
            filtered_contracts_df = self.select_strike_range(contracts_df)
            
            if filtered_contracts_df.empty:
                raise Exception("No contracts selected after filtering")
                
            print(f"\nSelected {len(filtered_contracts_df)} contracts for quote fetching")
            
            # Fetch quotes for selected contracts
            contract_tickers = filtered_contracts_df['ticker'].unique()[:self.max_contract_tickers]
            all_quotes = []
            
            for contract_ticker in contract_tickers:
                print(f"Fetching quotes for {contract_ticker}...")
                quotes = []
                for t in self.client.list_quotes(
                    contract_ticker,
                    limit=5000,
                    order="desc",
                    sort="timestamp",
                ):
                    quotes.append({
                        'ask_price': t.ask_price,
                        'ask_size': t.ask_size,
                        'bid_price': t.bid_price,
                        'bid_size': t.bid_size,
                        'timestamp': pd.to_datetime(t.sip_timestamp, unit='ns'),
                        'ticker': contract_ticker,
                    })
                print(f"Retrieved {len(quotes)} quotes")
                all_quotes.extend(quotes)
                
            if not all_quotes:
                raise Exception("No quotes retrieved for any contracts")
                
            quotes_df = pd.DataFrame(all_quotes)
            
            # Merge contracts and quotes
            merged_df = filtered_contracts_df.merge(quotes_df, on='ticker', how='inner')
            final_df = merged_df.set_index('timestamp')[
                ['ticker', 'underlying_ticker', 'contract_type', 'strike_price',
                 'expiration_date', 'ask_price', 'ask_size', 'bid_price', 'bid_size']
            ]
            
            return filtered_contracts_df, final_df
            
        except Exception as e:
            print(f"Error in fetch_option_data: {str(e)}")
            return None, None

    def save_market_data(self, results_df, merged_df):
        """Save stock and option data to parquet"""
        try:
            data_dir = Path(f"market_data/{self.ticker}")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            current_date = datetime.now().strftime("%Y%m%d")
            
            if not results_df.empty:
                stock_path = data_dir / f"stock_{current_date}.parquet"
                results_df.to_parquet(stock_path, engine='pyarrow')
                print(f"Stock data saved to {stock_path}")
            
            if not merged_df.empty:
                options_path = data_dir / f"options_{current_date}.parquet"
                merged_df.to_parquet(options_path, engine='pyarrow')
                print(f"Options data saved to {options_path}")
                
        except Exception as e:
            print(f"Error saving market data: {str(e)}")

def main():
    fetcher = EnhancedOptionFetcher(
        ticker="NKE",
        num_strikes=2,
        days_to_expiration=15,
        max_contract_tickers=5
    )
    
    try:
        # Fetch stock data
        print("Fetching stock data...")
        stock_df = fetcher.fetch_stock_data()
        
        # Fetch option data
        print("\nFetching option data...")
        contracts_df, final_df = fetcher.fetch_option_data()
        
        if contracts_df is not None and final_df is not None:
            # Save data
            print("\nSaving data to parquet...")
            fetcher.save_market_data(stock_df, final_df)
            
            # Save quotes to CSV
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"NKE_selected_options_{current_time}.csv"
            final_df.reset_index().to_csv(csv_filename, index=False)
            print(f"Selected options data saved to {csv_filename}")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()