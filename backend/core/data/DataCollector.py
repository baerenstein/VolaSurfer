from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from datetime import datetime
from .DataHandler import DataHandler

@dataclass
class OptionsData:
    symbol: str
    timestamps: List[datetime]
    expiration_dates: List[datetime]
    strikes: List[float]
    calls: pd.DataFrame
    puts: pd.DataFrame

class DataCollector:
    def __init__(self):
        self.data_handler = DataHandler()
        self.cache = {}  # Simple in-memory cache

    def collect_options_data(self, file_path: str) -> Optional[OptionsData]:
        """Collect and organize options data from the log file"""
        try:
            # Use existing DataHandler to parse the file
            df = self.data_handler.parse_file(file_path)
            df = self.data_handler.get_basic_data(df)

            # Split into calls and puts
            calls_df = df[df['option_type'] == 'Call']
            puts_df = df[df['option_type'] == 'Put']

            # Create OptionsData object
            options_data = OptionsData(
                symbol='SPY',
                timestamps=sorted(df['timestamp'].unique().tolist()),
                expiration_dates=sorted(df['maturity'].unique().tolist()),
                strikes=sorted(df['strike'].unique().tolist()),
                calls=calls_df,
                puts=puts_df
            )

            # Cache the results
            self.cache[file_path] = options_data
            
            return options_data

        except Exception as e:
            raise Exception(f"Error collecting options data: {str(e)}")

    def get_cached_data(self, file_path: str) -> Optional[OptionsData]:
        """Retrieve cached data if available, otherwise collect new data"""
        if file_path in self.cache:
            return self.cache[file_path]
        return self.collect_options_data(file_path)
    
    def get_cached_data_as_df(self, file_path: str) -> pd.DataFrame:
        """Retrieve cached data as a DataFrame indexed by timestamp"""
        options_data = self.get_cached_data(file_path)
        combined_df = pd.concat([options_data.calls, options_data.puts])
        combined_df = combined_df.set_index(['timestamp', 'days_to_expiry', 'strike', 'option_type'])
        combined_df.sort_index(inplace=True)
        return combined_df