import pandas as pd
import numpy as np
from datetime import datetime
import logging

from core.data.DataHandler import DataHandler

dh = DataHandler()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class OptionsDataMagician:
    def __init__(self):
        logger.info("🎭 Ladies and gentlemen, welcome to the greatest show in financial mathematics!")
        self.start_time = datetime.now()

    def _transform_time_dimensions(self, data):
        """
        Time transformation worthy of a Nobel Prize!
        Now with extra type checking - as Michael Scott would say: 'I'm not superstitious, but I am a little stitious about data types!'
        """
        logger.info("🔍 Inspecting our data types - trust but verify!")
        logger.info(f"Initial data types:\n{data.dtypes}")

        logger.info("\n🧹 Cleaning up the numeric columns first...")
        # Clean up numeric columns
        numeric_columns = ['strike', 'close']
        for col in numeric_columns:
            if col in data.columns:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    logger.info(f"Successfully converted {col} to numeric type!")
                except Exception as e:
                    logger.error(f"Failed to convert {col}: {str(e)}")
                    raise

        logger.info("\n🕰️ Now for the temporal transformation sequence...")
        try:
            # Convert timestamp and maturity to datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                logger.info("Timestamp conversion: CHECK! ✅")
            
            if 'maturity' in data.columns:
                data['maturity'] = pd.to_datetime(data['maturity'])
                logger.info("Maturity conversion: CHECK! ✅")

            # Set index only if timestamp exists
            if 'timestamp' in data.columns:
                data.set_index('timestamp', inplace=True)
                logger.info("Index setting: CHECK! ✅")

            logger.info("\n⚡ Computing time to expiry - the heart of options pricing!")
            # Ensure we're working with datetime objects
            if isinstance(data.index, pd.DatetimeIndex) and pd.api.types.is_datetime64_any_dtype(data['maturity']):
                data['time_to_expiry'] = (data['maturity'] - data.index).dt.total_seconds() / (365 * 24 * 60 * 60)
                logger.info("Time to expiry calculation: CHECK! ✅")
            else:
                raise TypeError("Index or maturity not in datetime format!")

            logger.info("\n📊 Calculating log strike - the backbone of Black-Scholes!")
            if 'strike' in data.columns and 'close' in data.columns:
                data['log_strike'] = np.log(data['strike'].astype(float) / data['close'].astype(float))
                logger.info("Log strike calculation: CHECK! ✅")
            else:
                raise KeyError("Missing required columns: strike or close")

        except Exception as e:
            logger.error(f"Houston, we have a problem! 🚨 {str(e)}")
            logger.info("\nDebug information:")
            logger.info(f"Data types after conversion attempt:\n{data.dtypes}")
            logger.info(f"Sample of data:\n{data.head()}")
            raise

        logger.info("\n✨ Time dimension transformation completed successfully!")
        return data

    def process_options_data(self, file_path):
        """
        The grand performance of options data processing!
        Now with enhanced error checking!
        """
        logger.info("Act 1: The Data Gathering 🎬")
        try:
            raw_data = dh.parse_file(file_path)
            logger.info(f"Successfully loaded {len(raw_data)} rows of raw market poetry!")
            
            # Quick data health check
            logger.info("\n🏥 Data Health Check:")
            logger.info(f"Columns present: {raw_data.columns.tolist()}")
            logger.info(f"Missing values:\n{raw_data.isnull().sum()}")
            
        except Exception as e:
            logger.error(f"Data loading failed! That's what she said... about the error: {str(e)} 😱")
            raise

        logger.info("\nAct 2: The Time Transformation ⏰")
        data = self._transform_time_dimensions(raw_data)

        return data

# The grand performance
if __name__ == "__main__":
    magician = OptionsDataMagician()
    try:
        processed_data = magician.process_options_data("Data/SPY_Options_log.txt")
        
        logger.info("\n🎯 Success! Let's look at our masterpiece:")
        logger.info(f"Processed data shape: {processed_data.shape}")
        logger.info(f"Columns: {processed_data.columns.tolist()}")
        logger.info("\nSample of processed data:")
        logger.info(f"\n{processed_data.head()}")
        
    except Exception as e:
        logger.error(f"The show must go on, but we've hit a snag: {str(e)} 😢")