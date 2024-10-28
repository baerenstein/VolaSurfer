import pandas as pd
import numpy as np
from datetime import datetime
import logging
from DataHandler import DataHandler
from DataMagician import OptionsDataMagician

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ImpliedVolatilityIndexMagician:
    def __init__(self, data_handler, days_to_expiry=30):
        logger.info("🎩 Welcome to the Volatility Show!")
        self.dh = data_handler
        self.target_days = days_to_expiry
        self.start_time = datetime.now()

    def prepare_and_calculate(self, raw_data):
        """
        The grand preparation and calculation show!
        """
        logger.info("🎭 Act 1: Data Processing Extravaganza!")
        try:
            # Process the data using DataHandler's process_data method
            processed_data = self.dh.process_data(raw_data)
            logger.info("✨ Basic data processing complete!")

            # Additional processing specific to IV index calculation
            processed_data = self._enhance_data(processed_data)
            logger.info("🌟 Enhanced data processing complete!")

            # Calculate IV index
            iv_index = self._calculate_iv_index(processed_data)
            
            return iv_index, processed_data

        except Exception as e:
            logger.error(f"💥 Show interrupted! Error: {str(e)}")
            raise

    def _enhance_data(self, data):
        """
        Enhance the processed data with additional metrics - now with proper datetime handling!
        """
        logger.info("🎨 Adding the finishing touches to our data...")
        
        try:
            # Calculate days to expiry using proper datetime arithmetic
            data['dte'] = (data['maturity'] - data.index).dt.total_seconds() / (24 * 60 * 60)
            logger.info("✅ DTE calculation complete!")

            # Calculate moneyness with input validation
            data['moneyness'] = np.where(
                (data['strike'] > 0) & (data['close'] > 0),
                data['strike'] / data['close'],
                np.nan
            )
            logger.info("✅ Moneyness calculation complete!")

            # Create moneyness bins with proper handling of edge cases
            data['moneyness_bin'] = pd.qcut(
                data['moneyness'].clip(lower=0.5, upper=1.5),  # Clip extreme values
                q=5,
                labels=['Deep Put', 'Put', 'ATM', 'Call', 'Deep Call'],
                duplicates='drop'  # Handle duplicate edge values
            )
            logger.info("✅ Moneyness binning complete!")

            # Calculate forward prices with validation
            data['forward'] = np.where(
                (data['close'] > 0) & (data['T'] >= 0),
                data['close'] * np.exp(0.05 * data['T']),
                np.nan
            )
            logger.info("✅ Forward price calculation complete!")

            # Remove any rows with invalid calculations
            valid_data = data.dropna(subset=['dte', 'moneyness', 'forward'])
            invalid_count = len(data) - len(valid_data)
            if invalid_count > 0:
                logger.warning(f"⚠️ Removed {invalid_count} rows with invalid calculations")
            
            # Add data quality metrics
            valid_data['data_quality'] = np.where(
                (valid_data['implied_volatility'] > 0) & 
                (valid_data['implied_volatility'] < 5) &
                (valid_data['bid_mid'] > 0) &
                (valid_data['ask_mid'] > valid_data['bid_mid']),
                'Good',
                'Suspect'
            )

            # Log some statistics
            logger.info("\n📊 Data Enhancement Statistics:")
            logger.info(f"Total rows: {len(valid_data)}")
            logger.info(f"Moneyness range: {valid_data['moneyness'].min():.2f} to {valid_data['moneyness'].max():.2f}")
            logger.info(f"DTE range: {valid_data['dte'].min():.1f} to {valid_data['dte'].max():.1f}")
            logger.info(f"Data quality: {valid_data['data_quality'].value_counts().to_dict()}")

            return valid_data

        except Exception as e:
            logger.error(f"❌ Data enhancement failed: {str(e)}")
            logger.error("Debug information:")
            logger.error(f"Data columns: {data.columns.tolist()}")
            logger.error(f"Data types: {data.dtypes}")
            raise

    def _calculate_iv_index(self, data):
        """
        The main event - calculating our volatility index with improved error handling!
        """
        logger.info("\n🎭 Act 2: The IV Index Calculation!")
        
        try:
            # Filter for valid data points
            valid_data = data[data['data_quality'] == 'Good'].copy()
            logger.info(f"Using {len(valid_data)} valid options for calculation")

            # Get options near target expiry
            target_dte = self.target_days
            valid_data['dte_diff'] = abs(valid_data['dte'] - target_dte)
            
            # Find two nearest expiries
            nearest_expiries = (
                valid_data.groupby('maturity')['dte_diff']
                .mean()
                .sort_values()
                .head(2)
                .index
                .tolist()
            )
            
            if len(nearest_expiries) < 2:
                raise ValueError(f"Need at least 2 expiries, found {len(nearest_expiries)}")
                
            logger.info(f"Selected expiries: {nearest_expiries}")

            # Calculate variance for each expiry
            variances = []
            weights = []
            
            for expiry in nearest_expiries:
                expiry_data = valid_data[valid_data['maturity'] == expiry].copy()
                if len(expiry_data) >= 5:  # Ensure enough points for calculation
                    variance = self._calculate_variance_contribution(expiry_data)
                    time_weight = expiry_data['T'].iloc[0]
                    variances.append(variance)
                    weights.append(time_weight)
                    logger.info(f"Variance for {expiry}: {variance:.4f}")
                else:
                    logger.warning(f"⚠️ Insufficient data points for {expiry}")

            if not variances:
                raise ValueError("No valid variances calculated")

            # Interpolate to target
            target_variance = self._interpolate_variance(variances, weights)
            iv_index = np.sqrt(target_variance * 365) * 100  # Annualized percentage
            
            # Log completion and results
            execution_time = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"\n🎉 Index calculation completed in {execution_time:.2f} seconds!")
            logger.info(f"🎯 IV Index: {iv_index:.2f}%")
            
            # Additional diagnostics
            logger.info("\n📊 Calculation Diagnostics:")
            logger.info(f"Number of expiries used: {len(variances)}")
            logger.info(f"Time weights: {weights}")
            logger.info(f"Raw variances: {variances}")
            
            return iv_index

        except Exception as e:
            logger.error(f"❌ IV Index calculation failed: {str(e)}")
            raise

    def _calculate_variance_contribution(self, data):
        """
        Calculate variance contribution for a specific expiry
        """
        try:
            # Sort by strike
            data = data.sort_values('strike')
            
            # Calculate forward price
            atm_index = (data['moneyness'] - 1).abs().idxmin()
            forward = data.loc[atm_index, 'forward']
            
            # Initialize contribution
            variance = 0
            
            # Sum up option contributions
            for i in range(1, len(data) - 1):
                K = data['strike'].iloc[i]
                delta_K = (data['strike'].iloc[i+1] - data['strike'].iloc[i-1]) / 2
                option_price = data['mid_price'].iloc[i]
                
                contribution = (delta_K / K**2) * option_price
                variance += contribution
            
            # Normalize by time to expiry
            variance *= 2 / data['T'].iloc[0]
            
            return variance

        except Exception as e:
            logger.error(f"❌ Variance calculation failed: {str(e)}")
            raise

    def _interpolate_variance(self, variances, weights):
        """
        Interpolate variance to target maturity
        """
        if len(variances) < 2:
            return variances[0]
        
        target_weight = self.target_days / 365
        w1 = (weights[1] - target_weight) / (weights[1] - weights[0])
        w2 = 1 - w1
        
        return w1 * variances[0] + w2 * variances[1]

if __name__ == "__main__":
    dh = DataHandler()
    raw_data = dh.parse_file("Data/SPY_Options_log.txt")
    
    iv_magician = ImpliedVolatilityIndexMagician(dh)
    
    try:
        logger.info("🎬 Starting the IV Index Show!")
        logger.info("\n🔍 Initial Data Check:")
        logger.info(f"Raw data shape: {raw_data.shape}")
        logger.info(f"Raw data columns: {raw_data.columns.tolist()}")
        
        iv_index, processed_data = iv_magician.prepare_and_calculate(raw_data)
        
        logger.info("\n📊 Final Analytics:")
        logger.info(f"IV Index: {iv_index:.2f}%")
        logger.info("Volatility Term Structure:")
        dte_groups = processed_data.groupby(pd.qcut(processed_data['dte'], 4))
        for name, group in dte_groups:
            logger.info(f"DTE {name.left:.0f}-{name.right:.0f}: {group['implied_volatility'].mean():.2%}")
        
        # Save results
        output_file = "processed_options_data.csv"
        processed_data.to_csv(output_file)
        logger.info(f"\n💾 Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"🎭 The show must go on, but we've hit a snag: {str(e)}")