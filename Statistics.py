import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta

class Statistics:
    def __init__(self, df):
        """
        Initialize the Statistics class with a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing option data
        """
        self.df = df
        print("Welcome to Statistics Island!")
        # print("Printing loaded dataframe columns")
        # print(self.df.columns)

    def compute_statistics(self, group_by=['maturity', 'option_type']):
        """
        Compute statistics for implied volatility, grouped by specified columns.
        
        Args:
            group_by (list): List of columns to group by. Default is ['maturity', 'option_type']
        
        Returns:
            pd.DataFrame: Table of statistics
        """
        stats_df = self.df.groupby(group_by)['implied_volatility'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('skew', stats.skew),
            ('kurtosis', stats.kurtosis),
            ('min', 'min'),
            ('max', 'max'),
            ('range', lambda x: x.max() - x.min()),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25))
        ]).reset_index()

        return stats_df

    def compute_smile_statistics(self):
        """
        Compute statistics for the volatility smile (IV vs. moneyness).
        
        Returns:
            pd.DataFrame: Table of smile statistics
        """
        self.df['moneyness'] = self.df['strike'] / self.df['close']
        
        def smile_slope(group):
            return np.polyfit(group['moneyness'], group['implied_volatility'], 1)[0]
        
        def smile_curvature(group):
            return np.polyfit(group['moneyness'], group['implied_volatility'], 2)[0]

        smile_stats = self.df.groupby(['maturity', 'option_type']).apply(
            lambda x: pd.Series({
                'smile_slope': smile_slope(x),
                'smile_curvature': smile_curvature(x),
                'atm_iv': x.loc[x['moneyness'].between(0.95, 1.05), 'implied_volatility'].mean(),
                'otm_put_skew': x.loc[(x['moneyness'] < 0.95) & (x['option_type'] == 'Put'), 'implied_volatility'].mean() - 
                                x.loc[x['moneyness'].between(0.95, 1.05), 'implied_volatility'].mean(),
                'otm_call_skew': x.loc[(x['moneyness'] > 1.05) & (x['option_type'] == 'Call'), 'implied_volatility'].mean() - 
                                 x.loc[x['moneyness'].between(0.95, 1.05), 'implied_volatility'].mean()
            })
        ).reset_index()

        return smile_stats

    def term_structure_statistics(self):
        """
        Compute statistics for the volatility term structure.
        
        Returns:
            pd.DataFrame: Table of term structure statistics
        """
        # Calculate time to expiration
        now = pd.Timestamp.now()
        self.df['time_to_expiration'] = (self.df['maturity'] - now).dt.total_seconds() / (24 * 60 * 60)
        
        # Filter ATM options
        atm_df = self.df[(self.df['moneyness'].between(0.95, 1.05))]
        
        # Calculate statistics
        term_structure_stats = pd.DataFrame({
            'short_term_iv': atm_df[atm_df['time_to_expiration'] <= 30]['implied_volatility'].mean(),
            
            'medium_term_iv': atm_df[(atm_df['time_to_expiration'] > 30) & 
                                    (atm_df['time_to_expiration'] <= 90)]['implied_volatility'].mean(),
            
            'long_term_iv': atm_df[atm_df['time_to_expiration'] > 90]['implied_volatility'].mean(),
            
            'term_structure_slope': np.polyfit(atm_df['time_to_expiration'], atm_df['implied_volatility'], 1)[0],
            
            'term_structure_curvature': np.polyfit(atm_df['time_to_expiration'], atm_df['implied_volatility'], 2)[0]
        }, index=['Value'])

        return term_structure_stats
    
    def generate_summary(self):
    #     """
    #     Generate a comprehensive summary of all statistics.
        
    #     Returns:
    #         dict: Dictionary containing all computed statistics
    #     """
    #     return {
    #         'overall_stats': self.compute_statistics(),
    #         'smile_stats': self.compute_smile_statistics(),
    #         'term_structure_stats': self.term_structure_statistics()
    #     }
        """
        Generate a comprehensive summary of all statistics.
        
        Returns:
            pd.DataFrame: Pandas DataFrame containing all computed statistics
        """
        # Compute all statistics
        overall_stats = self.compute_statistics()
        print("overall stats loaded")
        smile_stats = self.compute_smile_statistics()
        term_structure_stats = self.term_structure_statistics().reset_index()
        
        # Add a 'statistic_type' column to each DataFrame
        overall_stats['statistic_type'] = 'overall'
        print("overall stats has type")

        smile_stats['statistic_type'] = 'smile'
        term_structure_stats['statistic_type'] = 'term_structure'
        
        # Combine all statistics into a single pandas DataFrame
        combined_stats = pd.concat([overall_stats, smile_stats, term_structure_stats], ignore_index=True)
        print("stats have been combined")

        
        # Reorder columns to put 'statistic_type' first
        cols = ['statistic_type'] + [col for col in combined_stats.columns if col != 'statistic_type']
        combined_stats = combined_stats[cols]
        print("just some data wrangling")

        
        return combined_stats


# # Usage example:
# stats = Statistics(df)
# summary = stats.generate_summary()
# print(summary['overall_stats'])
# print(summary['smile_stats'])
# print(summary['term_structure_stats'])