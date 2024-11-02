# Data 

Within the data folder we have three scripts, namely the DataHandler, the DataValidator, and the DataCollector.

## DataHandler

The DataHandler is a class tha reads options chain data, as logged by the quantconnect trade bar consolidator. The class reads a .txt file and parses it into a pandas dataframe, this is done with the parse_file method. In addition, the get_basic_data method takes the produced dataframe in and computes relevant columns, such as mid prices, days to expiry, implied volatility, log strike, and vega.

## DataValidator

The DataValidator class is a support class for the DataHandler. For one it checks whether the dataframe has relevant columns like timestamp, maturity, and strike but also indexes the the dataframe.
This class can be seen as a health check on the options chain data.

## DataCollector

This class is the basis for further operations as it collects the data using the DataHandler class and organises it into the OptionsData dataclass. The DataCollector features two methods: collect_options_data and get_cached data. 
collecl_options_data uses the DataHandler to parse and get the Quantconnect log file, then splits the data into a call and put dataframe and lastly creates the OptionsData object. The results are then cached.
get_cached_data simply retrieves cahed data if available, otherwise it collects new data.