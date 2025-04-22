import pandas as pd
import numpy as np
import requests
from io import StringIO
import os
import json 
from collections import defaultdict

#=======================
# Data Functions
#=======================
def calculate_mape_long_term(actuals_peak, actuals_energy, forecast_series, forecast_series_energy):
    """Calculates and returns MAPE for long-term forecasts."""

    horizon_errors_peak = defaultdict(list)
    horizon_errors_energy = defaultdict(list)

    max_actual_peak_year = max(actuals_peak.keys()) if actuals_peak else 0
    max_actual_energy_year = max(actuals_energy.keys()) if actuals_energy else 0

    for rel_year, forecasts in forecast_series.items():
        for target_year, forecast_peak in forecasts:
            if target_year in actuals_peak and target_year <= max_actual_peak_year:
                horizon = target_year - rel_year
                if horizon > 5:
                    continue
                error = abs(forecast_peak - actuals_peak[target_year]) / actuals_peak[target_year] * 100
                horizon_errors_peak[horizon].append(error)

    for rel_year, forecasts in forecast_series_energy.items():
        for target_year, forecast_energy in forecasts:
            if target_year in actuals_energy and target_year <= max_actual_energy_year:
                horizon = target_year - rel_year
                if horizon > 5:
                    continue
                error = abs(forecast_energy - actuals_energy[target_year]) / actuals_energy[target_year] * 100
                horizon_errors_energy[horizon].append(error)

    mape_peak_all = np.nanmean([item for sublist in horizon_errors_peak.values() for item in sublist]) if horizon_errors_peak else np.nan
    mape_energy_all = np.nanmean([item for sublist in horizon_errors_energy.values() for item in sublist]) if horizon_errors_energy else np.nan

    return {"Peak Demand MAPE (%)": mape_peak_all, "Energy MAPE (%)": mape_energy_all} # Return a dictionary



def load_config(config_file_path="config.json"):
    """Loads configuration from a JSON file in the same directory as the script."""
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current script
    full_config_path = os.path.join(script_dir, config_file_path) # Construct the full path to config.json

    print(f"Script directory: {script_dir}") # Debug print
    print(f"Full config path: {full_config_path}") # Debug print

    try:
        with open(full_config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: FileNotFoundError: Configuration file not found at: {full_config_path}.  Make sure 'config.json' is in the same folder as your script.") # Debug print
        return None  # Or you could raise an exception
    except json.JSONDecodeError:
        print(f"Error: JSONDecodeError: Could not decode JSON from file: {full_config_path}.  Check if your 'config.json' file is valid JSON.") # Debug print
        return None


if __name__ == "__main__":
    # Create a minimal config.json file for testing if it doesn't exist
    config_file = "config.json"
    if not os.path.exists(config_file):
        print(f"Creating a minimal '{config_file}' file for testing...")
        minimal_config = {"DEBUG_MESSAGE": "Config file loaded successfully!"}
        with open(config_file, 'w') as f:
            json.dump(minimal_config, f, indent=4) # indent for pretty formatting

    loaded_config = load_config()

    if loaded_config:
        print("\nConfig loaded successfully!")
        print("Loaded Config Data:")
        print(json.dumps(loaded_config, indent=4)) # Print nicely formatted JSON
    else:
        print("\nFailed to load config. See error messages above.")


# def download_data(filenames):
#     """Download and concatenate data files"""
#     dfs = []
#     for filename in filenames:
#         url = f"https://www.eia.gov/electricity/wholesalemarkets/csv/{filename}"
#         try:
#             response = requests.get(url)
#             response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
#             dfs.append(pd.read_csv(StringIO(response.text), skiprows=3))
#         except requests.exceptions.RequestException as e:
#             print(f"Error downloading {filename}: {e}")
#             return None  # Or handle the error as appropriate for your application
#         except pd.errors.ParserError as e:
#             print(f"Error parsing {filename}: {e}")
#             return None
#     if dfs:
#         return pd.concat(dfs, ignore_index=True)
#     else:
#         return None

# #=======================
# # Processing Functions
# #=======================

# def preprocess_spp(data):
#     """SPP-specific data processing"""
#     if data is None:
#         return None
#     df = data

#     # Use the expected column name for Central Time
#     required_columns = ['Local Timestamp Central Time (Interval Beginning)',
#                         'SPP Total Actual Load (MW)', 'SPP Total Forecast Load (MW)']
#     if not all(col in df.columns for col in required_columns):
#         missing = [col for col in required_columns if col not in df.columns]
#         raise KeyError(f"Missing required columns for SPP processing: {missing}")

#     # Convert and sort timestamps (handling timezone internally)
#     df['Timestamp'] = pd.to_datetime(df['Local Timestamp Central Time (Interval Beginning)'])

#     # Remove timezone information
#     df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

#     df = df.sort_values('Timestamp').set_index('Timestamp')

#     # Column standardization (if needed)
#     df = df.rename(columns={
#         'SPP Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
#         'SPP Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
#     })

#     # Forecast calculations
#     df['Forecast Error (MW)'] = df['SystemTotal Forecast Load (MW)'] - df['TOTAL Actual Load (MW)']
#     df['APE (%)'] = (abs(df['Forecast Error (MW)']) / df['TOTAL Actual Load (MW)']).replace(np.inf, np.nan) * 100
#     df['Percentage Error (%)'] = (df['Forecast Error (MW)'] / df['TOTAL Actual Load (MW)']).replace(np.inf, np.nan) * 100

#     # Rolling metrics
#     df['Rolling MAPE (30D)'] = df['APE (%)'].rolling('30D').mean()
#     df['Rolling Avg Error (MW)'] = df['Forecast Error (MW)'].rolling('7D').mean()

#     return df.dropna()
# def preprocess_miso(data):
#     """MISO-specific data processing"""
#     if data is None:
#         return None
#     df = data

#     # Use the original expected column name (even though the data is in Eastern Time)
#     required_columns = ['Local Timestamp Central Time (Interval Beginning)',  # We'll handle the timezone internally
#                         'MISO Total Actual Load (MW)', 'MISO Total Forecast Load (MW)']

#     if not all(col in df.columns for col in required_columns):
#         # Rename the Eastern Time column to the expected Central Time name TEMPORARILY
#         df = df.rename(columns={'Local Timestamp Eastern Standard Time (Interval Beginning)': 'Local Timestamp Central Time (Interval Beginning)'})
#         # Recheck if all required columns are present after renaming
#         if not all(col in df.columns for col in required_columns):
#             missing = [col for col in required_columns if col not in df.columns]
#             raise KeyError(f"Missing required columns for MISO processing: {missing}")

#     # Convert and sort timestamps
#     df['Timestamp'] = pd.to_datetime(df['Local Timestamp Central Time (Interval Beginning)'])

#     # Remove timezone information
#     df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)

#     df = df.sort_values('Timestamp').set_index('Timestamp')

#     # Column standardization (if needed) - modify as necessary
#     df = df.rename(columns={
#         'MISO Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
#         'MISO Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
#     })

#     # Forecast calculations
#     df['Forecast Error (MW)'] = df['SystemTotal Forecast Load (MW)'] - df['TOTAL Actual Load (MW)']
#     df['APE (%)'] = (abs(df['Forecast Error (MW)']) / df['TOTAL Actual Load (MW)']).replace(np.inf, np.nan) * 100
#     df['Percentage Error (%)'] = (df['Forecast Error (MW)'] / df['TOTAL Actual Load (MW)']).replace(np.inf, np.nan) * 100

#     # Rolling metrics
#     df['Rolling MAPE (30D)'] = df['APE (%)'].rolling('30D').mean()
#     df['Rolling Avg Error (MW)'] = df['Forecast Error (MW)'].rolling('7D').mean()

#     return df.dropna()
# def preprocess_pjm(data):
#     """PJM-specific data processing"""
#     if data is None:
#         return None
#     df = data

#     # Ensure required columns are present
#     required_columns = ['Local Timestamp Eastern Time (Interval Beginning)',
#                         'PJM Total Actual Load (MW)', 'PJM Total Forecast Load (MW)']
#     if not all(col in df.columns for col in required_columns):
#         missing = [col for col in required_columns if col not in df.columns]
#         raise KeyError(f"Missing required columns for PJM processing: {missing}")

#     # Convert and sort timestamps
#     df['Timestamp'] = pd.to_datetime(df['Local Timestamp Eastern Time (Interval Beginning)'])
#     df = df.sort_values('Timestamp').set_index('Timestamp')

#     # Column standardization (if needed) - modify as necessary
#     df = df.rename(columns={
#         'PJM Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
#         'PJM Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
#     })

#     # Forecast calculations
#     df['Forecast Error (MW)'] = df['SystemTotal Forecast Load (MW)'] - df['TOTAL Actual Load (MW)']
#     df['APE (%)'] = (abs(df['Forecast Error (MW)']) / df['TOTAL Actual Load (MW)']).replace(np.inf, np.nan) * 100
#     df['Percentage Error (%)'] = (df['Forecast Error (MW)'] / df['TOTAL Actual Load (MW)']).replace(np.inf, np.nan) * 100

#     # Rolling metrics
#     df['Rolling MAPE (30D)'] = df['APE (%)'].rolling('30D').mean()
#     df['Rolling Avg Error (MW)'] = df['Forecast Error (MW)'].rolling('7D').mean()

#     return df.dropna()
# def preprocess_ercot(data):
#     """Enhanced ERCOT processing"""
#     if data is None:
#         return None
#     df = data

#     # Ensure required columns are present
#     required_columns = ['Local Timestamp Central Time (Interval Beginning)',
#                         'SystemTotal Forecast Load (MW)', 'TOTAL Actual Load (MW)']
#     if not all(col in df.columns for col in required_columns):
#         missing = [col for col in required_columns if col not in df.columns]
#         raise KeyError(f"Missing required columns for ERCOT processing: {missing}")

#     # Convert and sort timestamps
#     df['Timestamp'] = pd.to_datetime(df['Local Timestamp Central Time (Interval Beginning)'])
#     df = df.sort_values('Timestamp').set_index('Timestamp')

#     # System forecast calculations
#     df['Forecast Error (MW)'] = df['SystemTotal Forecast Load (MW)'] - df['TOTAL Actual Load (MW)']
#     df['APE (%)'] = (abs(df['Forecast Error (MW)']) / df['TOTAL Actual Load (MW)']).replace(np.inf, np.nan) * 100
#     df['Percentage Error (%)'] = (df['Forecast Error (MW)'] / df['TOTAL Actual Load (MW)']).replace(np.inf, np.nan) * 100

#     # Rolling metrics
#     df['Rolling MAPE (30D)'] = df['APE (%)'].rolling('30D').mean()
#     df['Rolling Avg Error (MW)'] = df['Forecast Error (MW)'].rolling('7D').mean()

#     return df.dropna()

# def preprocess_caiso(data):
#     """CAISO-specific data processing (Revised)"""
#     if data is None:
#         return None
#     df = data

#     # Ensure required columns are present
#     required_columns = ['Local Timestamp Pacific Time (Interval Beginning)',
#                         'CAISO Total Actual Load (MW)', 'CAISO Total Forecast Load (MW)']
#     if not all(col in df.columns for col in required_columns):
#         missing = [col for col in required_columns if col not in df.columns]
#         raise KeyError(f"Missing required columns for CAISO processing: {missing}")

#     # Convert and sort timestamps
#     df['Timestamp'] = pd.to_datetime(df['Local Timestamp Pacific Time (Interval Beginning)'])
#     df = df.sort_values('Timestamp').set_index('Timestamp')

#     # Column standardization
#     df = df.rename(columns={
#         'CAISO Total Actual Load (MW)': 'TOTAL Actual Load (MW)',
#         'CAISO Total Forecast Load (MW)': 'SystemTotal Forecast Load (MW)'
#     })

#     # Forecast calculations
#     df['Forecast Error (MW)'] = df['SystemTotal Forecast Load (MW)'] - df['TOTAL Actual Load (MW)']
#     df['APE (%)'] = (abs(df['Forecast Error (MW)']) / df['TOTAL Actual Load (MW)']).replace(np.inf, np.nan) * 100
#     df['Percentage Error (%)'] = (df['Forecast Error (MW)'] / df['TOTAL Actual Load (MW)']).replace(np.inf, np.nan) * 100

#     # Rolling metrics
#     df['Rolling MAPE (30D)'] = df['APE (%)'].rolling('30D').mean()
#     df['Rolling Avg Error (MW)'] = df['Forecast Error (MW)'].rolling('7D').mean()

#     return df.dropna()

# def load_all_iso_data():
#     """
#     Download and preprocess data for ALL ISOs at once.
#     Returns a dict of DataFrames, keyed by ISO name.
#     e.g. {'SPP': df_spp, 'ERCOT': df_ercot, ...}
#     """
#     iso_data = {}
#     for iso_key, cfg in ISO_CONFIG.items():
#         raw_data = download_data(cfg['filenames'])
#         if raw_data is not None:
#             df = cfg['processor'](raw_data)
#             iso_data[iso_key] = df
#         else:
#             iso_data[iso_key] = None
#     return iso_data

