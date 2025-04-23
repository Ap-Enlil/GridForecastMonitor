import os
import pandas as pd
import numpy as np
import requests
from io import StringIO
import pytz
from sklearn.ensemble import IsolationForest
import streamlit as st
from functions import  load_config
def generate_persistence_forecast(df, actual_col, lag_hours=1):
    """
    Generates a persistence forecast by shifting the actual load by a specified number of hours.
    
    Args:
        df (pd.DataFrame): DataFrame with a datetime index.
        actual_col (str): Name of the column with actual load values.
        lag_hours (int): Number of hours to shift (default is 24 for "yesterday's" load).
    
    Returns:
        pd.DataFrame: DataFrame with new forecast columns and error metrics.
    """
    # Create the persistence forecast column.
    print("yomamam")

    df['Persistence Forecast (MW)'] = df[actual_col].shift(lag_hours)

    
  
    # Calculate forecast error and absolute percentage error.
    df['Forecast Error (Persistence)'] = df[actual_col] - df['Persistence Forecast (MW)']
    df['APE (Persistence)'] = (
        abs(df['Forecast Error (Persistence)']) / df[actual_col]
    ).replace([np.inf, -np.inf], np.nan) * 100


    return df
# ------------------------------
# ISO configuration dictionary
# ------------------------------
ISO_CONFIG=load_config()
print(ISO_CONFIG)
# ------------------------------
# Data Downloading
# ------------------------------
def download_data(filenames):
    """
    Download and concatenate CSV data files.
    """
    dfs = []
    base_url = "https://www.eia.gov/electricity/wholesalemarkets/csv/"
    for filename in filenames:
        url = f"{base_url}{filename}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            dfs.append(pd.read_csv(StringIO(response.text), skiprows=3))
        except (requests.exceptions.RequestException, pd.errors.ParserError) as e:
            print(f"Error downloading/parsing {filename}: {e}")
            return None
    return pd.concat(dfs, ignore_index=True) if dfs else None

def preprocess_iso_data(data, iso_key):
    """
    Preprocess raw data for a given ISO:
    - Rename columns as needed.
    - Convert timestamp to datetime and sort.
    - Optionally, generate a persistence forecast.
    - Calculate forecast error and rolling metrics.
    """
    if data is None:
        return None
 
    df = data.copy()
    config = ISO_CONFIG[iso_key]

    # Rename columns if necessary
    if not all(col in df.columns for col in config['required_columns']):
        if 'rename_map' in config:
            df = df.rename(columns=config['rename_map'])
            missing = [col for col in config['required_columns'] if col not in df.columns]
            if missing:
                raise KeyError(f"Missing required columns for {iso_key} after renaming: {missing}")
        else:
            missing = [col for col in config['required_columns'] if col not in df.columns]
            raise KeyError(f"Missing required columns for {iso_key}: {missing}")

    # Convert timestamp and sort
    df['Timestamp'] = pd.to_datetime(df[config['timestamp_column']])
    df = df.sort_values('Timestamp').set_index('Timestamp')
    # (Optional) If a rename_map is provided, rename columns.
  
    if 'rename_map' in config:
    
        df = df.rename(columns=config['rename_map'])


    



    # Check which forecast method to use.
    if config.get("forecast_method", "iso") == "persistence":
        # Use the persistence forecast.
        df['SystemTotal Forecast Load (MW)'] = df[config['actual_column']].shift(24)
        # Calculate forecast error and absolute percentage error.
        df['Forecast Error (MW)'] = df[config['actual_column']] - df['SystemTotal Forecast Load (MW)']
        df['APE (%)'] = (abs(df['Forecast Error (MW)']) / df[config['actual_column']]).replace(np.inf, np.nan) * 100
        df['Percentage Error (%)'] = (df['Forecast Error (MW)'] / df[config['actual_column']]).replace(np.inf, np.nan) * 100
        df = df[df['APE (%)'] <= 20]
        # (Optional) Remove any rows where the forecast is not available.
        df = df.dropna(subset=['SystemTotal Forecast Load (MW)'])

    else:
        # Use the ISO's forecast column as provided.
        df['Forecast Error (MW)'] =  + df[config['actual_column']] -df[config['forecast_column']]
        df['APE (%)'] = (abs(df['Forecast Error (MW)']) / df[config['actual_column']]).replace(np.inf, np.nan) * 100
        df['Percentage Error (%)'] = (df['Forecast Error (MW)'] / df[config['actual_column']]).replace(np.inf, np.nan) * 100
        df = df[df['APE (%)'] <= 10]

    # Calculate additional rolling metrics, for example:
    df['Rolling MAPE (30D)'] = df['APE (%)'].rolling('30D').mean() if 'APE (%)' in df.columns else np.nan
    df['Rolling Avg Error (MW)'] = df['Forecast Error (MW)'].rolling('7D').mean() if 'Forecast Error (MW)' in df.columns else np.nan

    return df.dropna()



# ------------------------------
# Ensure Uniform Hourly Datetime Index
# ------------------------------
def ensure_uniform_hourly_index(df, iso_key):
    """
    Ensure the DataFrame has a complete hourly index in UTC.
    - Remove duplicates.
    - Localize timestamps with proper DST handling.
    - Convert to UTC.
    - Reindex to a full hourly range and interpolate numeric columns.
    """
    config = ISO_CONFIG[iso_key]
    # Remove duplicate indices.
    df = df[~df.index.duplicated(keep='first')]

    # Ensure index is datetime.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')

    # Drop rows where index is NaT.
    df = df[df.index.notna()]
    if df.empty:
        print("DataFrame is empty after dropping NaT indices.")
        return df

    # Localize timestamps if not already localized.
    try:
        if df.index.tz is None:
            df = df.tz_localize(
                config['timezone'],
                ambiguous='infer',
                nonexistent='shift_forward'
            )
        else:
            df = df.tz_convert(config['timezone'])
    except pytz.exceptions.AmbiguousTimeError:
        df = df.tz_localize(
            config['timezone'],
            ambiguous=False,
            nonexistent='shift_forward'
        )

    # Convert to UTC.
    df = df.tz_convert('UTC')

    # Remove duplicates again.
    df = df[~df.index.duplicated(keep='first')]

    # Get start and end for the full range.
    start = df.index.min()
    end = df.index.max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("DataFrame index has NaT for start or end. Check timestamp conversion.")

    # Reindex to a complete hourly range.
    full_range = pd.date_range(start=start, end=end, freq='h', tz='UTC')
    df = df.reindex(full_range)

    # Interpolate numeric columns.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].interpolate(method='time')

    return df

def add_price_data_to_existing_df(existing_df, iso_key, target_column="LMP Difference (USD)"):
    """
    Adds real-time (RT) and day-ahead (DA) price data to an existing DataFrame.

    Args:
        existing_df (pd.DataFrame): The DataFrame to add price data to.
        iso_key (str): The key representing the ISO (e.g., "PJM", "CAISO").
        target_column (str): The name of the column to store the price difference.

    Returns:
        pd.DataFrame: The DataFrame with the added price data, or the original
                      DataFrame if an error occurs.
    """
    config = ISO_CONFIG.get(iso_key)
    if config is None:
        print(f"Error: Configuration for ISO '{iso_key}' not found.")
        st.error(f"Configuration for ISO '{iso_key}' not found.")  # Display error in Streamlit
        return existing_df

    print(f"Configuration for {iso_key}: {config}")

    base_url = "https://www.eia.gov/electricity/wholesalemarkets/csv/"
    price_col = config.get('price_column', 'Bus average LMP')
    print(f"Price column being used: {price_col}")

    # ------------------------------
    # Process Real-Time (RT) Price Data
    # ------------------------------
    rt_dfs = []
    for filename in config.get("rt_filenames", []):
        url = f"{base_url}{filename}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            rt_df = pd.read_csv(StringIO(response.text), skiprows=3, on_bad_lines='skip')

            timestamp_col = 'UTC Timestamp (Interval Ending)'
            if timestamp_col not in rt_df.columns:
                print(f"File {filename} is missing '{timestamp_col}'. Columns found: {rt_df.columns.tolist()}")
                st.error(f"Missing timestamp column in RT file: {filename}")  # Streamlit error
                continue  # Skip to the next file if timestamp is missing

            # Parse timestamps as UTC so the index is already tz-aware.
            rt_df[timestamp_col] = pd.to_datetime(rt_df[timestamp_col], errors='coerce', utc=True)
            rt_df = rt_df.set_index(timestamp_col)
            rt_dfs.append(rt_df)
            print(f"Successfully loaded RT file {filename}")

        except Exception as e:
            print(f"Error downloading or processing RT price file {filename}: {e}")
            st.error(f"Error processing RT file: {filename}")

    if not rt_dfs:
        print("No RT price data loaded. Price data merge skipped.")
        return existing_df

    rt_df = pd.concat(rt_dfs)
    rt_numeric = rt_df.select_dtypes(include=[np.number])
    rt_df_hourly = rt_numeric.resample('h').mean()

    if price_col in rt_df_hourly.columns:
        rt_df_hourly = rt_df_hourly.rename(columns={price_col: "RT " + price_col})
        print(f"RT price column renamed to: RT {price_col}")
    else:
        print(f"RT price column '{price_col}' not found in the data.")
        print(f"Columns in rt_df_hourly: {rt_df_hourly.columns}")
        st.error(f"RT price column '{price_col}' not found in the data after processing.")

    # ------------------------------
    # Process Day-Ahead (DA) Price Data
    # ------------------------------
    da_dfs = []
    for filename in config.get("da_filenames", []):
        url = f"{base_url}{filename}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            da_df = pd.read_csv(StringIO(response.text), skiprows=3, on_bad_lines='skip')

            timestamp_col = 'UTC Timestamp (Interval Ending)'
            if timestamp_col not in da_df.columns:
                print(f"File {filename} is missing '{timestamp_col}'. Columns found: {da_df.columns.tolist()}")
                st.error(f"Missing timestamp column in DA file: {filename}")
                continue

            # Parse timestamps as UTC.
            da_df[timestamp_col] = pd.to_datetime(da_df[timestamp_col], errors='coerce', utc=True)
            da_df = da_df.set_index(timestamp_col)
            da_dfs.append(da_df)
            print(f"Successfully loaded DA file {filename}")

        except Exception as e:
            print(f"Error downloading or processing DA price file {filename}: {e}")
            st.error(f"Error processing DA file: {filename}")

    if not da_dfs:
        print("No DA price data loaded. Price data merge skipped.")
        return existing_df

    da_df = pd.concat(da_dfs)
    da_numeric = da_df.select_dtypes(include=[np.number])
    da_df_hourly = da_numeric.resample('h').mean()

    if price_col in da_df_hourly.columns:
        da_df_hourly = da_df_hourly.rename(columns={price_col: "DA " + price_col})
        print(f"DA price column renamed to: DA {price_col}")
    else:
        print(f"DA price column '{price_col}' not found in the data.")
        print(f"Columns in da_df_hourly: {da_df_hourly.columns}")
        st.error(f"DA price column '{price_col}' not found in the data after processing.")

    # ------------------------------
    # Merge Price Data with the Existing DataFrame
    # ------------------------------
    print(f"Columns in existing_df before RT merge: {existing_df.columns}")
    print(f"Columns in rt_df_hourly before merge: {rt_df_hourly.columns}")
    try:
        merged_df = existing_df.merge(rt_df_hourly[['RT ' + price_col]], left_index=True, right_index=True, how='left')
        print(f"Columns in merged_df after RT merge: {merged_df.columns}")

        print(f"Columns in da_df_hourly before merge: {da_df_hourly.columns}")
        merged_df = merged_df.merge(da_df_hourly[['DA ' + price_col]], left_index=True, right_index=True, how='left')
        print(f"Columns in merged_df after DA merge: {merged_df.columns}")

    except Exception as e:
        print("Error during merging price data:", e)
        st.error("Error during merging price data. Check console for details.")
        return existing_df

    # Compute the target column as the difference between RT and DA prices.
    rt_col_name = "RT " + price_col
    da_col_name = "DA " + price_col

    if rt_col_name in merged_df.columns and da_col_name in merged_df.columns:
        merged_df[target_column] = merged_df[rt_col_name] - merged_df[da_col_name]
    else:
        print(f"Error: Could not calculate '{target_column}' because one or both of the following columns are missing: '{rt_col_name}', '{da_col_name}'")
       # st.error(f"Could not calculate price difference. Check console for details.")

    return merged_df




# ------------------------------
# Load Data for All ISOs
# ------------------------------
def load_all_iso_data():
    """
    Download, preprocess, and (if available) merge price data for all ISOs.
    Returns:
        dict: {iso_key: DataFrame or None}
    """
    iso_data = {}
    ISO_CONFIG=load_config()
    
    for iso_key, cfg in ISO_CONFIG.items():
        print(iso_key)
        # 1. Check timeframe in config
        timeframe = cfg.get("timeframe", "short")  # default to short if missing

        if timeframe == "short":
            if 'filenames' in cfg: # Check if 'filenames' key exists for short timeframe
                raw_data = download_data(cfg['filenames'])

                #print(raw_data)
                if raw_data is not None:
                    try:
                        # Preprocess the load data (short-term).
                        df = preprocess_iso_data(raw_data, iso_key)
                        print(df.columns)
                        print(df.iloc[100:200, 20:])    
                        print("confiddg")
                        # Ensure a uniform hourly UTC index (short-term only).
                        df = ensure_uniform_hourly_index(df, iso_key)

                        # If price data is configured, merge it.
                        if "rt_filenames" in cfg and "da_filenames" in cfg:
                            df = add_price_data_to_existing_df(df, iso_key, target_column="LMP Difference (USD)")

                    except KeyError as e:
                        print(f"Error processing data for {iso_key} (short-term): {e}")
                        df = None

                    iso_data[iso_key] = df

                else:
                    # raw_data is None
                    iso_data[iso_key] = None
            else:
                print(f"Warning: 'filenames' key is missing in ISO_CONFIG for {iso_key} which is configured for 'short' timeframe. Skipping short-term data loading for this ISO.")
                iso_data[iso_key] = None


        else:
            # 2. For non-"short" timeframes, do something else or skip
            # e.g. store None, or call a load_long_term_data() if you have it
            iso_data[iso_key] = None
            # Or if you do have long-term logic:
            # if timeframe == "long":
            #     iso_data[iso_key] = load_long_term_data(iso_key)
            # else:
            #     iso_data[iso_key] = None

    return iso_data


# Example: To load and inspect the data for ERCOT (with price data merged)
import matplotlib.pyplot as plt

if __name__ == "__main__":
    all_iso_data = load_all_iso_data()
    ercot_data = all_iso_data.get("ERCOT")