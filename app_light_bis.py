# import sys # Keep for potential future use, though not strictly necessary now
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
# from collections import defaultdict # No longer needed
# import scipy.stats # No longer explicitly required by this version's plots
import statsmodels.api as sm
import statsmodels.tsa.stattools as tsat
# from statsmodels.graphics.tsaplots import plot_acf # Using Plotly implementation instead for consistency
import logging # For logging to terminal
from functools import lru_cache # For potential helper caching if needed

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Module Imports with Error Handling ---
try:
    from functions import load_config
    logging.info("Successfully imported 'load_config' from 'functions.py'.")
except ImportError as e:
    logging.error(f"Error importing 'functions.py': {e}")
    st.error(f"Error importing 'functions.py': {e}")
    st.error("Please ensure 'functions.py' is in the same directory and contains 'load_config'.")
    st.stop()

try:
    from iso_data_integration2 import load_all_iso_data, ensure_uniform_hourly_index, add_price_data_to_existing_df
    logging.info("Successfully imported from 'iso_data_integration2.py'.")
except ImportError as e:
    logging.error(f"Error importing from 'iso_data_integration2.py': {e}")
    st.error(f"Error importing from 'iso_data_integration2.py': {e}")
    st.error("Please ensure 'iso_data_integration2.py' is in the same directory and contains the necessary functions.")
    st.stop()

# --- Configuration ---
try:
    ISO_CONFIG = load_config()
    if ISO_CONFIG is None:
        raise ValueError("load_config returned None")
    logging.info("Configuration loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load configuration from config.json: {e}. Cannot proceed.")
    st.error(f"Failed to load configuration from config.json: {e}. Cannot proceed.")
    st.stop()

TARGET_ISO_KEY = "ERCOT_Load_From_ISO"

# --- Define Standard Column Names (CRITICAL for consistency) ---
ACTUAL_LOAD_COL = 'TOTAL Actual Load (MW)'
FORECAST_LOAD_COL = 'SystemTotal Forecast Load (MW)'
FORECAST_ERROR_COL = 'Forecast Error (MW)' # Actual - Forecast
PRICE_DIFF_COL = "LMP Difference (USD)" # e.g., RT LMP - DA LMP
HOURLY_COST_COL = 'Hourly Cost of Error ($)'
APE_COL = 'Absolute Percentage Error (%)'
HOUR_COL = 'Hour'
DOW_COL = 'DayOfWeek' # Monday=0, Sunday=6
ERROR_UNITS = 'MW' # Units for forecast error (used in plot labels)

REQUIRED_INPUT_COLS = [ACTUAL_LOAD_COL, FORECAST_LOAD_COL] # Base columns needed before derivations

# --- Data Loading (Cached) ---
@st.cache_data(ttl=24 * 60 * 60)
def load_all_data_cached():
    """Loads data for all ISOs defined in config, returns a dict of DataFrames."""
    logging.info("Attempting to load all ISO data (cached)...")
    all_data = {} # Initialize
    try:
        # Attempt calling with config, fallback to parameterless call if TypeError occurs
        try:
            all_data = load_all_iso_data(ISO_CONFIG)
        except TypeError:
            logging.warning("Attempting to call `load_all_iso_data` without arguments due to TypeError.")
            all_data = load_all_iso_data()
        logging.info(f"Data loaded for keys: {list(all_data.keys())}")
        return all_data
    except FileNotFoundError as fnf_err:
        logging.error(f"Error: Required file not found during loading: {fnf_err}")
        st.error(f"Error: Required file not found during loading: {fnf_err}")
        return {}
    except Exception as e:
        logging.error("An unexpected error occurred during data loading:", exc_info=True)
        st.error("An unexpected error occurred during data loading:")
        st.exception(e)
        return {}

# --- Helper: Get Global Date Range ---
# Not using LRU cache here as Streamlit's caching handles the underlying data dict
def get_global_date_range(iso_data_dict):
    """Calculates the minimum start and maximum end date across all loaded DataFrames."""
    logging.debug("Calculating global date range...")
    valid_dates = []
    for key, df in iso_data_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df_index = df.index
            if not isinstance(df_index, pd.DatetimeIndex):
                 try:
                     df_index_conv = pd.to_datetime(df_index, errors='coerce')
                     if not df_index_conv.isna().all():
                         df_index = df_index_conv[~df_index_conv.isna()]
                     else:
                         logging.warning(f"Could not convert index to datetime for ISO '{key}'. Skipping date range calculation.")
                         continue
                 except Exception as conv_err:
                    logging.warning(f"Error converting index for ISO '{key}': {conv_err}. Skipping.")
                    continue

            if isinstance(df_index, pd.DatetimeIndex) and pd.api.types.is_datetime64_any_dtype(df_index) and not df_index.empty:
                current_index = df_index
                # Ensure timezone before comparison
                try:
                    if current_index.tz is None:
                        tz_info_str = ISO_CONFIG.get(key, {}).get('timezone', 'UTC')
                        current_index = current_index.tz_localize(tz_info_str)
                    # Convert to UTC for consistent comparison
                    current_index_utc = current_index.tz_convert('UTC')
                    if not current_index_utc.empty:
                        valid_dates.append(current_index_utc.min())
                        valid_dates.append(current_index_utc.max())
                except Exception as tz_err:
                    logging.warning(f"Could not localize/convert index for {key}: {tz_err}. Skipping.")
                    continue

    if not valid_dates:
        logging.warning("No valid datetime indices found across loaded data to determine range.")
        fallback_end = datetime.date.today()
        fallback_start = fallback_end - datetime.timedelta(days=30)
        return fallback_start, fallback_end

    global_min_utc = min(valid_dates)
    global_max_utc = max(valid_dates)

    logging.debug(f"Global date range determined: {global_min_utc.date()} to {global_max_utc.date()}")
    return global_min_utc.date(), global_max_utc.date()


# --- Data Preparation Function (Tidy Data Principle) ---
# No Streamlit cache here as it runs on filtered data, should be fast.
def prepare_analysis_data(df_in, actual_col, forecast_col, price_diff_col=None):
    """
    Takes the filtered, standardized DataFrame and calculates all necessary derived columns.
    Returns a new DataFrame ready for analysis and plotting.
    """
    if not isinstance(df_in, pd.DataFrame) or df_in.empty:
        logging.warning("prepare_analysis_data received empty or invalid DataFrame.")
        return pd.DataFrame() # Return empty if input is invalid

    logging.debug(f"Preparing analysis data. Input shape: {df_in.shape}")
    df = df_in.copy() # Work on a copy

    # Ensure required numeric columns are numeric, coerce errors to NaN
    df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')
    df[forecast_col] = pd.to_numeric(df[forecast_col], errors='coerce')
    dropped_rows = df[[actual_col, forecast_col]].isna().any(axis=1).sum()
    if dropped_rows > 0:
        logging.warning(f"Dropped {dropped_rows} rows in prepare_analysis_data due to non-numeric load/forecast values.")
        df = df.dropna(subset=[actual_col, forecast_col])
        if df.empty:
            logging.warning("DataFrame became empty after dropping NaN load/forecast values.")
            return pd.DataFrame()

    # 1. Calculate Forecast Error
    df[FORECAST_ERROR_COL] = df[actual_col] - df[forecast_col]

    # 2. Calculate Hourly Cost (Requires Price Difference)
    if price_diff_col and price_diff_col in df.columns:
         df[price_diff_col] = pd.to_numeric(df[price_diff_col], errors='coerce')
         # Calculate cost only where both error and price diff are valid numbers
         # Note: Result will be NaN if either FORECAST_ERROR_COL or price_diff_col is NaN for a row
         df[HOURLY_COST_COL] = df[FORECAST_ERROR_COL] * df[price_diff_col]
         logging.debug(f"Calculated '{HOURLY_COST_COL}'. Non-NaN count: {df[HOURLY_COST_COL].notna().sum()}")
    else:
        df[HOURLY_COST_COL] = np.nan
        logging.info(f"Price difference column '{price_diff_col}' not found or invalid. '{HOURLY_COST_COL}' not calculated.")

    # 3. Calculate Absolute Percentage Error (APE)
    actual_load_safe = df[actual_col].replace(0, np.nan) # Avoid division by zero
    df[APE_COL] = (df[FORECAST_ERROR_COL].abs() / actual_load_safe.abs()) * 100
    df[APE_COL] = df[APE_COL].replace([np.inf, -np.inf], np.nan)
    logging.debug(f"Calculated '{APE_COL}'. Non-NaN count: {df[APE_COL].notna().sum()}")

    # 4. Add Time Features
    if isinstance(df.index, pd.DatetimeIndex):
        df[HOUR_COL] = df.index.hour
        df[DOW_COL] = df.index.dayofweek # Monday=0, Sunday=6
        logging.debug("Added Hour and DayOfWeek columns.")
    else:
        df[HOUR_COL] = np.nan
        df[DOW_COL] = np.nan
        logging.warning("Could not add time features, index is not DatetimeIndex.")

    logging.debug(f"Finished preparing analysis data. Output shape: {df.shape}")
    return df

# --- Memoized Stats Calculations ---
@st.cache_resource(ttl=60*60) # Cache for 1 hour
def calculate_acf(_series, nlags=48, fft=True):
    """Calculates ACF using statsmodels, returns lags, ACF values, and confint."""
    logging.debug(f"Calculating ACF for series (length {len(_series)}), nlags={nlags}")
    if not isinstance(_series, pd.Series) or _series.isnull().all():
        logging.warning("ACF calculation skipped: Input series is invalid or all NaN.")
        return np.array([]), np.array([]), np.array([])

    series_clean = _series.dropna()
    if len(series_clean) < nlags * 2: # Ensure enough data for ACF calculation
        if len(series_clean) > 10: # Arbitrary minimum length for ACF
            nlags = max(1, len(series_clean) // 2 - 1) # Ensure nlags >= 1
            logging.warning(f"Reduced nlags to {nlags} due to insufficient data points ({len(series_clean)}).")
        else:
            logging.warning(f"ACF calculation skipped: Insufficient non-NaN data points ({len(series_clean)}).")
            return np.array([]), np.array([]), np.array([])

    try:
        # Calculate nlags + 1 values (for lags 0 to nlags)
        acf_values, confint = sm.tsa.acf(series_clean, nlags=nlags, alpha=0.05, fft=fft)
        lags = np.arange(nlags + 1) # Lags from 0 to nlags
        logging.debug("ACF calculation successful.")
        return lags, acf_values, confint
    except Exception as e:
        logging.error(f"ACF calculation failed: {e}", exc_info=True)
        st.warning(f"ACF calculation failed: {e}")
        return np.array([]), np.array([]), np.array([])

@st.cache_resource(ttl=60*60)
def calculate_ccf(_series1, _series2, maxlags=48):
    """Calculates CCF using statsmodels for lags -maxlags to +maxlags."""
    logging.debug(f"Calculating CCF between series1 (len {len(_series1)}) and series2 (len {len(_series2)}), maxlags={maxlags}")
    if not isinstance(_series1, pd.Series) or not isinstance(_series2, pd.Series):
         logging.warning("CCF calculation skipped: Invalid input series.")
         return np.array([]), np.array([]), np.nan

    df_temp = pd.DataFrame({'s1': _series1, 's2': _series2}).dropna()
    if len(df_temp) < maxlags * 2:
        if len(df_temp) > 10: # Arbitrary minimum length
            maxlags = max(1, len(df_temp) // 2 - 1) # Ensure maxlags >= 1
            logging.warning(f"Reduced maxlags to {maxlags} due to insufficient overlapping data ({len(df_temp)}).")
        else:
            logging.warning(f"CCF calculation skipped: Insufficient overlapping non-NaN data points ({len(df_temp)}).")
            return np.array([]), np.array([]), np.nan

    s1_clean = df_temp['s1'].values
    s2_clean = df_temp['s2'].values
    n_obs = len(s1_clean)

    try:
        # Calculate CCF for positive lags (s1 leads s2 for lag > 0)
        ccf_pos_lags = sm.tsa.stattools.ccf(s1_clean, s2_clean, adjusted=False)[: maxlags + 1] # Include lag 0
        # Calculate CCF for negative lags (s2 leads s1 for lag > 0 in this call)
        ccf_neg_lags_rev = sm.tsa.stattools.ccf(s2_clean, s1_clean, adjusted=False)[1 : maxlags + 1] # Exclude lag 0

        # Combine: Reverse negative lags, append positive lags
        ccf_final = np.concatenate([ccf_neg_lags_rev[::-1], ccf_pos_lags])
        lags = np.arange(-maxlags, maxlags + 1) # Generate lags from -maxlags to +maxlags

        # Basic check for length consistency
        if len(lags) != len(ccf_final):
             logging.error(f"CCF length mismatch after calculation. Lags: {len(lags)}, CCF: {len(ccf_final)}. Maxlags={maxlags}. Returning empty.")
             return np.array([]), np.array([]), np.nan

        conf_level = 1.96 / np.sqrt(n_obs) if n_obs > 0 else np.nan
        logging.debug("CCF calculation successful.")
        return lags, ccf_final, conf_level
    except Exception as e:
        logging.error(f"CCF calculation failed: {e}", exc_info=True)
        st.warning(f"CCF calculation failed: {e}")
        return np.array([]), np.array([]), np.nan

# --- Short-term Risk Calculation Function (with Timestamps) ---
def calc_next_risk(df_analysis, phi, forecast_window_hours):
    """
    Calculates the projected cost of forecast error for the next N hours.

    Args:
        df_analysis (pd.DataFrame): DataFrame containing prepared analysis data,
                                      must include FORECAST_ERROR_COL and PRICE_DIFF_COL.
        phi (float): Lag-1 autocorrelation of the forecast error (AR(1) parameter).
        forecast_window_hours (int): Number of hours to forecast ahead (e.g., 6, 24).

    Returns:
        tuple: (cost_series, total_cost, avg_hourly_cost, forecast_start_time, forecast_end_time)
               - cost_series (pd.Series): Predicted hourly cost ($) indexed by future timestamps.
                                          Returns empty Series if calculation fails.
               - total_cost (float): Sum of predicted costs over the horizon. NaN if fails.
               - avg_hourly_cost (float): Average predicted hourly cost. NaN if fails.
               - forecast_start_time (pd.Timestamp): Start time of the forecast window. pd.NaT if fails.
               - forecast_end_time (pd.Timestamp): End time of the forecast window. pd.NaT if fails.
    """
    cost_series = pd.Series(dtype=float) # Default empty series
    total_cost = np.nan
    avg_hourly_cost = np.nan
    forecast_start_time = pd.NaT
    forecast_end_time = pd.NaT

    # Input Validation
    if not isinstance(df_analysis, pd.DataFrame) or df_analysis.empty or not isinstance(df_analysis.index, pd.DatetimeIndex):
        logging.warning("calc_next_risk: Input DataFrame is empty or has invalid index.")
        return cost_series, total_cost, avg_hourly_cost, forecast_start_time, forecast_end_time
    if FORECAST_ERROR_COL not in df_analysis.columns:
        logging.warning(f"calc_next_risk: Missing required column '{FORECAST_ERROR_COL}'.")
        return cost_series, total_cost, avg_hourly_cost, forecast_start_time, forecast_end_time
    if PRICE_DIFF_COL not in df_analysis.columns:
        logging.warning(f"calc_next_risk: Missing required column '{PRICE_DIFF_COL}'. Costs will be NaN.")
        # Allow calculation without price, costs will be NaN
    if not isinstance(forecast_window_hours, int) or forecast_window_hours <= 0:
         logging.warning(f"calc_next_risk: Invalid forecast_window_hours ({forecast_window_hours}).")
         return cost_series, total_cost, avg_hourly_cost, forecast_start_time, forecast_end_time
    if not pd.notna(phi) or not (-1 <= phi <= 1):
         logging.warning(f"calc_next_risk: Invalid phi value ({phi}). Cannot project error.")
         # Still return NaNs but capture timestamps if possible
         try:
             last_timestamp = df_analysis.index[-1]
             start_forecast = last_timestamp + pd.Timedelta(hours=1)
             horizon_index = pd.date_range(start=start_forecast, periods=forecast_window_hours, freq='H', tz=df_analysis.index.tz)
             forecast_start_time = horizon_index.min()
             forecast_end_time = horizon_index.max()
         except Exception:
             pass # Ignore errors here, just trying to get timestamps
         return cost_series, total_cost, avg_hourly_cost, forecast_start_time, forecast_end_time

    try:
        # Get Last Error
        error_series = df_analysis[FORECAST_ERROR_COL].dropna()
        if error_series.empty:
            logging.warning("calc_next_risk: No valid historical errors found.")
            return cost_series, total_cost, avg_hourly_cost, forecast_start_time, forecast_end_time
        last_err = error_series.iloc[-1]
        if not pd.notna(last_err):
             logging.warning("calc_next_risk: Last historical error is NaN.")
             return cost_series, total_cost, avg_hourly_cost, forecast_start_time, forecast_end_time

        # Get Price Difference Series (may contain NaNs)
        price_diff_series = df_analysis.get(PRICE_DIFF_COL) # Use .get for safety

        # Define Horizon & Timestamps
        last_timestamp = df_analysis.index[-1]
        start_forecast = last_timestamp + pd.Timedelta(hours=1)
        horizon_index = pd.date_range(start=start_forecast, periods=forecast_window_hours, freq='H', tz=df_analysis.index.tz)
        forecast_start_time = horizon_index.min()
        forecast_end_time = horizon_index.max()

        # Predict Future Errors (AR(1) decay)
        pred_err = [last_err * (phi ** h) for h in range(1, forecast_window_hours + 1)] # Corrected: phi^h for h steps ahead

        # Predict Future Price Difference (Simple: last 6h rolling average, aligned)
        pred_price_diff = 0.0 # Default
        if price_diff_series is not None and price_diff_series.notna().any():
            # Align with error series index to get most recent values before calculating rolling mean
            price_diff_aligned = price_diff_series.reindex(error_series.index)
            # Calculate rolling mean on aligned series
            rolling_mean_price = price_diff_aligned.rolling(6, min_periods=1).mean()
            if not rolling_mean_price.empty and pd.notna(rolling_mean_price.iloc[-1]):
                pred_price_diff = rolling_mean_price.iloc[-1]
            else:
                logging.warning("calc_next_risk: Could not calculate valid recent rolling avg price diff. Using 0.")
        else:
             logging.warning(f"calc_next_risk: No valid data in '{PRICE_DIFF_COL}'. Using 0 for predicted price difference.")

        # Calculate Predicted Hourly Cost
        cost_values = np.array(pred_err) * pred_price_diff
        cost_series = pd.Series(cost_values, index=horizon_index, name="Predicted Hourly Cost ($)")

        # Calculate Aggregates
        total_cost = cost_series.sum()
        avg_hourly_cost = cost_series.mean()

        logging.info(f"Calculated risk for {forecast_start_time} to {forecast_end_time}: Total=${total_cost:,.0f}, Avg=${avg_hourly_cost:,.2f}/h (phi={phi:.3f}, last_err={last_err:.1f}, pred_price_diff={pred_price_diff:.2f})")
        return cost_series, total_cost, avg_hourly_cost, forecast_start_time, forecast_end_time

    except Exception as e:
        logging.error(f"Error during calc_next_risk: {e}", exc_info=True)
        # Attempt to return timestamps even if calculation fails
        try:
             last_timestamp = df_analysis.index[-1]
             start_forecast = last_timestamp + pd.Timedelta(hours=1)
             horizon_index = pd.date_range(start=start_forecast, periods=forecast_window_hours, freq='H', tz=df_analysis.index.tz)
             forecast_start_time = horizon_index.min()
             forecast_end_time = horizon_index.max()
        except Exception:
            pass # Ignore errors, return NaT if needed
        return pd.Series(dtype=float), np.nan, np.nan, forecast_start_time, forecast_end_time


# --- Streamlit App Setup ---
st.set_page_config(layout="wide")
st.title(f"Grid Forecast Monitor: Accuracy & Financial Impact of Load Prediction Error")

# --- Load Data ---
all_iso_data = load_all_data_cached()

# --- Check Data Loading Status ---
if not all_iso_data:
    logging.error("Data loading failed or returned no data. Aborting application.")
    st.error("Data loading failed or returned no data. Aborting application.")
    st.stop()

# --- Get ERCOT Data and Validate ---
logging.info(f"Attempting to load data for key: '{TARGET_ISO_KEY}'")
df_ercot_raw = all_iso_data.get(TARGET_ISO_KEY)

# --- Initial Validation (Existence, Type, Emptiness) ---
if df_ercot_raw is None:
    logging.error(f"Could not retrieve data for the key '{TARGET_ISO_KEY}'.")
    st.error(f"Could not retrieve data for the key '{TARGET_ISO_KEY}'. Check 'config.json' and data loading functions.")
    st.info(f"Available keys in loaded data: {list(all_iso_data.keys())}")
    st.stop()
if not isinstance(df_ercot_raw, pd.DataFrame):
    logging.error(f"Data retrieved for '{TARGET_ISO_KEY}' is not a Pandas DataFrame (Type: {type(df_ercot_raw)}).")
    st.error(f"Data retrieved for '{TARGET_ISO_KEY}' is not a Pandas DataFrame (Type: {type(df_ercot_raw)}). Cannot proceed.")
    st.stop()
if df_ercot_raw.empty:
    logging.error(f"Data for '{TARGET_ISO_KEY}' was loaded but the DataFrame is empty.")
    st.error(f"Data for '{TARGET_ISO_KEY}' was loaded but the DataFrame is empty. Cannot proceed.")
    st.stop()
logging.info(f"Successfully retrieved raw data for '{TARGET_ISO_KEY}'. Shape: {df_ercot_raw.shape}")

# --- Deeper Validation (Index, Timezone, Required Input Columns) ---
try:
    # 1. Ensure DatetimeIndex
    if not isinstance(df_ercot_raw.index, pd.DatetimeIndex):
        logging.info("Attempting to convert ERCOT DataFrame index to DatetimeIndex.")
        original_index_name = df_ercot_raw.index.name
        df_ercot_raw.index = pd.to_datetime(df_ercot_raw.index, errors='coerce')
        initial_rows = len(df_ercot_raw)
        df_ercot_raw = df_ercot_raw[df_ercot_raw.index.notna()]
        dropped_rows = initial_rows - len(df_ercot_raw)
        if dropped_rows > 0:
             logging.warning(f"Dropped {dropped_rows} rows due to invalid date conversion in the index.")
             st.warning(f"Dropped {dropped_rows} rows due to invalid date conversion in the index.")
        if df_ercot_raw.empty:
            raise ValueError("Index conversion to DatetimeIndex failed or resulted in an empty DataFrame.")
        if original_index_name:
             df_ercot_raw.index.name = original_index_name
        logging.info("Index successfully converted to DatetimeIndex.")

    # 2. Ensure Timezone Awareness
    if df_ercot_raw.index.tz is None:
        tz_info = ISO_CONFIG.get(TARGET_ISO_KEY, {}).get('timezone', 'UTC') # Default to UTC if not specified
        logging.info(f"Localizing ERCOT data timezone to '{tz_info}'.")
        try:
            df_ercot_raw = df_ercot_raw.tz_localize(tz_info)
        except Exception as tz_err:
             logging.error(f"Failed to localize timezone to {tz_info}: {tz_err}. Attempting UTC fallback.", exc_info=True)
             st.error(f"Failed to localize timezone to {tz_info}: {tz_err}. Trying UTC fallback.")
             try:
                 logging.warning("Attempting to localize to UTC as fallback.")
                 df_ercot_raw = df_ercot_raw.tz_localize('UTC')
             except Exception as utc_err:
                  logging.error(f"Fallback UTC localization also failed: {utc_err}. Stopping.", exc_info=True)
                  st.error(f"Fallback UTC localization also failed: {utc_err}. Stopping.")
                  st.stop()
    else:
        logging.info(f"ERCOT data timezone is already set: {df_ercot_raw.index.tz}")

    # 3. Check Required *Input* Columns
    logging.info(f"Checking for required input columns: {REQUIRED_INPUT_COLS}")
    missing_cols = [col for col in REQUIRED_INPUT_COLS if col not in df_ercot_raw.columns]
    if missing_cols:
        logging.error(f"ERCOT data is missing required columns for core analysis: {missing_cols}")
        st.error(f"ERCOT data is missing required columns for core analysis: {missing_cols}")
        st.info(f"Available columns: {df_ercot_raw.columns.tolist()}")
        st.stop()
    logging.info("Required input columns found.")

except Exception as e:
    logging.error(f"Critical error during ERCOT data validation: {e}", exc_info=True)
    st.error(f"Critical error during ERCOT data validation: {e}")
    st.exception(e)
    st.stop()


# --- Sidebar Setup ---
st.sidebar.header("Analysis Options")

# Date Range Selection
st.sidebar.subheader("Date Range")
global_min_date, global_max_date = get_global_date_range(all_iso_data)

if global_min_date is None or global_max_date is None:
    st.sidebar.error("Could not determine a valid global date range. Using ERCOT's range.")
    if not df_ercot_raw.empty and isinstance(df_ercot_raw.index, pd.DatetimeIndex):
        ercot_min_date = df_ercot_raw.index.min().date()
        ercot_max_date = df_ercot_raw.index.max().date()
    else: # Absolute fallback
        ercot_max_date = datetime.date.today()
        ercot_min_date = ercot_max_date - datetime.timedelta(days=365)
    global_min_date, global_max_date = ercot_min_date, ercot_max_date

# Determine effective range for ERCOT data, constrained by global range
ercot_min_date_avail = global_min_date
ercot_max_date_avail = global_max_date
if not df_ercot_raw.empty and isinstance(df_ercot_raw.index, pd.DatetimeIndex):
     try:
         # Get ERCOT's actual min/max dates
         ercot_min_date_calc = df_ercot_raw.index.min().date()
         ercot_max_date_calc = df_ercot_raw.index.max().date()
         # Clamp ERCOT range by global range
         ercot_min_date_avail = max(global_min_date, ercot_min_date_calc)
         ercot_max_date_avail = min(global_max_date, ercot_max_date_calc)
         # Ensure valid range (start <= end)
         if ercot_min_date_avail > ercot_max_date_avail:
             ercot_min_date_avail, ercot_max_date_avail = global_min_date, global_max_date # Fallback to global
     except Exception as date_err:
         logging.warning(f"Could not refine date range using ERCOT data: {date_err}")
         pass # Use the global range

# Sensible defaults: last 7 days within available ERCOT range
default_end = ercot_max_date_avail
default_start = max(ercot_min_date_avail, default_end - datetime.timedelta(days=7)) # Default to 7 days or start of data

# Ensure start is not after end
if default_start > default_end:
    default_start = ercot_min_date_avail # Fallback to min available date if calculation fails

st.sidebar.info(f"Data Available: {ercot_min_date_avail} to {ercot_max_date_avail}")
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=ercot_min_date_avail, max_value=ercot_max_date_avail)
end_date = st.sidebar.date_input("End Date", value=default_end, min_value=start_date, max_value=ercot_max_date_avail)

# --- REMOVED: Sigma Threshold Slider ---
# st.sidebar.subheader("Alert Thresholds")
# sigma_threshold_k = st.sidebar.slider(...) # REMOVED

# --- NEW: Forecast Window Slider ---
st.sidebar.subheader("Short-Term Risk Forecast")
forecast_window_hours = st.sidebar.slider(
    "Forecast Horizon (Hours)",
    min_value=6, max_value=48, value=6, step=1, # Default to 6h
    help="Select the number of hours ahead to forecast the potential cost of error."
)

# --- Filter ERCOT Data based on Selection ---
df_ercot_filtered = pd.DataFrame()
try:
    if start_date and end_date and isinstance(df_ercot_raw.index, pd.DatetimeIndex):
        # Ensure raw data timezone exists (validated earlier)
        tz = df_ercot_raw.index.tz
        if tz is None:
            # This *shouldn't* happen after validation, but handle defensively
            logging.error("Timezone lost from validated raw data before filtering. Stopping.")
            st.error("Internal error: Timezone information lost. Cannot filter data correctly.")
            st.stop()

        # Localize start/end dates to the DATA's timezone for comparison
        # Combine date with min/max time, then localize
        start_dt = pd.Timestamp(datetime.datetime.combine(start_date, datetime.time.min), tz=tz)
        # End date is inclusive, so filter up to the very end of that day
        end_dt = pd.Timestamp(datetime.datetime.combine(end_date, datetime.time.max), tz=tz)

        logging.info(f"Filtering data from {start_dt} to {end_dt} (inclusive, timezone: {tz})")
        mask = (df_ercot_raw.index >= start_dt) & (df_ercot_raw.index <= end_dt)
        df_ercot_filtered = df_ercot_raw.loc[mask].copy()

        logging.info(f"Filtered data for {start_date} to {end_date}. Shape: {df_ercot_filtered.shape}")
        if df_ercot_filtered.empty:
            st.warning(f"No ERCOT data available between {start_date} and {end_date}. Adjust the date range or check raw data.")
            # Don't stop, let the rest handle the empty DataFrame gracefully
    else:
         logging.warning("Could not perform date filtering. Ensure dates are selected and raw data index is valid.")

except Exception as e:
     logging.error(f"Error during date filtering: {e}", exc_info=True)
     st.error(f"Error during date filtering: {e}")
     st.exception(e)
     st.stop() # Stop if filtering fails fundamentally

# --- Standardize Index (Hourly) ---
df_ercot_std = pd.DataFrame()
if not df_ercot_filtered.empty:
    logging.info("Standardizing ERCOT data to uniform hourly index...")
    try:
        expected_tz = df_ercot_filtered.index.tz
        df_ercot_std = ensure_uniform_hourly_index(df_ercot_filtered, TARGET_ISO_KEY)

        if df_ercot_std.empty:
            logging.warning("ERCOT data became empty after ensuring uniform hourly index. Check for large gaps or resampling issues.")
        else:
            logging.info(f"Index standardized. Shape after standardization: {df_ercot_std.shape}")
            # Verify timezone consistency after standardization
            if df_ercot_std.index.tz != expected_tz:
                 logging.warning(f"Timezone changed during standardization! Expected {expected_tz}, got {df_ercot_std.index.tz}. Attempting conversion back.")
                 try:
                      df_ercot_std.index = df_ercot_std.index.tz_convert(expected_tz)
                      logging.info(f"Timezone successfully converted back to {expected_tz}.")
                 except Exception as tz_reconv_err:
                      logging.error(f"Failed to convert timezone back after standardization: {tz_reconv_err}")
                      st.error("Timezone consistency issue after data standardization. Results might be affected.")
            elif df_ercot_std.index.tz is None and expected_tz is not None:
                 logging.warning(f"Timezone became naive during standardization! Expected {expected_tz}. Attempting re-localization.")
                 try:
                     df_ercot_std = df_ercot_std.tz_localize(expected_tz)
                     logging.info(f"Timezone successfully re-localized back to {expected_tz}.")
                 except Exception as tz_reloc_err:
                      logging.error(f"Failed to re-localize timezone back after standardization: {tz_reloc_err}")
                      st.error("Timezone became naive after data standardization. Results might be affected.")

    except Exception as e:
        logging.error(f"Error during index standardization for ERCOT: {e}", exc_info=True)
        st.error(f"Error during index standardization for ERCOT: {e}")
        st.exception(e)
        # If standardization fails, proceed with the filtered data but warn? Or stop?
        # Let's try proceeding with filtered, but analysis might be less robust.
        logging.warning("Proceeding with non-standardized index due to error. Analysis might be affected.")
        df_ercot_std = df_ercot_filtered.copy() # Fallback to filtered data
else:
    logging.info("Skipping index standardization as filtered ERCOT data is empty.")

# --- Add Price Data (if available for ERCOT) ---
df_ercot_priced = pd.DataFrame()
if not df_ercot_std.empty:
    logging.info(f"Attempting to add/calculate price difference column: '{PRICE_DIFF_COL}'")
    try:
        # Pass a copy to avoid modifying df_ercot_std directly if function has side effects
        df_ercot_priced = add_price_data_to_existing_df(df_ercot_std.copy(), TARGET_ISO_KEY, target_column=PRICE_DIFF_COL)
        if PRICE_DIFF_COL not in df_ercot_priced.columns:
            logging.warning(f"Could not find or calculate price difference data ('{PRICE_DIFF_COL}') for ERCOT in the selected range.")
            st.info(f"Price difference column '{PRICE_DIFF_COL}' not found or calculated. Price-related analysis will be limited.")
        elif df_ercot_priced[PRICE_DIFF_COL].isnull().all():
            logging.warning(f"Price difference column '{PRICE_DIFF_COL}' was added but contains only NaN values.")
            st.info(f"Price difference column '{PRICE_DIFF_COL}' contains only missing values. Price-related analysis will be limited.")
        else:
            logging.info(f"Successfully added/found '{PRICE_DIFF_COL}' column.")
    except KeyError as ke:
        logging.warning(f"Could not calculate '{PRICE_DIFF_COL}'. Missing required input column(s): {ke}. Price analysis will be limited.")
        st.warning(f"Could not calculate '{PRICE_DIFF_COL}'. Missing required input column(s): {ke}. Price analysis will be limited.")
        df_ercot_priced = df_ercot_std.copy() # Proceed without price column
    except FileNotFoundError as fnf_err:
        logging.warning(f"Price data file not found: {fnf_err}. Price analysis will be limited.")
        st.warning(f"Price data file not found: {fnf_err}. Price analysis will be limited.")
        df_ercot_priced = df_ercot_std.copy() # Proceed without price column
    except Exception as e:
        logging.error(f"An unexpected error occurred while adding price data for ERCOT: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while adding price data for ERCOT: {e}")
        st.exception(e)
        df_ercot_priced = df_ercot_std.copy() # Proceed without price column
else:
    logging.info("Skipping price data addition as standardized ERCOT data is empty.")
    df_ercot_priced = df_ercot_std.copy() # Ensure it's assigned even if empty

# --- Prepare Final Analysis DataFrame ---
logging.info("Preparing final analysis DataFrame...")
df_analysis = prepare_analysis_data(
    df_ercot_priced,
    actual_col=ACTUAL_LOAD_COL,
    forecast_col=FORECAST_LOAD_COL,
    price_diff_col=PRICE_DIFF_COL # Pass the name, function will check if it exists
)

# =============================
# == Main Application Layout ==
# =============================

if df_analysis.empty:
    logging.warning("No data available for analysis after filtering and preparation.")
    st.warning("No data available for analysis after filtering and preparation. Please adjust the date range or check data sources.")
    st.stop() # Stop if no data to analyze

# --- Section 1: KPI Cards & Analytics Addons ---
st.subheader("Forecast Performance KPIs & Alerts")

# Calculate Core KPIs from df_analysis
kpi_bias = df_analysis[FORECAST_ERROR_COL].mean() if FORECAST_ERROR_COL in df_analysis else np.nan
kpi_mape = df_analysis[APE_COL].mean() if APE_COL in df_analysis else np.nan
avg_load = df_analysis[ACTUAL_LOAD_COL].mean() if ACTUAL_LOAD_COL in df_analysis else np.nan

# Initialize historical cost KPIs and the cost per MW <<< CORRECTION: Added Initialization
kpi_cost_hist = np.nan
cost_underforecast_hist = np.nan
cost_overforecast_hist = np.nan
avg_historical_cost_per_mw = np.nan # Initialize here

required_cols_for_cost = [FORECAST_ERROR_COL, HOURLY_COST_COL]
if all(col in df_analysis.columns for col in required_cols_for_cost) and df_analysis[HOURLY_COST_COL].notna().any():
    # Calculate total historical cost
    kpi_cost_hist = df_analysis[HOURLY_COST_COL].sum()
    # Calculate cost from under-forecasting (Error > 0)
    underforecast_mask = df_analysis[FORECAST_ERROR_COL] > 0
    cost_underforecast_hist = df_analysis.loc[underforecast_mask, HOURLY_COST_COL].sum()
    # Calculate cost from over-forecasting (Error < 0)
    overforecast_mask = df_analysis[FORECAST_ERROR_COL] < 0
    cost_overforecast_hist = df_analysis.loc[overforecast_mask, HOURLY_COST_COL].sum()
    logging.info(f"Calculated Historical Costs based on RT DA difference: Total={kpi_cost_hist:.0f}, Underforecast={cost_underforecast_hist:.0f}, Overforecast={cost_overforecast_hist:.0f}")

    # --- <<< CORRECTION: Added the calculation block here >>> ---
    # --- Calculate Average Historical Cost per MW Error ($X) ---
    try:
        # Create temporary series with absolute values, avoiding division by zero/NaN
        abs_cost = df_analysis[HOURLY_COST_COL].abs()
        abs_error_mw = df_analysis[FORECAST_ERROR_COL].abs()

        # Filter for valid rows where error is non-zero and cost is not NaN
        # Using small tolerance for error > 0 check to avoid float precision issues
        valid_mask = (abs_error_mw > 1e-6) & abs_cost.notna()
        if valid_mask.any():
            cost_per_mw = abs_cost[valid_mask] / abs_error_mw[valid_mask]
            # Assign the calculated value to the initialized variable
            avg_historical_cost_per_mw = cost_per_mw.mean()
            logging.info(f"Calculated Avg Historical Cost per MW Error: ${avg_historical_cost_per_mw:.2f}/MW")
        else:
             # avg_historical_cost_per_mw remains np.nan (from initialization)
             logging.warning("Could not calculate Avg Historical Cost per MW: No valid non-zero errors with associated costs found.")
    except Exception as e_cost_mw:
        # avg_historical_cost_per_mw remains np.nan (from initialization)
        logging.error(f"Error calculating Avg Historical Cost per MW: {e_cost_mw}", exc_info=True)
    # <<< --- End of added calculation block --- >>>

else:
    # This else corresponds to the outer `if all(col in df_analysis.columns ...)`
    # avg_historical_cost_per_mw remains np.nan (from initialization)
    logging.warning(f"Could not calculate historical cost KPIs or Avg Cost/MW. Missing columns ({required_cols_for_cost}) or no valid data in '{HOURLY_COST_COL}'.")


# --- Calculate Phi (Lag-1 ACF) for use in alerts and risk forecast ---
# (This comes AFTER cost calculations)
phi = np.nan # Initialize phi
lag1_acf = np.nan
lag1_outside_ci = False
if FORECAST_ERROR_COL in df_analysis:
    # ... (rest of ACF calculation as before) ...
    error_series_clean = df_analysis[FORECAST_ERROR_COL].dropna()
    if len(error_series_clean) > 10: # Arbitrary minimum length
        acf_lags, acf_values, acf_confint = calculate_acf(error_series_clean, nlags=1) # Only need lag 1
        if len(acf_values) > 1:
            lag1_acf = acf_values[1]
            phi = lag1_acf # Use lag-1 ACF as phi
            if acf_confint is not None and acf_confint.ndim == 2 and acf_confint.shape[0] > 1:
                ci_lower_lag1 = acf_confint[1, 0]
                ci_upper_lag1 = acf_confint[1, 1]
                lag1_outside_ci = pd.notna(lag1_acf) and pd.notna(ci_lower_lag1) and pd.notna(ci_upper_lag1) and \
                                  (lag1_acf < ci_lower_lag1 or lag1_acf > ci_upper_lag1)
                logging.info(f"Calculated Lag-1 ACF (phi) = {phi:.3f}. CI=[{ci_lower_lag1:.3f}, {ci_upper_lag1:.3f}]. Outside CI: {lag1_outside_ci}")
            else:
                logging.warning("Could not obtain confidence intervals for Lag-1 ACF.")
        else:
            logging.warning("ACF calculation did not return lag-1 value.")
    else:
        logging.warning(f"Insufficient data ({len(error_series_clean)} points) to calculate reliable Lag-1 ACF.")


# Define Thresholds (adjust as needed)
# ... (threshold definitions as before) ...
mape_threshold_warn = 3.0
mape_threshold_alert = 4.0
bias_perc_threshold_warn = 0.01 # 1%
bias_perc_threshold_alert = 0.02 # 2%
bias_abs_threshold_warn = abs(avg_load * bias_perc_threshold_warn) if pd.notna(avg_load) and avg_load != 0 else np.inf
bias_abs_threshold_alert = abs(avg_load * bias_perc_threshold_alert) if pd.notna(avg_load) and avg_load != 0 else np.inf


# Determine KPI colors
# ... (color logic as before) ...
mape_color = "normal"
if pd.notna(kpi_mape):
    if kpi_mape > mape_threshold_alert: mape_color = "inverse"
    elif kpi_mape > mape_threshold_warn: mape_color = "off"

bias_color = "normal"
if pd.notna(kpi_bias) and pd.notna(avg_load) and avg_load != 0: # Check avg_load exists and non-zero for percentage comparison
    if abs(kpi_bias) > bias_abs_threshold_alert: bias_color = "inverse"
    elif abs(kpi_bias) > bias_abs_threshold_warn: bias_color = "off"
elif pd.notna(kpi_bias) and abs(kpi_bias) > 100: # Fallback: Alert if bias > 100MW absolute if avg_load not available
    bias_color = "off" # Use 'off' color for fallback absolute threshold


# --- Bias-Correction Alert ---
# ... (alert logic as before) ...
bias_alert_triggered = False
if pd.notna(kpi_bias) and pd.notna(avg_load) and avg_load != 0 and \
   (abs(kpi_bias) > bias_abs_threshold_warn) and lag1_outside_ci:
    bias_alert_triggered = True
    bias_perc_of_load = (kpi_bias / avg_load) * 100 if avg_load != 0 else np.nan
    st.error(f"**Bias Correction Alert:** Forecast bias ({kpi_bias:,.1f} MW, {bias_perc_of_load:.1f}% of avg load) is significant (> ±{bias_perc_threshold_warn*100:.0f}%) "
             f"AND errors show short-term persistence (Lag-1 ACF = {lag1_acf:.2f}, outside 95% CI). Review forecast inputs or consider bias adjustments.")
    logging.warning("Bias Correction Alert triggered.")


# --- Calculate Short-Term Risk using the new function ---
# (This comes AFTER cost and ACF calculations)
pred_cost_series, pred_total_cost, pred_avg_hourly_cost, forecast_start_time, forecast_end_time = calc_next_risk(
    df_analysis, phi, forecast_window_hours
)


# --- Display Historical KPIs (Using 6 columns now) ---
# (This section now correctly comes AFTER avg_historical_cost_per_mw is defined/calculated.)
col1, col2, col3, col4, col5, col6 = st.columns(6)

# ... (rest of the code displaying col1 to col5 as before) ...

with col1: # MAPE
    st.metric(label="MAPE (%)", value=f"{kpi_mape:.2f}%" if pd.notna(kpi_mape) else "N/A",
              delta_color=mape_color, help="Mean Absolute Percentage Error (vs Actual Load). Lower is better.")
with col2: # Bias
    bias_help_text = f"Mean Forecast Error (Actual - Forecast) over selected period ({start_date} to {end_date})."
    if pd.notna(avg_load) and avg_load != 0:
         bias_help_text += f" Alert if > ±{bias_perc_threshold_warn*100:.0f}% of Avg Load ({avg_load:,.0f} MW) & persistent."
    st.metric(label="Hist. Bias (MW)", value=f"{kpi_bias:,.1f}" if pd.notna(kpi_bias) else "N/A",
              delta_color=bias_color, help=bias_help_text)
with col3: # Historical Cost Under
    if pd.notna(cost_underforecast_hist):
         st.metric(label="Hist. Under-FC Cost ($)", value=f"${cost_underforecast_hist:,.0f}",
                   help="Sum of HISTORICAL Cost when Actual > Forecast (Error > 0). Positive implies cost.")
    else:
         st.metric(label="Hist. Under-FC Cost ($)", value="N/A", help="Requires error & price data.")
with col4: # Historical Cost Over
    if pd.notna(cost_overforecast_hist):
        st.metric(label="Hist. Over-FC Cost ($)", value=f"${cost_overforecast_hist:,.0f}",
                  help="Sum of HISTORICAL Cost when Actual < Forecast (Error < 0). Negative implies cost (or gain if prices inverted).")
    else:
        st.metric(label="Hist. Over-FC Cost ($)", value="N/A", help="Requires error & price data.")
with col5: # Average Actual Load
     st.metric(label="Avg Actual Load (MW)", value=f"{avg_load:,.0f}" if pd.notna(avg_load) else "N/A",
               help=f"Average actual load over the selected period ({start_date} to {end_date}).")


# --- Corrected 6th Column Display ---
with col6:
    # Prepare the $X value string, handling NaN (avg_historical_cost_per_mw is now guaranteed to exist)
    hist_cost_mw_str = f"${avg_historical_cost_per_mw:,.2f}" if pd.notna(avg_historical_cost_per_mw) else "N/A"
    hist_cost_mw_val_for_help = f"${avg_historical_cost_per_mw:,.2f}" if pd.notna(avg_historical_cost_per_mw) else "an undetermined amount"

    # Create the help text
    help_cost_mw = "Could not calculate average cost per MW error (requires valid cost and error data)."
    if pd.notna(avg_historical_cost_per_mw):
         help_cost_mw = (f"Based on the selected period ({start_date} to {end_date}): "
                         f"Every absolute MW mis-forecast cost the desk approx. "
                         f"**{hist_cost_mw_val_for_help}** on average.")

    st.metric(label="Hist. Cost / MW Error ($)",
               value=hist_cost_mw_str,
               help=help_cost_mw)

st.markdown("---") # Separator before the forecast KPIs



# --- Display Predicted Risk KPIs (with Timestamp Info) ---
# Format timestamps nicely
time_format = '%Y-%m-%d %H:%M %Z'
start_time_str = forecast_start_time.strftime(time_format) if pd.notna(forecast_start_time) else "N/A"
end_time_str = forecast_end_time.strftime(time_format) if pd.notna(forecast_end_time) else "N/A"

st.subheader(f"Predicted Risk ({forecast_window_hours} Hours: {start_time_str} to {end_time_str})")

col_risk1, col_risk2 = st.columns(2)

# With one of the options, for example Option 1:
help_total = f"Estimated total financial risk ($) from forecast errors over the next {forecast_window_hours} hours, if recent trends continue."

# Then use it in the metric:
with col_risk1:
    st.metric(label=f"Next {forecast_window_hours}h Cost (Total)",
              value=f"${pred_total_cost:,.0f}" if pd.notna(pred_total_cost) else "N/A",
              help=help_total) # <--- Uses the new help text

with col_risk2:
    help_avg = f"Average predicted hourly cost over the next {forecast_window_hours} hours."
    st.metric(label=f"Avg $/h Risk (Next {forecast_window_hours}h)",
              value=f"${pred_avg_hourly_cost:,.2f}" if pd.notna(pred_avg_hourly_cost) else "N/A",
              help=help_avg)






































# --- Display Hourly Breakdown in Expander ---
with st.expander(f"View Predicted Hourly Cost Breakdown (Next {forecast_window_hours}h)"):
    if not pred_cost_series.empty:
        st.caption("Predicted cost for each hour based on AR(1) error decay and recent average price difference.")
        # Option 1: Dataframe
        df_display = pred_cost_series.to_frame()
        df_display.index.name = "Timestamp"
        df_display.columns = ["Predicted Cost ($)"]
        st.dataframe(df_display.style.format({"Predicted Cost ($)": "${:,.2f}"}), use_container_width=True)

        # Option 2: Plotly Bar Chart
        try:
            fig_risk_hourly = go.Figure()
            # Determine color based on predicted cost (positive/negative)
            marker_colors = ['rgba(0,0,255,0.6)' if c > 0 else 'rgba(255,0,0,0.6)' for c in pred_cost_series.values] # Blue for positive cost (under-forecast driven), Red for negative cost (over-forecast driven)
            fig_risk_hourly.add_trace(go.Bar(
                x=pred_cost_series.index,
                y=pred_cost_series.values,
                name='Predicted Hourly Cost',
                marker_color=marker_colors
            ))
            fig_risk_hourly.update_layout(
                title=f"Predicted Hourly Cost of Error - Next {forecast_window_hours} Hours",
                xaxis_title="Time",
                yaxis_title="Predicted Cost ($)",
                hovermode='x unified',
                height=350,
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_risk_hourly, use_container_width=True)
        except Exception as plot_err:
             logging.error(f"Could not generate predicted cost bar chart: {plot_err}")
             st.warning("Could not display predicted cost breakdown chart.")

    elif pd.isna(phi) or not (-1 <= phi <= 1):
         st.warning("Cannot calculate predicted costs: Lag-1 ACF (phi) is invalid or could not be calculated.")
    elif forecast_start_time is pd.NaT:
         st.warning("Cannot calculate predicted costs: Could not determine forecast time window.")
    else:
         st.info("Predicted cost data is not available (check calculation logs). This might occur if price data is missing.")


# --- Download Button ---
st.sidebar.subheader("Download Data")
@st.cache_data # Cache the CSV conversion result
def convert_df_to_csv(df):
   """Converts the analysis DataFrame to a downloadable CSV string."""
   # Select and potentially rename columns for clarity in download
   cols_to_download = {
       ACTUAL_LOAD_COL: 'Actual_Load_MW',
       FORECAST_LOAD_COL: 'Forecast_Load_MW',
       FORECAST_ERROR_COL: 'Forecast_Error_MW',
       PRICE_DIFF_COL: 'Price_Diff_RTminusDA_USDperMWh',
       HOURLY_COST_COL: 'Hourly_Cost_of_Error_USD',
       APE_COL: 'Abs_Percent_Error_perc',
       HOUR_COL: 'Hour_of_Day',
       DOW_COL: 'Day_of_Week' # 0=Mon, 6=Sun
   }
   # Include only columns that actually exist in df_analysis
   valid_cols_dict = {k: v for k, v in cols_to_download.items() if k in df.columns}
   if not valid_cols_dict:
       logging.warning("No valid columns found for CSV download.")
       return None # Indicate failure

   df_download = df[list(valid_cols_dict.keys())].copy()
   df_download.rename(columns=valid_cols_dict, inplace=True)

   # Add Day of Week Names for convenience
   if 'Day_of_Week' in df_download.columns:
        day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        # Insert Day_Name after Day_of_Week if possible
        try:
            dow_index = df_download.columns.get_loc('Day_of_Week')
            df_download.insert(dow_index + 1, 'Day_Name', df_download['Day_of_Week'].map(day_map))
        except KeyError: # Fallback if column name changed
            df_download['Day_Name'] = df_download['Day_of_Week'].map(day_map)


   # Format index (timestamp) for readability in CSV
   df_download.index = df_download.index.strftime('%Y-%m-%d %H:%M:%S %Z')
   df_download.index.name = "Timestamp"

   try:
        return df_download.to_csv(index=True).encode('utf-8')
   except Exception as csv_err:
        logging.error(f"Error converting DataFrame to CSV: {csv_err}")
        return None

if not df_analysis.empty:
    csv_data = convert_df_to_csv(df_analysis)
    if csv_data:
        st.sidebar.download_button(
           label="Download Filtered Data (CSV)",
           data=csv_data,
           file_name=f"ercot_forecast_analysis_{start_date}_to_{end_date}.csv",
           mime="text/csv",
           help="Downloads the hourly data (incl. errors, costs, time features) for the selected date range."
        )
    else:
        st.sidebar.warning("Could not generate CSV for download.")
else:
    st.sidebar.info("No data available to download.")


st.markdown("---", unsafe_allow_html=True) # Use markdown for horizontal rule


# --- Section 2: Timeline: "Error -> Cost" (Plot 1 - Always Visible) ---
st.subheader("Timeline: Forecast Error & Price Spreads")
st.markdown(f"""
Links forecast misses (`{FORECAST_ERROR_COL}`, bars) to price differences (`{PRICE_DIFF_COL}`, line).
Blue bars = Under-forecast (Actual > Forecast, Error > 0), Red = Over-forecast (Actual < Forecast, Error < 0). Weekend shading.
""")
required_timeline_cols = [FORECAST_ERROR_COL, PRICE_DIFF_COL]
if all(col in df_analysis.columns for col in required_timeline_cols) and df_analysis[PRICE_DIFF_COL].notna().any():
    try:
        fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])

        # Dynamic range based on quantiles, but capped for readability
        error_q_low, error_q_high = df_analysis[FORECAST_ERROR_COL].quantile([0.01, 0.99])
        error_abs_max = max(abs(error_q_low), abs(error_q_high)) if pd.notna(error_q_low) and pd.notna(error_q_high) else df_analysis[FORECAST_ERROR_COL].abs().max()
        # Ensure some minimum range, avoid overly large range if max is huge outlier
        error_range_max = min(error_abs_max * 1.1, df_analysis[FORECAST_ERROR_COL].abs().quantile(0.999) * 1.5) if pd.notna(error_abs_max) else 500 # Cap based on 99.9th percentile
        error_range = [-error_range_max, error_range_max] if error_range_max > 0 else None

        price_q_low, price_q_high = df_analysis[PRICE_DIFF_COL].quantile([0.01, 0.99])
        price_abs_max = max(abs(price_q_low), abs(price_q_high)) if pd.notna(price_q_low) and pd.notna(price_q_high) else df_analysis[PRICE_DIFF_COL].abs().max()
        price_range_cap = 300 # Cap price axis slightly higher
        price_final_max = min(max(price_abs_max * 1.1, 50), price_range_cap) if pd.notna(price_abs_max) else price_range_cap # Ensure some minimum range, cap max
        price_range = [-price_final_max, price_final_max]

        # Split errors for coloring: Positive error = Under-forecast (Blue), Negative error = Over-forecast (Red)
        error_underforecast = df_analysis[FORECAST_ERROR_COL].clip(lower=0) # Positive values (Actual > Forecast)
        error_overforecast = df_analysis[FORECAST_ERROR_COL].clip(upper=0)  # Negative values (Actual < Forecast)

        # Use Bar traces for the errors
        fig_timeline.add_trace(go.Bar(x=df_analysis.index, y=error_underforecast, name='Under-forecast Error (Actual > Fcst)',
                                    marker_color='rgba(0,0,255,0.6)', hoverinfo='skip'), secondary_y=False) # Blue, slightly transparent
        fig_timeline.add_trace(go.Bar(x=df_analysis.index, y=error_overforecast, name='Over-forecast Error (Actual < Fcst)',
                                    marker_color='rgba(255,0,0,0.6)', hoverinfo='skip'), secondary_y=False) # Red, slightly transparent

        # Use Scattergl for the price line (potentially many points)
        fig_timeline.add_trace(go.Scattergl(x=df_analysis.index, y=df_analysis[PRICE_DIFF_COL], name='RT-DA Price Δ',
                                         mode='lines', line=dict(color='orange', width=2),
                                         connectgaps=True, # Connect gaps in price data
                                         hovertemplate='Price Δ: %{y:$.2f}<extra></extra>'
                                         ), secondary_y=True)

        # Add invisible trace for unified hover (shows error value)
        fig_timeline.add_trace(go.Scatter(
            x=df_analysis.index, y=df_analysis[FORECAST_ERROR_COL],
            mode='markers', marker=dict(opacity=0), showlegend=False,
            name="Error Details",
            hovertemplate='<b>%{x|%Y-%m-%d %H:%M %Z}</b><br>Error: %{y:,.0f} MW<extra></extra>'
            ), secondary_y=False)


        # Weekend Shading
        shapes = []
        if DOW_COL in df_analysis.columns:
            df_weekends = df_analysis[df_analysis[DOW_COL] >= 5] # Saturday=5, Sunday=6
            weekend_starts = df_weekends.index.normalize().unique()
            for weekend_start_date in weekend_starts:
                 start_ts = weekend_start_date # Start at 00:00 Saturday
                 end_ts = start_ts + pd.Timedelta(days=2) # End at 00:00 Monday (covers Sat/Sun)
                 # Ensure start/end are within plot bounds
                 plot_start, plot_end = df_analysis.index.min(), df_analysis.index.max()
                 if start_ts < plot_end and end_ts > plot_start:
                     shapes.append(dict(type="rect", xref="x", yref="paper",
                                   x0=max(start_ts, plot_start), y0=0,
                                   x1=min(end_ts, plot_end + pd.Timedelta(hours=1)), y1=1, # Extend slightly past last point if needed
                                   fillcolor="grey", opacity=0.15, layer="below", line_width=0))
        else:
            logging.warning("DayOfWeek column not found, cannot add weekend shading.")


        fig_timeline.update_layout(
            xaxis_title="Date / Time",
            yaxis_title=f"Forecast Error ({ERROR_UNITS})",
            yaxis2_title=f"RT-DA Price Δ ($/MWh)",
            height=450,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='relative', # Stack positive/negative parts (though they don't overlap)
            bargap=0, # No gap between bars for continuous feel
            yaxis=dict(range=error_range, zeroline=True, zerolinewidth=1, zerolinecolor='grey', title_font=dict(color="navy")),
            yaxis2=dict(range=price_range, zeroline=True, zerolinewidth=1, zerolinecolor='darkorange', overlaying='y', side='right', showgrid=False, title_font=dict(color="orange")),
            shapes=shapes, # Add weekend shading shapes
            margin=dict(l=50, r=50, t=40, b=40) # Adjusted margins slightly
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    except Exception as e:
        logging.error(f"Error generating Error/Price timeline plot: {e}", exc_info=True)
        st.error(f"Error generating Error/Price timeline plot: {e}")
        st.exception(e)
else:
    missing_reason = []
    if FORECAST_ERROR_COL not in df_analysis.columns: missing_reason.append(f"'{FORECAST_ERROR_COL}'")
    if PRICE_DIFF_COL not in df_analysis.columns: missing_reason.append(f"'{PRICE_DIFF_COL}'")
    elif PRICE_DIFF_COL in df_analysis.columns and not df_analysis[PRICE_DIFF_COL].notna().any():
        missing_reason.append(f"'{PRICE_DIFF_COL}' has no valid data")
    st.info(f"Cannot generate Error/Price timeline: Requires { ' and '.join(missing_reason) }.")

st.markdown("---", unsafe_allow_html=True)

# --- Section 3: Persistence Panel (Plot 2 - Always Visible) ---
st.subheader("Error Persistence & Price Correlation")

# Re-calculate ACF/CCF needed for plots (use cached versions)
acf_lags_plot, acf_values_plot, acf_confint_plot = np.array([]), np.array([]), np.array([])
if FORECAST_ERROR_COL in df_analysis and df_analysis[FORECAST_ERROR_COL].notna().any():
    error_series_clean = df_analysis[FORECAST_ERROR_COL].dropna()
    if len(error_series_clean) > 100: # Need sufficient points for 48 lags
        acf_lags_plot, acf_values_plot, acf_confint_plot = calculate_acf(error_series_clean, nlags=48)
    else:
        logging.warning("Not enough data to calculate ACF plot (need > 100 points).")


has_price_data = PRICE_DIFF_COL in df_analysis.columns and df_analysis[PRICE_DIFF_COL].notna().any()
ccf_lags, ccf_values, ccf_conf_level = np.array([]), np.array([]), np.nan
if has_price_data and FORECAST_ERROR_COL in df_analysis:
    # Ensure enough overlapping data
    df_ccf_temp = df_analysis[[FORECAST_ERROR_COL, PRICE_DIFF_COL]].dropna()
    if len(df_ccf_temp) > 100:
        ccf_lags, ccf_values, ccf_conf_level = calculate_ccf(
            df_ccf_temp[FORECAST_ERROR_COL], df_ccf_temp[PRICE_DIFF_COL], maxlags=48
        )
    else:
         logging.warning("Not enough overlapping data to calculate CCF plot (need > 100 points).")


# Persistence Alert Banner Logic (separate from Bias Alert now)
persistence_alert = False
alert_reason = []
if len(acf_values_plot) > 1 and acf_confint_plot is not None and acf_confint_plot.ndim == 2 and acf_confint_plot.shape[0] == len(acf_lags_plot):
    n_lags_available = len(acf_values_plot) - 1 # Number of non-zero lags calculated

    # Check Lag 1 (using data already calculated for phi alert)
    if n_lags_available >= 1 and lag1_outside_ci:
        alert_reason.append("lag-1")

    # Check Lag 24 if available
    if n_lags_available >= 24:
        lag24_acf_p = acf_values_plot[24]
        ci_lower_lag24_p = acf_confint_plot[24, 0]
        ci_upper_lag24_p = acf_confint_plot[24, 1]
        lag24_outside = pd.notna(lag24_acf_p) and pd.notna(ci_lower_lag24_p) and pd.notna(ci_upper_lag24_p) and \
                        (lag24_acf_p < ci_lower_lag24_p or lag24_acf_p > ci_upper_lag24_p)
        if lag24_outside: alert_reason.append("lag-24")

    # Check Lag 168 (weekly) if available
    if n_lags_available >= 168: # Check if ACF was calculated for enough lags
        try:
             lag168_acf_p = acf_values_plot[168]
             ci_lower_lag168_p = acf_confint_plot[168, 0]
             ci_upper_lag168_p = acf_confint_plot[168, 1]
             lag168_outside = pd.notna(lag168_acf_p) and pd.notna(ci_lower_lag168_p) and pd.notna(ci_upper_lag168_p) and \
                             (lag168_acf_p < ci_lower_lag168_p or lag168_acf_p > ci_upper_lag168_p)
             if lag168_outside: alert_reason.append("lag-168 (Weekly)")
        except IndexError:
             logging.warning("Lag 168 not available in calculated ACF for persistence check.")


    if alert_reason and not bias_alert_triggered: # Show persistence warning only if bias alert wasn't triggered
        persistence_alert = True
        st.warning(f"**Persistence Warning:** Forecast error autocorrelation at { ', '.join(alert_reason) } is statistically significant (outside 95% CI). This suggests predictable patterns might remain in the errors.")
        logging.warning(f"Persistence Warning triggered for lags: {alert_reason}")


col_persist1, col_persist2 = st.columns(2)

with col_persist1:
    st.markdown("**Forecast Error Autocorrelation (ACF)**")
    st.caption("Shows if errors are correlated with past errors (lags 1-48h). Bars outside dashed lines suggest significant correlation (95% CI).")
    if len(acf_lags_plot) > 1:
        try:
            fig_acf = go.Figure()
            # Determine CI bounds for plotting (use the ones from calculate_acf)
            ci_lower, ci_upper = np.nan, np.nan
            if acf_confint_plot is not None and acf_confint_plot.ndim == 2 and acf_confint_plot.shape[0] == len(acf_lags_plot) and len(acf_lags_plot)>1:
                 # Use the bounds directly from the calculation (lag 1 onwards)
                 ci_lower_vals = acf_confint_plot[1:, 0]
                 ci_upper_vals = acf_confint_plot[1:, 1]
                 # For plotting lines, typically use the approximate Bartlett formula bounds if CI array isn't constant
                 # Let's use the first lag's CI as indicative dashed lines
                 ci_lower = acf_confint_plot[1, 0]
                 ci_upper = acf_confint_plot[1, 1]

            # Plot bars for lags 1 to 48
            fig_acf.add_trace(go.Bar(x=acf_lags_plot[1:], y=acf_values_plot[1:], name='ACF', marker_color='steelblue'))
            # Add confidence interval lines
            if pd.notna(ci_upper):
                fig_acf.add_hline(y=ci_upper, line=dict(color='rgba(255,0,0,0.5)', dash='dash'), name='95% CI Upper')
            if pd.notna(ci_lower):
                fig_acf.add_hline(y=ci_lower, line=dict(color='rgba(255,0,0,0.5)', dash='dash'), name='95% CI Lower')

            fig_acf.add_hline(y=0, line=dict(color='grey', width=1))
            fig_acf.update_layout(xaxis_title="Lag (Hours)", yaxis_title="Autocorrelation",
                                  yaxis_range=[-1, 1], height=350, margin=dict(l=40, r=20, t=30, b=40),
                                  showlegend=False) # Legend cluttered, use caption
            st.plotly_chart(fig_acf, use_container_width=True)
        except Exception as e:
            logging.error(f"Error plotting ACF: {e}", exc_info=True)
            st.error(f"Error plotting ACF: {e}")
    else:
        st.info(f"Not enough data or '{FORECAST_ERROR_COL}' missing/invalid for ACF plot.")

with col_persist2:
    st.markdown("**Error vs. Price Cross-Correlation (CCF)**")
    st.caption("Corr(Error(t), PriceΔ(t+lag)). Bars outside dashed lines = significant correlation (95% CI). Helps see if errors lead/lag price moves.")
    if has_price_data:
        if len(ccf_lags) > 0:
            try:
                fig_ccf = go.Figure()
                # Color bars based on significance
                colors = ['red' if pd.notna(ccf_conf_level) and abs(val) > ccf_conf_level else 'steelblue' for val in ccf_values]
                fig_ccf.add_trace(go.Bar(x=ccf_lags, y=ccf_values, name='CCF', marker_color=colors))
                if pd.notna(ccf_conf_level):
                    fig_ccf.add_hline(y=ccf_conf_level, line=dict(color='rgba(255,0,0,0.5)', dash='dash'), name='95% CI')
                    fig_ccf.add_hline(y=-ccf_conf_level, line=dict(color='rgba(255,0,0,0.5)', dash='dash'), showlegend=False)
                fig_ccf.add_hline(y=0, line=dict(color='grey', width=1))
                fig_ccf.add_vline(x=0, line=dict(color='grey', width=1, dash='dot')) # Mark lag 0
                fig_ccf.update_layout(xaxis_title="Lag (Hours) [Error(t) vs PriceΔ(t+lag)]", yaxis_title="Correlation",
                                      yaxis_range=[-1, 1], height=350, margin=dict(l=40, r=20, t=30, b=40), showlegend=False)
                st.plotly_chart(fig_ccf, use_container_width=True)
            except Exception as e:
                logging.error(f"Error plotting CCF: {e}", exc_info=True)
                st.error(f"Error plotting CCF: {e}")
        else:
            st.info("Not enough overlapping data for Error and Price Difference to calculate CCF.")
    else:
        st.info(f"Price difference column '{PRICE_DIFF_COL}' not available for CCF.")

st.markdown("---", unsafe_allow_html=True)































# --- Plots Hidden by Default with st.toggle ---

# --- Section: Hour-of-day RT-DA Price heatmap ---
show_hourly_heatmap_rtda = st.toggle("Show Hourly RT-DA Price Difference Heatmap", value=False, key="toggle_heatmap_price")
if show_hourly_heatmap_rtda:
    st.subheader("Average Price Difference by Hour and Day of Week")
    st.markdown(f"""
    Shows the typical RT–DA Price Difference (`{PRICE_DIFF_COL}`) for each hour across the days of the week.
    Helps identify systematic time-based pricing patterns. Red = higher RT price, Blue = lower RT price vs DA.
    """)
    required_cols_plot_price_heatmap = [PRICE_DIFF_COL, HOUR_COL, DOW_COL]
    if all(col in df_analysis.columns for col in required_cols_plot_price_heatmap) and df_analysis[PRICE_DIFF_COL].notna().any():
        try:
            hourly_pivot = df_analysis.pivot_table(
                index=HOUR_COL, columns=DOW_COL, values=PRICE_DIFF_COL, aggfunc='mean'
            )
            hourly_pivot = hourly_pivot.reindex(index=range(24), columns=range(7)) # Ensure all cells exist
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            # Determine symmetric color range around 0 based on max absolute value in pivot
            max_abs_val = hourly_pivot.abs().max().max()
            # Add small buffer, ensure range is not zero
            color_limit = max(abs(max_abs_val) * 1.05, 1) if pd.notna(max_abs_val) else 10
            zmin, zmax = -color_limit, color_limit

            fig_hour_heatmap = go.Figure(data=go.Heatmap(
                   z=hourly_pivot.values, x=day_names, y=hourly_pivot.index,
                   colorscale='RdBu', # Red-Blue diverging scale centered at 0
                   zmid=0, zmin=zmin, zmax=zmax, # Use symmetric range
                   hoverongaps = False,
                   hovertemplate = "<b>Day:</b> %{x}<br><b>Hour:</b> %{y}:00<br><b>Avg PriceΔ (RT-DA):</b> %{z:.2f} $/MWh<extra></extra>"
                   ))
            fig_hour_heatmap.update_layout(
                xaxis_title='Day of Week', yaxis_title='Hour of Day (Start)',
                yaxis=dict(tickmode='array', tickvals=list(range(0, 24, 2)), autorange='reversed'), # Show every 2nd hour label, 0 at top
                height=550, margin=dict(l=40, r=40, t=40, b=40),
                coloraxis_colorbar_title_text='Avg Price Δ ($/MWh)' # Add title to colorbar
            )
            st.plotly_chart(fig_hour_heatmap, use_container_width=True)

        except Exception as e:
            logging.error(f"Error generating hourly RT-DA price difference heatmap: {e}", exc_info=True)
            st.error(f"Error generating hourly RT-DA price difference heatmap: {e}")
            st.exception(e)
    else:
        missing_info = [col for col in required_cols_plot_price_heatmap if col not in df_analysis.columns]
        if PRICE_DIFF_COL in df_analysis.columns and not df_analysis[PRICE_DIFF_COL].notna().any():
            missing_info.append(f"'{PRICE_DIFF_COL}' has no data")
        st.info(f"Cannot generate hourly RT-DA price heatmap. Requires valid data for { ', '.join(required_cols_plot_price_heatmap) }. Missing/Invalid: {', '.join(missing_info)}")
    st.markdown("---", unsafe_allow_html=True)

# --- Section: Hour-of-day Forecast Error heatmap ---
show_hourly_error_heatmap = st.toggle("Show Hourly Forecast Error Heatmap", value=False, key="toggle_heatmap_error")
if show_hourly_error_heatmap:
    st.subheader("Average Forecast Error by Hour and Day of Week")
    st.markdown(f"""
    Shows the typical Forecast Error (`{FORECAST_ERROR_COL}`) for each hour across the days of the week.
    Helps identify systematic time-based forecast biases or patterns. Red = Over-forecast (Error < 0), Blue = Under-forecast (Error > 0).
    """)
    required_cols_plot_error_heatmap = [FORECAST_ERROR_COL, HOUR_COL, DOW_COL]
    if all(col in df_analysis.columns for col in required_cols_plot_error_heatmap) and df_analysis[FORECAST_ERROR_COL].notna().any():
        try:
            hourly_pivot_error = df_analysis.pivot_table(
                index=HOUR_COL, columns=DOW_COL, values=FORECAST_ERROR_COL, aggfunc='mean'
            )
            hourly_pivot_error = hourly_pivot_error.reindex(index=range(24), columns=range(7))
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            # Determine symmetric color range around 0
            max_abs_val_err = hourly_pivot_error.abs().max().max()
            color_limit_err = max(abs(max_abs_val_err) * 1.05, 1) if pd.notna(max_abs_val_err) else 10
            zmin_err, zmax_err = -color_limit_err, color_limit_err

            fig_hour_heatmap_err = go.Figure(data=go.Heatmap(
                   z=hourly_pivot_error.values, x=day_names, y=hourly_pivot_error.index,
                   colorscale='RdBu', # Red(-ve error/OverFC) to Blue(+ve error/UnderFC)
                   reversescale=True, # Reverse RdBu so Red = Negative Error (Over-forecast)
                   zmid=0, zmin=zmin_err, zmax=zmax_err,
                   hoverongaps=False,
                   hovertemplate=f"<b>Day:</b> %{{x}}<br><b>Hour:</b> %{{y}}:00<br><b>Avg Forecast Error:</b> %{{z:.1f}} {ERROR_UNITS}<extra></extra>"
                   ))
            fig_hour_heatmap_err.update_layout(
                xaxis_title='Day of Week', yaxis_title='Hour of Day (Start)',
                yaxis=dict(tickmode='array', tickvals=list(range(0, 24, 2)), autorange='reversed'),
                height=550, margin=dict(l=40, r=40, t=40, b=40),
                coloraxis_colorbar_title_text=f'Avg Error ({ERROR_UNITS})'
            )
            st.plotly_chart(fig_hour_heatmap_err, use_container_width=True)
        except Exception as e:
            logging.error(f"Error generating hourly forecast error heatmap: {e}", exc_info=True)
            st.error(f"Error generating hourly forecast error heatmap: {e}")
            st.exception(e)
    else:
        missing_info = [col for col in required_cols_plot_error_heatmap if col not in df_analysis.columns]
        if FORECAST_ERROR_COL in df_analysis.columns and not df_analysis[FORECAST_ERROR_COL].notna().any():
            missing_info.append(f"'{FORECAST_ERROR_COL}' has no data")
        st.info(f"Cannot generate hourly forecast error heatmap. Requires valid data for { ', '.join(required_cols_plot_error_heatmap) }. Missing/Invalid: {', '.join(missing_info)}")
    st.markdown("---", unsafe_allow_html=True)


# --- Section: Hourly Pattern Plots ---
show_hourly_patterns = st.toggle("Show Hourly Error Aggregation Plots", value=False, key="toggle_hourly_patterns")
if show_hourly_patterns:
    st.subheader("Hourly Patterns Analysis")
    st.markdown(f"""
    Do forecast errors (`{FORECAST_ERROR_COL}`) exhibit specific patterns depending on the hour of the day? These plots aggregate error data by hour.
    *   **Top Plot (Average Magnitude):** Shows the average size of over-forecast errors (Red, Error < 0) and under-forecast errors (Blue, Error > 0) for each hour. This helps identify hours where the *magnitude* of misses is typically larger.
    *   **Bottom Plot (Frequency Count):** Shows the total number of times over-forecasting (Red) and under-forecasting (Blue) occurred for each hour. This identifies hours where the forecast is more *likely* to be wrong in a particular direction.
    *   **Day Type Filter:** Use the radio buttons to focus the analysis on Weekdays, Weekends, or All Days, as patterns can differ significantly.
    **Why Monitor?** Identifying hourly biases or weaknesses provides actionable insights for tuning the forecast model's diurnal profile adjustments.
    """)

    try:
        required_cols_hourly = [FORECAST_ERROR_COL, HOUR_COL, DOW_COL]
        missing_cols_hourly = [col for col in required_cols_hourly if col not in df_analysis.columns]
        has_data_hourly = False
        if not missing_cols_hourly:
            if FORECAST_ERROR_COL in df_analysis.columns and df_analysis[FORECAST_ERROR_COL].notna().any():
                 has_data_hourly = True

        if not df_analysis.empty and has_data_hourly:

            day_filter = st.radio(
                "Select Day Type for Hourly Aggregations:",
                options=["All Days", "Weekdays", "Weekends"],
                index=0, horizontal=True, key="hourly_day_filter"
            )

            df_hourly_filtered = pd.DataFrame() # Initialize
            filter_desc = ""
            # Use DOW_COL for filtering
            if day_filter == "Weekdays":
                df_hourly_filtered = df_analysis[df_analysis[DOW_COL] < 5].copy()
                filter_desc = " (Weekdays Only)"
            elif day_filter == "Weekends":
                df_hourly_filtered = df_analysis[df_analysis[DOW_COL] >= 5].copy()
                filter_desc = " (Weekends Only)"
            else: # All Days
                df_hourly_filtered = df_analysis.copy()
                filter_desc = " (All Days)"

            if df_hourly_filtered.empty or df_hourly_filtered[FORECAST_ERROR_COL].isnull().all():
                st.warning(f"No valid forecast error data available for the selected day type '{day_filter}'. Hourly plots skipped.")
            else:
                # Proceed with plots using df_hourly_filtered

                # --- Plot 1: Avg Error Magnitude by Hour ---
                # Calculate positive errors (under-forecast)
                df_hourly_filtered['Underforecast_Error'] = df_hourly_filtered[FORECAST_ERROR_COL].clip(lower=0)
                # Calculate absolute value of negative errors (over-forecast) for magnitude comparison
                df_hourly_filtered['Overforecast_Error_abs'] = df_hourly_filtered[FORECAST_ERROR_COL].clip(upper=0).abs()

                grouped_magnitude = df_hourly_filtered.groupby(HOUR_COL).agg(
                    Avg_Underforecast=('Underforecast_Error', 'mean'),
                    Avg_Overforecast_abs=('Overforecast_Error_abs', 'mean')
                ).reset_index()

                # Ensure all hours 0-23 are present, filling missing with 0
                all_hours = pd.DataFrame({HOUR_COL: range(24)})
                grouped_magnitude = pd.merge(all_hours, grouped_magnitude, on=HOUR_COL, how='left').fillna(0)

                fig_hourly_mag = go.Figure()
                # Blue bars for Under-forecast magnitude
                fig_hourly_mag.add_trace(go.Bar(
                    x=grouped_magnitude[HOUR_COL], y=grouped_magnitude['Avg_Underforecast'],
                    name=f'Avg. Under-forecast Magnitude ({ERROR_UNITS})', marker_color='rgba(0,0,255,0.7)' # Blue
                ))
                # Red bars for Over-forecast magnitude
                fig_hourly_mag.add_trace(go.Bar(
                    x=grouped_magnitude[HOUR_COL], y=grouped_magnitude['Avg_Overforecast_abs'],
                    name=f'Avg. Over-forecast Magnitude ({ERROR_UNITS})', marker_color='rgba(255,0,0,0.7)' # Red
                ))
                fig_hourly_mag.update_layout(
                    title=f"Average Forecast Error Magnitude by Hour{filter_desc}",
                    xaxis_title="Hour of Day (0-23)",
                    yaxis_title=f"Average Error Magnitude ({ERROR_UNITS})",
                    barmode='group', height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(tickmode='linear', dtick=1) # Ensure all hours are labeled
                )
                st.plotly_chart(fig_hourly_mag, use_container_width=True)

                # --- Plot 2: Count of Errors by Hour ---
                tolerance = 1e-6 # Tolerance for zero comparison
                df_hourly_filtered['Is_Underforecast'] = (df_hourly_filtered[FORECAST_ERROR_COL] > tolerance).astype(int) # Error > 0
                df_hourly_filtered['Is_Overforecast'] = (df_hourly_filtered[FORECAST_ERROR_COL] < -tolerance).astype(int) # Error < 0

                grouped_count = df_hourly_filtered.groupby(HOUR_COL).agg(
                    Count_Underforecast=('Is_Underforecast', 'sum'),
                    Count_Overforecast=('Is_Overforecast', 'sum')
                ).reset_index()

                grouped_count = pd.merge(all_hours, grouped_count, on=HOUR_COL, how='left').fillna(0)

                fig_hourly_count = go.Figure()
                # Blue bars for Under-forecast count
                fig_hourly_count.add_trace(go.Bar(
                    x=grouped_count[HOUR_COL], y=grouped_count['Count_Underforecast'],
                    name='Under-forecast Count', marker_color='rgba(0,0,255,0.7)' # Blue
                ))
                # Red bars for Over-forecast count
                fig_hourly_count.add_trace(go.Bar(
                    x=grouped_count[HOUR_COL], y=grouped_count['Count_Overforecast'],
                    name='Over-forecast Count', marker_color='rgba(255,0,0,0.7)' # Red
                ))
                fig_hourly_count.update_layout(
                    title=f"Count of Over/Under Forecast Occurrences by Hour{filter_desc}",
                    xaxis_title="Hour of Day (0-23)", yaxis_title="Number of Occurrences",
                    barmode='group', height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(tickmode='linear', dtick=1) # Ensure all hours are labeled
                )
                st.plotly_chart(fig_hourly_count, use_container_width=True)

        elif df_analysis.empty:
            st.info("Dataframe is empty. Cannot generate hourly pattern plots.")
        else: # Dataframe not empty, but required columns/data missing
            st.warning(f"Skipping Hourly Pattern plots: Ensure the dataframe contains valid data in required columns: {', '.join(required_cols_hourly)}.")
            if missing_cols_hourly:
                 st.info(f"Missing columns: {', '.join(missing_cols_hourly)}")
            elif not has_data_hourly:
                 st.info(f"The column '{FORECAST_ERROR_COL}' contains no valid (non-missing) data.")

    except Exception as e:
        st.error(f"An error occurred while generating hourly pattern plots: {e}")
        logging.error(f"Error in hourly pattern plots: {e}", exc_info=True)
        st.exception(e) # Provides traceback in logs/console for debugging
    st.markdown("---", unsafe_allow_html=True)


# --- Advanced Diagnostics Expander ---
with st.expander("Advanced Diagnostics & Original Plots"):
    st.markdown("Detailed plots for deeper analysis. These were part of the original dashboard or provide additional context.")

    # --- Plot: Load vs Forecast and Forecast Error (Using Scattergl) ---
    st.subheader("Load vs. Forecast & Error Time Series (Detailed)")
    st.markdown(f"Shows actual load (`{ACTUAL_LOAD_COL}`) vs. forecast (`{FORECAST_LOAD_COL}`), and the resulting error (`{FORECAST_ERROR_COL}`). Uses WebGL for potentially better performance.")
    try:
        plot1_cols_adv = [ACTUAL_LOAD_COL, FORECAST_LOAD_COL, FORECAST_ERROR_COL]
        if all(col in df_analysis.columns for col in plot1_cols_adv) and df_analysis[plot1_cols_adv].notna().any(axis=None): # Check if *any* value is not NaN
            fig1_adv = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                                     row_heights=[0.6, 0.4], subplot_titles=("Load vs. Forecast", "Forecast Error"))

            # Load vs Forecast (Top Plot) - Use Scattergl
            fig1_adv.add_trace(go.Scattergl(x=df_analysis.index, y=df_analysis[ACTUAL_LOAD_COL], name='Actual Load',
                                          mode='lines', line=dict(color='rgba(0,100,80,1)', width=1.5), connectgaps=True), row=1, col=1) # Removed fill for performance
            fig1_adv.add_trace(go.Scattergl(x=df_analysis.index, y=df_analysis[FORECAST_LOAD_COL], name='Forecast Load',
                                          mode='lines', line=dict(color='rgba(0,0,255,0.8)', width=1), connectgaps=True), row=1, col=1)
            fig1_adv.update_yaxes(title_text=f"Load ({ERROR_UNITS})", row=1, col=1)

            # Forecast Error (Bottom Plot) - Use Scattergl
            # Split for coloring, Plot lines first for legend/hover, then add fills below
            error_underforecast_adv = df_analysis[FORECAST_ERROR_COL].clip(lower=0) # Positive Error = Under-forecast
            error_overforecast_adv = df_analysis[FORECAST_ERROR_COL].clip(upper=0)  # Negative Error = Over-forecast

            # Lines (on top)
            fig1_adv.add_trace(go.Scattergl(x=df_analysis.index, y=error_underforecast_adv.replace(0, np.nan), name='Under-forecast Error',
                                          mode='lines', line=dict(color='rgba(0,0,255,0.7)', width=1), connectgaps=False), row=2, col=1)
            fig1_adv.add_trace(go.Scattergl(x=df_analysis.index, y=error_overforecast_adv.replace(0, np.nan), name='Over-forecast Error',
                                          mode='lines', line=dict(color='rgba(255,0,0,0.7)', width=1), connectgaps=False), row=2, col=1)
            # Fills (below lines)
            fig1_adv.add_trace(go.Scattergl(x=df_analysis.index, y=error_underforecast_adv, showlegend=False,
                                          mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.3)', # Blue fill for Under
                                          connectgaps=False), row=2, col=1)
            fig1_adv.add_trace(go.Scattergl(x=df_analysis.index, y=error_overforecast_adv, showlegend=False,
                                          mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)', # Red fill for Over
                                          connectgaps=False), row=2, col=1)
            fig1_adv.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=2, col=1)

            # Add Rolling Average Error (optional, can be slow for large datasets)
            # window_7d = 24 * 7
            # if len(df_analysis[FORECAST_ERROR_COL].dropna()) >= window_7d / 2: # Require at least half a window
            #     error_ma_7d = df_analysis[FORECAST_ERROR_COL].rolling(window=window_7d, min_periods=24, center=True).mean() # Center rolling window
            #     fig1_adv.add_trace(go.Scattergl(x=df_analysis.index, y=error_ma_7d, name='7-Day Avg Error',
            #                                   mode='lines', line=dict(color='rgba(80,80,80,0.9)', width=1.5, dash='dot'), connectgaps=True), row=2, col=1)

            fig1_adv.update_yaxes(title_text=f"Error ({ERROR_UNITS})", row=2, col=1)
            fig1_adv.update_xaxes(title_text="Date / Time", row=2, col=1)
            fig1_adv.update_layout(height=600, hovermode='x unified',
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                   margin=dict(l=40, r=40, t=60, b=40)) # Add top margin for subplot titles
            st.plotly_chart(fig1_adv, use_container_width=True)
        else:
            st.info("Required columns for detailed Load/Forecast plot not available or contain only NaNs.")
    except Exception as e:
        logging.error(f"Error generating detailed Load/Forecast plot: {e}", exc_info=True)
        st.error(f"Error generating detailed Load/Forecast plot: {e}")
        st.exception(e)

    # --- Plot: Cumulative Forecast Bias ---
    st.subheader("Cumulative Forecast Bias Over Time")
    st.markdown(f"Shows the running total of forecast error (`{FORECAST_ERROR_COL}`), highlighting persistent bias trends.")
    try:
        if FORECAST_ERROR_COL in df_analysis.columns and df_analysis[FORECAST_ERROR_COL].notna().any():
            cumulative_error = df_analysis[FORECAST_ERROR_COL].fillna(0).cumsum()
            fig_cumul = go.Figure()
            fig_cumul.add_trace(go.Scattergl(x=df_analysis.index, y=cumulative_error, mode='lines', # Use Scattergl
                                           name='Cumulative Error', line=dict(color='purple', width=2), connectgaps=True))
            fig_cumul.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
            fig_cumul.update_layout(
                 xaxis_title="Date / Time", yaxis_title=f"Cumulative Error ({ERROR_UNITS}h)", height=400, hovermode='x unified',
                 margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_cumul, use_container_width=True)
        else:
            st.info(f"Cumulative bias plot requires '{FORECAST_ERROR_COL}'.")
    except Exception as e:
        logging.error(f"Error generating cumulative error plot: {e}", exc_info=True)
        st.error(f"Error generating cumulative error plot: {e}")
        st.exception(e)

    # --- Plot: Error vs. Actual Load Scatter ---
    st.subheader("Forecast Error vs. Actual Load Level")
    st.markdown("Shows if forecast accuracy changes depending on the system load level. Uses WebGL.")
    try:
        scatter_cols_adv = [FORECAST_ERROR_COL, ACTUAL_LOAD_COL]
        if all(col in df_analysis.columns for col in scatter_cols_adv) and df_analysis[scatter_cols_adv].notna().all(axis=1).any():
            # Use only non-NaN pairs for scatter
            df_scatter_adv = df_analysis[scatter_cols_adv].dropna().copy()
            if not df_scatter_adv.empty:
                fig_scatter_adv = go.Figure()
                fig_scatter_adv.add_trace(go.Scattergl( # Use Scattergl for potentially many points
                    x=df_scatter_adv[ACTUAL_LOAD_COL], y=df_scatter_adv[FORECAST_ERROR_COL], mode='markers', name='Hourly Error',
                    marker=dict(color='rgba(0, 128, 128, 0.5)', size=5), # Teal color, slightly larger markers
                    customdata=df_scatter_adv.index.strftime('%Y-%m-%d %H:%M %Z'), # Format time for hover
                    hovertemplate=(f"<b>Time:</b> %{{customdata}}<br>"
                                   f"<b>Actual Load:</b> %{{x:,.0f}} {ERROR_UNITS}<br>"
                                   f"<b>Forecast Error:</b> %{{y:,.0f}} {ERROR_UNITS}<extra></extra>")
                ))
                fig_scatter_adv.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
                fig_scatter_adv.update_layout(
                    xaxis_title=f"Actual Load ({ERROR_UNITS})", yaxis_title=f"Forecast Error ({ERROR_UNITS})",
                    height=450, hovermode='closest', margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_scatter_adv, use_container_width=True)
            else:
                 st.info(f"No valid data points (where both '{FORECAST_ERROR_COL}' and '{ACTUAL_LOAD_COL}' are non-missing) available for scatter plot.")
        else:
             st.info(f"Error vs Load scatter requires '{FORECAST_ERROR_COL}' and '{ACTUAL_LOAD_COL}' with valid data.")
    except Exception as e:
        logging.error(f"Error generating error vs. load scatter plot: {e}", exc_info=True)
        st.error(f"Error generating error vs. load scatter plot: {e}")
        st.exception(e)


logging.info("Streamlit script execution finished.")