# import sys # Keep for potential future use, though not strictly necessary now
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
# from collections import defaultdict # No longer needed directly
# import scipy.stats # No longer explicitly required by this version's plots
import statsmodels.api as sm
import statsmodels.tsa.stattools as tsat
# from statsmodels.graphics.tsaplots import plot_acf # Using Plotly implementation instead for consistency

# --- Module Imports with Error Handling ---
# (Keep existing error handling for custom modules)
try:
    from functions import load_config
except ImportError as e:
    st.error(f"Error importing 'functions.py': {e}")
    st.error("Please ensure 'functions.py' is in the same directory and contains 'load_config'.")
    st.stop()

try:
    from iso_data_integration2 import load_all_iso_data, ensure_uniform_hourly_index, add_price_data_to_existing_df
except ImportError as e:
    st.error(f"Error importing from 'iso_data_integration2.py': {e}")
    st.error("Please ensure 'iso_data_integration2.py' is in the same directory and contains the necessary functions.")
    st.stop()

# Remove metrics_calculation import - metrics will be calculated directly for simplicity here
# try:
#     from metrics_calculation import compute_iso_metrics
# except ImportError as e:
#     st.error(f"Error importing from 'metrics_calculation.py': {e}")
#     st.error("Please ensure 'metrics_calculation.py' is in the same directory and contains 'compute_iso_metrics'.")
#     st.stop()

# --- Configuration ---
try:
    ISO_CONFIG = load_config()
    if ISO_CONFIG is None:
        raise ValueError("load_config returned None")
except Exception as e:
    st.error(f"Failed to load configuration from config.json: {e}. Cannot proceed.")
    st.stop()

TARGET_ISO_KEY = "ERCOT_Load_From_ISO"

# --- Define Standard Column Names (CRITICAL for consistency) ---
# Ensure these match EXACTLY the columns in your processed ERCOT DataFrame
ACTUAL_LOAD_COL = 'TOTAL Actual Load (MW)'
FORECAST_LOAD_COL = 'SystemTotal Forecast Load (MW)'
# Define derived columns we will create
FORECAST_ERROR_COL = 'Forecast Error (MW)' # Actual - Forecast
PRICE_DIFF_COL = "LMP Difference (USD)" # e.g., RT LMP - DA LMP
HOURLY_COST_COL = 'Hourly Cost of Error ($)'
APE_COL = 'Absolute Percentage Error (%)'
HOUR_COL = 'Hour'
DOW_COL = 'DayOfWeek' # Monday=0, Sunday=6

REQUIRED_INPUT_COLS = [ACTUAL_LOAD_COL, FORECAST_LOAD_COL] # Base columns needed before derivations

# --- Data Loading (Cached) ---
@st.cache_data(ttl=24 * 60 * 60)
def load_all_data_cached():
    """Loads data for all ISOs defined in config, returns a dict of DataFrames."""
    all_data = {} # Initialize
    try:
        # Attempt calling with config, fallback to parameterless call if TypeError occurs
        try:
            all_data = load_all_iso_data(ISO_CONFIG)
        except TypeError:
            st.warning("Attempting to call `load_all_iso_data` without arguments due to TypeError.")
            all_data = load_all_iso_data()
        return all_data
    except FileNotFoundError as fnf_err:
        st.error(f"Error: Required file not found during loading: {fnf_err}")
        return {}
    except Exception as e:
        st.error("An unexpected error occurred during data loading:")
        st.exception(e)
        return {}

# --- Helper: Get Global Date Range ---
# (Keep existing get_global_date_range function as it handles multiple ISOs)
def get_global_date_range(iso_data_dict):
    """Calculates the minimum start and maximum end date across all loaded DataFrames."""
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
                         # st.warning(f"Could not convert index to datetime for ISO '{key}'. Skipping date range calculation for this ISO.")
                         continue
                 except Exception as conv_err:
                    #  st.warning(f"Error converting index for ISO '{key}': {conv_err}. Skipping.")
                     continue

            if isinstance(df_index, pd.DatetimeIndex) and pd.api.types.is_datetime64_any_dtype(df_index) and not df_index.empty:
                current_index = df_index
                if current_index.tz is None:
                    tz_info_str = ISO_CONFIG.get(key, {}).get('timezone', 'UTC')
                    try:
                        current_index = current_index.tz_localize(tz_info_str)
                    except Exception:
                        try:
                           current_index = current_index.tz_localize('UTC')
                        except Exception:
                           continue
                if not current_index.empty:
                    valid_dates.append(current_index.min())
                    valid_dates.append(current_index.max())

    if not valid_dates:
        # st.warning("No valid datetime indices found across loaded data to determine range.")
        fallback_end = datetime.date.today()
        fallback_start = fallback_end - datetime.timedelta(days=30)
        return fallback_start, fallback_end

    valid_dates_utc = []
    for d in valid_dates:
        try:
            if d.tz is None:
                valid_dates_utc.append(d.tz_localize('UTC'))
            else:
                valid_dates_utc.append(d.tz_convert('UTC'))
        except Exception as tz_conv_err:
            # st.warning(f"Could not convert date {d} to UTC for comparison: {tz_conv_err}. Skipping this date.")
            continue

    if not valid_dates_utc:
        #  st.warning("No valid dates remained after attempting UTC conversion.")
         fallback_end = datetime.date.today()
         fallback_start = fallback_end - datetime.timedelta(days=30)
         return fallback_start, fallback_end

    global_min_utc = min(valid_dates_utc)
    global_max_utc = max(valid_dates_utc)

    # Return as date objects
    return global_min_utc.date(), global_max_utc.date()

# --- Data Preparation Function (Tidy Data Principle) ---
def prepare_analysis_data(df_in, actual_col, forecast_col, price_diff_col=None):
    """
    Takes the filtered, standardized DataFrame and calculates all necessary derived columns.
    Returns a new DataFrame ready for analysis and plotting.
    """
    if not isinstance(df_in, pd.DataFrame) or df_in.empty:
        return pd.DataFrame() # Return empty if input is invalid

    df = df_in.copy() # Work on a copy

    # Ensure required numeric columns are numeric, coerce errors to NaN
    df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')
    df[forecast_col] = pd.to_numeric(df[forecast_col], errors='coerce')

    # 1. Calculate Forecast Error
    if actual_col in df.columns and forecast_col in df.columns:
        df[FORECAST_ERROR_COL] = df[actual_col] - df[forecast_col]
    else:
        st.warning(f"Could not calculate '{FORECAST_ERROR_COL}'. Missing input columns.")
        df[FORECAST_ERROR_COL] = np.nan

    # 2. Calculate Hourly Cost (Requires Price Difference)
    # Ensure price diff col exists and is numeric
    if price_diff_col and price_diff_col in df.columns:
         df[price_diff_col] = pd.to_numeric(df[price_diff_col], errors='coerce')
         # Calculate cost only where both error and price diff are valid numbers
         df[HOURLY_COST_COL] = df[FORECAST_ERROR_COL] * df[price_diff_col]
    else:
        # Create the column as NaN if price diff isn't available
        df[HOURLY_COST_COL] = np.nan
        # Optionally warn if price_diff_col was expected but missing
        # if price_diff_col:
        #     st.info(f"Price difference column '{price_diff_col}' not found or invalid. '{HOURLY_COST_COL}' not calculated.")

    # 3. Calculate Absolute Percentage Error (APE)
    if FORECAST_ERROR_COL in df.columns and actual_col in df.columns:
        # Avoid division by zero or near-zero, and handle potential NaNs
        actual_load_safe = df[actual_col].replace(0, np.nan) # Replace 0 with NaN
        df[APE_COL] = (df[FORECAST_ERROR_COL].abs() / actual_load_safe.abs()) * 100
        # Replace infinite values that might arise from near-zero actual load
        df[APE_COL] = df[APE_COL].replace([np.inf, -np.inf], np.nan)
    else:
        df[APE_COL] = np.nan

    # 4. Add Time Features
    if isinstance(df.index, pd.DatetimeIndex):
        df[HOUR_COL] = df.index.hour
        df[DOW_COL] = df.index.dayofweek # Monday=0, Sunday=6
    else:
        df[HOUR_COL] = np.nan
        df[DOW_COL] = np.nan

    return df

# --- Memoized Stats Calculations ---
@st.cache_resource(ttl=60*60) # Cache for 1 hour, or adjust as needed
def calculate_acf(_series, nlags=48, fft=True):
    """Calculates ACF using statsmodels, returns lags, ACF values, and confint."""
    if not isinstance(_series, pd.Series) or _series.isnull().all() or len(_series.dropna()) < nlags * 2:
        return np.array([]), np.array([]), np.array([]) # Return empty arrays if insufficient data

    series_clean = _series.dropna()
    try:
        acf_values, confint = sm.tsa.acf(series_clean, nlags=nlags, alpha=0.05, fft=fft)
        lags = np.arange(nlags + 1) # Lags from 0 to nlags
        # Return lags, acf, and confint (lower, upper bounds relative to acf)
        return lags, acf_values, confint # confint[:,0]-acf_values, confint[:,1]-acf_values gives interval relative to 0
    except Exception as e:
        st.warning(f"ACF calculation failed: {e}")
        return np.array([]), np.array([]), np.array([])

@st.cache_resource(ttl=60*60)
def calculate_ccf(_series1, _series2, maxlags=48):
    """Calculates CCF using statsmodels for lags -maxlags to +maxlags."""
    if not isinstance(_series1, pd.Series) or not isinstance(_series2, pd.Series):
        return np.array([]), np.array([])
    
    # Align series and drop NaNs where *either* series is NaN
    df_temp = pd.DataFrame({'s1': _series1, 's2': _series2}).dropna()
    if len(df_temp) < maxlags * 2: # Need sufficient overlapping data
        return np.array([]), np.array([])

    s1_clean = df_temp['s1'].values
    s2_clean = df_temp['s2'].values
    n_obs = len(s1_clean)

    try:
        # CCF(x, y) calculates Cor(x[t], y[t+k]) for k >= 0
        ccf_pos_lags = sm.tsa.stattools.ccf(s1_clean, s2_clean, adjusted=False)[ : maxlags + 1] # Lags 0 to maxlags

        # For negative lags (k < 0), Cor(x[t], y[t+k]) = Cor(y[t], x[t-k])
        # Calculate CCF(y, x) for lags 1 to maxlags
        ccf_neg_lags_rev = sm.tsa.stattools.ccf(s2_clean, s1_clean, adjusted=False)[1 : maxlags + 1] # Lags 1 to maxlags for CCF(y,x)

        # Combine: [neg lags reversed, lag 0, pos lags]
        ccf_final = np.concatenate([ccf_neg_lags_rev[::-1], ccf_pos_lags])
        lags = np.arange(-maxlags, maxlags + 1)

        # Calculate approximate confidence intervals
        conf_level = 1.96 / np.sqrt(n_obs)

        return lags, ccf_final, conf_level
    except Exception as e:
        st.warning(f"CCF calculation failed: {e}")
        return np.array([]), np.array([]), np.nan

# --- Streamlit App Setup ---
st.set_page_config(layout="wide")
st.title(f"ERCOT Load Forecast: Performance & Financial Impact")

# --- Load Data ---
all_iso_data = load_all_data_cached()

# --- Check Data Loading Status ---
if not all_iso_data:
    st.error("Data loading failed or returned no data. Aborting application.")
    st.stop()

# --- Get ERCOT Data and Validate ---
# st.info(f"Attempting to load data for key: '{TARGET_ISO_KEY}'") # Less verbose
df_ercot_raw = all_iso_data.get(TARGET_ISO_KEY)

if df_ercot_raw is None:
    st.error(f"Could not retrieve data for the key '{TARGET_ISO_KEY}'. Check 'config.json' and data loading functions.")
    st.info(f"Available keys in loaded data: {list(all_iso_data.keys())}")
    st.stop()
if not isinstance(df_ercot_raw, pd.DataFrame):
    st.error(f"Data retrieved for '{TARGET_ISO_KEY}' is not a Pandas DataFrame (Type: {type(df_ercot_raw)}). Cannot proceed.")
    st.stop()
if df_ercot_raw.empty:
    st.error(f"Data for '{TARGET_ISO_KEY}' was loaded but the DataFrame is empty. Cannot proceed.")
    st.stop()
# st.success(f"Successfully retrieved raw data for '{TARGET_ISO_KEY}'. Shape: {df_ercot_raw.shape}") # Less verbose

# --- Validate ERCOT DataFrame (Index, Timezone, Required Input Columns) ---
try:
    # 1. Ensure DatetimeIndex
    if not isinstance(df_ercot_raw.index, pd.DatetimeIndex):
        st.info("Attempting to convert ERCOT DataFrame index to DatetimeIndex.")
        original_index_name = df_ercot_raw.index.name
        df_ercot_raw.index = pd.to_datetime(df_ercot_raw.index, errors='coerce')
        initial_rows = len(df_ercot_raw)
        # Drop rows where index conversion failed
        df_ercot_raw = df_ercot_raw[df_ercot_raw.index.notna()]
        dropped_rows = initial_rows - len(df_ercot_raw)
        if dropped_rows > 0:
             st.warning(f"Dropped {dropped_rows} rows due to invalid date conversion in the index.")
        if df_ercot_raw.empty:
            raise ValueError("Index conversion to DatetimeIndex failed or resulted in an empty DataFrame.")
        if original_index_name:
             df_ercot_raw.index.name = original_index_name
        st.success("Index successfully converted to DatetimeIndex.")

    # 2. Ensure Timezone Awareness
    if df_ercot_raw.index.tz is None:
        tz_info = ISO_CONFIG.get(TARGET_ISO_KEY, {}).get('timezone', 'UTC')
        st.info(f"Localizing ERCOT data timezone to '{tz_info}'.")
        try:
            df_ercot_raw = df_ercot_raw.tz_localize(tz_info)
        except Exception as tz_err:
             st.error(f"Failed to localize timezone to {tz_info}: {tz_err}. Cannot ensure timezone consistency.")
             # Attempt UTC localization as fallback
             try:
                 st.warning("Attempting to localize to UTC as fallback.")
                 df_ercot_raw = df_ercot_raw.tz_localize('UTC')
             except Exception as utc_err:
                  st.error(f"Fallback UTC localization also failed: {utc_err}. Stopping.")
                  st.stop()
    else:
        st.info(f"ERCOT data timezone is already set: {df_ercot_raw.index.tz}")

    # 3. Check Required *Input* Columns
    st.info(f"Checking for required input columns: {REQUIRED_INPUT_COLS}")
    missing_cols = [col for col in REQUIRED_INPUT_COLS if col not in df_ercot_raw.columns]
    if missing_cols:
        st.error(f"ERCOT data is missing required columns for core analysis: {missing_cols}")
        st.info(f"Available columns: {df_ercot_raw.columns.tolist()}")
        st.stop()
    st.success("Required input columns found.")

except Exception as e:
    st.error(f"Critical error during ERCOT data validation: {e}")
    st.exception(e)
    st.stop()


# --- Sidebar Date Range Selection ---
st.sidebar.header("Date Range Selection")
global_min_date, global_max_date = get_global_date_range(all_iso_data)

if global_min_date is None or global_max_date is None:
    st.sidebar.error("Could not determine a valid global date range. Using ERCOT's range as fallback.")
    if not df_ercot_raw.empty and isinstance(df_ercot_raw.index, pd.DatetimeIndex):
        global_min_date = df_ercot_raw.index.min().date()
        global_max_date = df_ercot_raw.index.max().date()
    else: # Absolute fallback
        global_max_date = datetime.date.today()
        global_min_date = global_max_date - datetime.timedelta(days=365)

# Use ERCOT's specific range if available and valid, otherwise use global
ercot_min_date = global_min_date
ercot_max_date = global_max_date
if not df_ercot_raw.empty and isinstance(df_ercot_raw.index, pd.DatetimeIndex):
     try:
         ercot_min_date = df_ercot_raw.index.min().date()
         ercot_max_date = df_ercot_raw.index.max().date()
     except Exception:
         pass

# Sensible defaults: last 7 days within available ERCOT range, clamped by global range
default_end = min(ercot_max_date, global_max_date)
default_start = max(global_min_date, ercot_min_date, default_end - datetime.timedelta(days=7)) # Default to 7 days

# Ensure start is not after end
if default_start > default_end:
    default_start = min(global_min_date, ercot_min_date)

st.sidebar.info(f"Available: {global_min_date} to {global_max_date}")
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=global_min_date, max_value=global_max_date)
end_date = st.sidebar.date_input("End Date", value=default_end, min_value=start_date, max_value=global_max_date)

# --- Filter ERCOT Data based on Selection ---
df_ercot_filtered = pd.DataFrame()
try:
    if start_date and end_date and isinstance(df_ercot_raw.index, pd.DatetimeIndex):
        tz = df_ercot_raw.index.tz
        if tz is None: # Should not happen after validation, but check anyway
            st.warning("Raw ERCOT data index lost timezone after validation. Assuming UTC for filtering.")
            tz = 'UTC'

        start_dt = pd.Timestamp(datetime.datetime.combine(start_date, datetime.time.min), tz=tz)
        end_dt = pd.Timestamp(datetime.datetime.combine(end_date + datetime.timedelta(days=1), datetime.time.min), tz=tz)

        # Convert start/end to the index's timezone for correct comparison
        try:
             start_dt = start_dt.tz_convert(df_ercot_raw.index.tz)
             end_dt = end_dt.tz_convert(df_ercot_raw.index.tz)
        except Exception as tz_conv_err:
             st.error(f"Error converting filter dates to data timezone ({df_ercot_raw.index.tz}): {tz_conv_err}")
             st.stop()

        mask = (df_ercot_raw.index >= start_dt) & (df_ercot_raw.index < end_dt)
        df_ercot_filtered = df_ercot_raw.loc[mask].copy()

        st.success(f"Filtered data for {start_date} to {end_date}. Shape: {df_ercot_filtered.shape}")
        if df_ercot_filtered.empty:
            st.warning(f"No ERCOT data available between {start_date} and {end_date}. Adjust the date range or check raw data.")
            # Don't stop, let the rest handle the empty DataFrame
    else:
         st.warning("Could not perform date filtering. Ensure dates are selected and raw data index is valid.")

except Exception as e:
     st.error(f"Error during date filtering: {e}")
     st.exception(e)
     st.stop()

# --- Standardize Index (Hourly) ---
df_ercot_std = pd.DataFrame()
if not df_ercot_filtered.empty:
    # st.info("Standardizing ERCOT data to uniform hourly index...") # Less verbose
    try:
        df_ercot_std = ensure_uniform_hourly_index(df_ercot_filtered, TARGET_ISO_KEY)
        if df_ercot_std.empty:
            st.warning("ERCOT data became empty after ensuring uniform hourly index. Check for large gaps or resampling issues.")
        else:
            # st.success(f"Index standardized. Shape after standardization: {df_ercot_std.shape}") # Less verbose
            pass
    except Exception as e:
        st.error(f"Error during index standardization for ERCOT: {e}")
        st.exception(e)
        # df_ercot_std remains empty
else:
    # If df_ercot_filtered was already empty, df_ercot_std should also be empty
    st.info("Skipping index standardization as filtered ERCOT data is empty.")

# --- Add Price Data (if available for ERCOT) ---
df_ercot_priced = pd.DataFrame()
if not df_ercot_std.empty:
    # st.info(f"Attempting to add/calculate price difference column: '{PRICE_DIFF_COL}'") # Less verbose
    try:
        # Pass a copy to potentially avoid modifying the original if the function does it in-place
        df_ercot_priced = add_price_data_to_existing_df(df_ercot_std.copy(), TARGET_ISO_KEY, target_column=PRICE_DIFF_COL)
        if PRICE_DIFF_COL not in df_ercot_priced.columns:
            st.warning(f"Could not find or calculate price difference data ('{PRICE_DIFF_COL}') for ERCOT in the selected range. Price-related analysis will be limited.")
        elif df_ercot_priced[PRICE_DIFF_COL].isnull().all():
            st.warning(f"Price difference column '{PRICE_DIFF_COL}' was added but contains only NaN values. Price-related analysis will be limited.")
        else:
            # st.success(f"Successfully added/found '{PRICE_DIFF_COL}' column.") # Less verbose
            pass
    except KeyError as ke:
        st.warning(f"Could not calculate '{PRICE_DIFF_COL}'. Missing required input column(s): {ke}. Price analysis will be limited.")
        df_ercot_priced = df_ercot_std.copy() # Proceed without price column
    except FileNotFoundError as fnf_err:
        st.warning(f"Price data file not found: {fnf_err}. Price analysis will be limited.")
        df_ercot_priced = df_ercot_std.copy() # Proceed without price column
    except Exception as e:
        st.error(f"An unexpected error occurred while adding price data for ERCOT: {e}")
        st.exception(e)
        df_ercot_priced = df_ercot_std.copy() # Proceed without price column
else:
    st.info("Skipping price data addition as standardized ERCOT data is empty.")
    df_ercot_priced = df_ercot_std.copy() # Ensure it's assigned even if empty

# --- Prepare Final Analysis DataFrame ---
# This is the single DataFrame used by all subsequent plots and calculations
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
    st.warning("No data available for analysis after filtering and preparation. Please adjust the date range or check data sources.")
    st.stop() # Stop if no data to analyze

# --- Section 1: KPI Cards ---
st.subheader("Forecast Performance KPIs")

# Calculate KPIs from df_analysis
kpi_bias = df_analysis[FORECAST_ERROR_COL].mean()
kpi_mape = df_analysis[APE_COL].mean() # Assumes APE_COL was calculated correctly
kpi_cost = df_analysis[HOURLY_COST_COL].sum()

# Calculate % hours outside 1 sigma
error_mean = df_analysis[FORECAST_ERROR_COL].mean()
error_std = df_analysis[FORECAST_ERROR_COL].std()
kpi_outside_1sigma = np.nan # Default
if pd.notna(error_std) and error_std > 0:
    outside_count = ((df_analysis[FORECAST_ERROR_COL] > error_mean + error_std) |
                     (df_analysis[FORECAST_ERROR_COL] < error_mean - error_std)).sum()
    total_valid_hours = df_analysis[FORECAST_ERROR_COL].notna().sum()
    if total_valid_hours > 0:
        kpi_outside_1sigma = (outside_count / total_valid_hours) * 100

# Define Thresholds (adjust as needed)
mape_threshold_warn = 3.0
mape_threshold_alert = 4.0
bias_threshold_warn = abs(df_analysis[ACTUAL_LOAD_COL].mean() * 0.01) # e.g., 1% of avg load
bias_threshold_alert = abs(df_analysis[ACTUAL_LOAD_COL].mean() * 0.02) # e.g., 2% of avg load
cost_threshold_alert = 10000 # Example threshold in $ for the period
sigma_threshold_warn = 35.0 # Approx 1/3 of hours outside is expected for normal dist
sigma_threshold_alert = 40.0

# Determine KPI colors
mape_color = "normal"
if pd.notna(kpi_mape):
    if kpi_mape > mape_threshold_alert: mape_color = "inverse" # Red if bad
    elif kpi_mape > mape_threshold_warn: mape_color = "off" # Amber if warning

bias_color = "normal"
if pd.notna(kpi_bias) and pd.notna(bias_threshold_alert) and pd.notna(bias_threshold_warn):
    if abs(kpi_bias) > bias_threshold_alert: bias_color = "inverse"
    elif abs(kpi_bias) > bias_threshold_warn: bias_color = "off"

cost_color = "normal" # Cost is usually just informational unless compared to budget
if pd.notna(kpi_cost) and abs(kpi_cost) > cost_threshold_alert:
    cost_color = "inverse" # Highlight large cost/gain

sigma_color = "normal"
if pd.notna(kpi_outside_1sigma):
     if kpi_outside_1sigma > sigma_threshold_alert: sigma_color = "inverse"
     elif kpi_outside_1sigma > sigma_threshold_warn: sigma_color = "off"


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="MAPE (%)", value=f"{kpi_mape:.2f}%" if pd.notna(kpi_mape) else "N/A",
              delta_color=mape_color, help="Mean Absolute Percentage Error. Lower is better.")
with col2:
    st.metric(label="Bias (MW)", value=f"{kpi_bias:,.1f}" if pd.notna(kpi_bias) else "N/A",
              delta_color=bias_color, help="Mean Forecast Error (Actual - Forecast). Closer to zero is better.")
with col3:
    # Check if cost column exists and has non-NaN values before displaying
    if HOURLY_COST_COL in df_analysis.columns and df_analysis[HOURLY_COST_COL].notna().any():
        st.metric(label="Cost of Error ($)", value=f"${kpi_cost:,.0f}" if pd.notna(kpi_cost) else "N/A",
                delta_color=cost_color, help="Sum of Hourly Error (MW) * Price Diff ($/MWh). Negative means over-forecasting saved money / under-forecasting cost money (assuming Error = Act-Fcst).")
    else:
        st.metric(label="Cost of Error ($)", value="N/A", help="Requires valid price difference data.")

with col4:
    st.metric(label="% Hours Outside ±1σ", value=f"{kpi_outside_1sigma:.1f}%" if pd.notna(kpi_outside_1sigma) else "N/A",
              delta_color=sigma_color, help="Percentage of hours where forecast error was more than 1 standard deviation from the mean error.")

st.markdown("---")

# --- Section 2: Timeline: "Error -> Cost" ---
st.subheader("Timeline: Forecast Error & Price Spreads")
st.markdown(f"""
Visually links forecast misses to real-time vs. day-ahead price differences. Large errors often coincide with price spikes.
- **Bars:** Forecast Error (`{FORECAST_ERROR_COL}`, Actual - Forecast). Red = Over-forecast (>0), Blue = Under-forecast (<0).
- **Line:** RT–DA Price Difference (`{PRICE_DIFF_COL}`, $/MWh) on the right axis.
- **Shaded Areas:** Weekends highlighted in grey.
""")

# Check if necessary columns exist
if FORECAST_ERROR_COL in df_analysis.columns and PRICE_DIFF_COL in df_analysis.columns and df_analysis[PRICE_DIFF_COL].notna().any():
    try:
        fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])

        # Determine reasonable y-range for error, clipping outliers for visual clarity might be useful
        error_abs_max = df_analysis[FORECAST_ERROR_COL].abs().quantile(0.99) # Cap at 99th percentile for range
        error_range = [-error_abs_max * 1.1, error_abs_max * 1.1] if pd.notna(error_abs_max) and error_abs_max > 0 else None

        # Price difference axis range
        price_abs_max = df_analysis[PRICE_DIFF_COL].abs().quantile(0.98) # Cap price axis too
        price_range_cap = 250 # As suggested
        price_final_max = max(price_abs_max * 1.1, price_range_cap) if pd.notna(price_abs_max) else price_range_cap
        price_range = [-price_final_max, price_final_max]


        # Split error for coloring bars
        error_pos = df_analysis[FORECAST_ERROR_COL].clip(lower=0)
        error_neg = df_analysis[FORECAST_ERROR_COL].clip(upper=0)

        # Add Error Bars (Primary Y-axis)
        fig_timeline.add_trace(go.Bar(x=df_analysis.index, y=error_pos, name='Over-forecast Error',
                                    marker_color='red', opacity=0.6), secondary_y=False)
        fig_timeline.add_trace(go.Bar(x=df_analysis.index, y=error_neg, name='Under-forecast Error',
                                    marker_color='blue', opacity=0.6), secondary_y=False)

        # Add Price Difference Line (Secondary Y-axis)
        fig_timeline.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis[PRICE_DIFF_COL], name='RT-DA Price Δ',
                                         mode='lines', line=dict(color='orange', width=2),
                                         connectgaps=True), secondary_y=True)

        # Add Weekend Shading
        df_weekends = df_analysis[df_analysis[DOW_COL] >= 5] # Saturday=5, Sunday=6
        for weekend_start_date in df_weekends.index.normalize().unique():
            start_ts = weekend_start_date
            end_ts = start_ts + pd.Timedelta(days=2) # Cover Sat and Sun
            fig_timeline.add_vrect(
                x0=start_ts, x1=end_ts,
                fillcolor="grey", opacity=0.15, layer="below", line_width=0,
            )

        fig_timeline.update_layout(
            title="Forecast Error (Bars, Left Axis) vs. Price Difference (Line, Right Axis)",
            xaxis_title="Date / Time",
            yaxis_title=f"Forecast Error ({FORECAST_ERROR_COL})",
            yaxis2_title=f"RT-DA Price Δ ({PRICE_DIFF_COL})",
            height=450,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='relative', # Stack pos/neg error parts correctly if needed, relative works well here
            bargap=0,
            yaxis=dict(range=error_range) if error_range else {}, # Apply error range if calculated
            yaxis2=dict(range=price_range)   # Apply price range
        )
        # Add zero lines for clarity
        fig_timeline.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", secondary_y=False)
        fig_timeline.add_hline(y=0, line_width=1, line_dash="dash", line_color="darkorange", opacity=0.5, secondary_y=True)

        st.plotly_chart(fig_timeline, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating Error/Price timeline plot: {e}")
        st.exception(e)
else:
    st.info(f"Cannot generate Error/Price timeline: Requires both '{FORECAST_ERROR_COL}' and '{PRICE_DIFF_COL}' with valid data.")

st.markdown("---")

# --- Section 3: Persistence Panel ---
st.subheader("Error Persistence & Price Correlation")

# Persistence Alert Banner Logic
persistence_alert = False
acf_lags, acf_values, acf_confint = calculate_acf(df_analysis[FORECAST_ERROR_COL], nlags=48) # Use cached function

if len(acf_values) > 24: # Ensure lag 1 and 24 exist
    # acf_confint gives [lower, upper] bounds for each lag
    ci_upper_lag1 = acf_confint[1, 1] if acf_confint.ndim == 2 and acf_confint.shape[0] > 1 else np.nan
    ci_lower_lag1 = acf_confint[1, 0] if acf_confint.ndim == 2 and acf_confint.shape[0] > 1 else np.nan
    ci_upper_lag24 = acf_confint[24, 1] if acf_confint.ndim == 2 and acf_confint.shape[0] > 24 else np.nan
    ci_lower_lag24 = acf_confint[24, 0] if acf_confint.ndim == 2 and acf_confint.shape[0] > 24 else np.nan
    
    # Check if lag 1 or lag 24 ACF value is outside its confidence interval
    lag1_outside = pd.notna(acf_values[1]) and pd.notna(ci_lower_lag1) and pd.notna(ci_upper_lag1) and (acf_values[1] < ci_lower_lag1 or acf_values[1] > ci_upper_lag1)
    lag24_outside = pd.notna(acf_values[24]) and pd.notna(ci_lower_lag24) and pd.notna(ci_upper_lag24) and (acf_values[24] < ci_lower_lag24 or acf_values[24] > ci_upper_lag24)

    if lag1_outside or lag24_outside:
        persistence_alert = True
        alert_reason = []
        if lag1_outside: alert_reason.append("lag-1")
        if lag24_outside: alert_reason.append("lag-24")
        st.warning(f"**Persistence Alert:** Forecast error autocorrelation at { ' and '.join(alert_reason) } is statistically significant. Consider bias correction or model review.")

col_persist1, col_persist2 = st.columns(2)

with col_persist1:
    st.markdown("**Forecast Error Autocorrelation (ACF)**")
    st.markdown("Shows if errors are correlated with past errors (lags 1-48h). Significant bars suggest predictable patterns remain.")
    if len(acf_lags) > 1:
        try:
            fig_acf = go.Figure()
            # Calculate CI bounds relative to 0
            ci_bound = acf_confint[:, 1] - acf_values # Upper bound distance from value
            # Use the first lag's CI as approximation for all for plotting (statsmodels uses Bartlett's formula, varies slightly)
            # A simpler fixed CI:
            n_obs_acf = len(df_analysis[FORECAST_ERROR_COL].dropna())
            conf_level_acf = 1.96 / np.sqrt(n_obs_acf) if n_obs_acf > 0 else 0

            # Plot bars for lags 1 to nlags
            fig_acf.add_trace(go.Bar(
                x=acf_lags[1:], y=acf_values[1:], name='ACF', marker_color='steelblue'
            ))
            # Add CI lines
            fig_acf.add_hline(y=conf_level_acf, line=dict(color='rgba(255,0,0,0.5)', dash='dash'), name='95% CI')
            fig_acf.add_hline(y=-conf_level_acf, line=dict(color='rgba(255,0,0,0.5)', dash='dash'), showlegend=False)
            fig_acf.add_hline(y=0, line=dict(color='grey', width=1))

            fig_acf.update_layout(
                # title="Forecast Error Autocorrelation (ACF)",
                xaxis_title="Lag (Hours)", yaxis_title="Autocorrelation",
                yaxis_range=[-1, 1], height=350, margin=dict(l=40, r=20, t=30, b=40), showlegend=False
            )
            st.plotly_chart(fig_acf, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting ACF: {e}")
    else:
        st.info("Not enough data to calculate ACF.")

with col_persist2:
    st.markdown("**Error vs. Price Cross-Correlation (CCF)**")
    st.markdown("Shows correlation between Error(t) and PriceΔ(t+lag). Significant bars indicate lead/lag relationships.")
    
    # Check if price diff exists and is valid
    if PRICE_DIFF_COL in df_analysis.columns and df_analysis[PRICE_DIFF_COL].notna().any():
        ccf_lags, ccf_values, ccf_conf_level = calculate_ccf(
            df_analysis[FORECAST_ERROR_COL], df_analysis[PRICE_DIFF_COL], maxlags=48
        )

        if len(ccf_lags) > 0:
            try:
                fig_ccf = go.Figure()
                # Add CCF bars
                colors = ['red' if abs(val) > ccf_conf_level else 'steelblue' for val in ccf_values]
                fig_ccf.add_trace(go.Bar(
                    x=ccf_lags, y=ccf_values, name='CCF', marker_color=colors
                ))
                # Add CI lines
                fig_ccf.add_hline(y=ccf_conf_level, line=dict(color='rgba(255,0,0,0.5)', dash='dash'), name='95% CI')
                fig_ccf.add_hline(y=-ccf_conf_level, line=dict(color='rgba(255,0,0,0.5)', dash='dash'), showlegend=False)
                fig_ccf.add_hline(y=0, line=dict(color='grey', width=1))
                fig_ccf.add_vline(x=0, line=dict(color='grey', width=1, dash='dot')) # Highlight lag 0

                fig_ccf.update_layout(
                    # title="Cross-Correlation: Error vs. Price Difference",
                    xaxis_title="Lag (Hours) [Error(t) vs PriceΔ(t+lag)]",
                    yaxis_title="Correlation",
                    yaxis_range=[-1, 1], height=350, margin=dict(l=40, r=20, t=30, b=40), showlegend=False
                )
                st.plotly_chart(fig_ccf, use_container_width=True)
                st.caption("Red bars indicate correlation outside approx. 95% confidence interval.")
            except Exception as e:
                st.error(f"Error plotting CCF: {e}")
        else:
            st.info("Not enough overlapping data for Error and Price Difference to calculate CCF.")
    else:
        st.info(f"Price difference column '{PRICE_DIFF_COL}' not available for CCF.")

st.markdown("---")

# --- Section 4: Scatter/Heat "How bad is expensive?" ---
st.subheader("How Bad is Expensive? Error vs. Price Difference")
st.markdown(f"""
Shows the relationship between forecast error size (`{FORECAST_ERROR_COL}`) and the resulting price difference (`{PRICE_DIFF_COL}`).
Warmer colors indicate more hours fall into that specific Error/Price combination. Helps identify the most common (and costly) error types.
""")

if FORECAST_ERROR_COL in df_analysis.columns and PRICE_DIFF_COL in df_analysis.columns and df_analysis[[FORECAST_ERROR_COL, PRICE_DIFF_COL]].notna().all(axis=1).any():
    try:
        # Drop NaNs for this specific plot
        df_scatter = df_analysis[[FORECAST_ERROR_COL, PRICE_DIFF_COL]].dropna()

        # Optional: Clip axes for better visualization of the dense area
        error_clip = df_scatter[FORECAST_ERROR_COL].quantile([0.01, 0.99])
        price_clip = df_scatter[PRICE_DIFF_COL].quantile([0.01, 0.99])
        # Ensure clips don't invert; use a default if needed
        error_range_plot = [error_clip.iloc[0], error_clip.iloc[1]] if error_clip.iloc[0] < error_clip.iloc[1] else None
        price_range_plot = [price_clip.iloc[0], price_clip.iloc[1]] if price_clip.iloc[0] < price_clip.iloc[1] else None
        price_range_plot = [-250, 250] # Force cap as requested

        fig_heat = go.Figure(go.Histogram2dContour(
                x = df_scatter[FORECAST_ERROR_COL],
                y = df_scatter[PRICE_DIFF_COL],
                colorscale = 'Blues', # Or 'Jet', 'Viridis' etc.
                # ncontours=15,
                contours=dict(coloring='heatmap', showlabels=False), # Show density heatmap
                line=dict(width=0), # Hide contour lines
                hoverinfo='skip' # Skip hover for contours
        ))
        # Overlay with scatter plot for sparse points (optional, can be slow)
        fig_heat.add_trace(go.Scattergl(
                x=df_scatter[FORECAST_ERROR_COL],
                y=df_scatter[PRICE_DIFF_COL],
                mode='markers',
                marker=dict(color='rgba(0,0,0,0.3)', size=3),
                name='Hourly Data',
                hovertext = df_scatter.index.strftime('%Y-%m-%d %H:%M'),
                hoverinfo='text+x+y'
        ))


        fig_heat.update_layout(
            title="Density: Forecast Error vs. Price Difference",
            xaxis_title=f"Forecast Error ({FORECAST_ERROR_COL})",
            yaxis_title=f"RT-DA Price Δ ({PRICE_DIFF_COL})",
            height=450,
            hovermode='closest',
            xaxis=dict(range=error_range_plot) if error_range_plot else {},
            yaxis=dict(range=price_range_plot) if price_range_plot else {},
            showlegend=False,
            coloraxis_showscale=True, # Show color bar for density
            coloraxis_colorbar=dict(title='Density')
        )
        fig_heat.add_vline(x=0, line=dict(color='grey', dash='dash'))
        fig_heat.add_hline(y=0, line=dict(color='grey', dash='dash'))
        st.plotly_chart(fig_heat, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating Error vs Price heatmap: {e}")
        st.exception(e)
else:
    st.info(f"Cannot generate Error vs Price heatmap: Requires both '{FORECAST_ERROR_COL}' and '{PRICE_DIFF_COL}' with valid data.")

st.markdown("---")

# --- Section 5: Cost-of-Error Histogram ---
st.subheader("Distribution of Hourly Cost of Error")
st.markdown(f"""
Shows the frequency of different financial outcomes (`{HOURLY_COST_COL}`) caused by forecast errors in each hour.
Highlights the impact of tail events (large cost/gain hours) on overall P&L.
Calculation: `Hourly Cost = Forecast Error (MW) × (RT LMP – DA LMP) ($/MWh)`
""")

if HOURLY_COST_COL in df_analysis.columns and df_analysis[HOURLY_COST_COL].notna().any():
    try:
        cost_data = df_analysis[HOURLY_COST_COL].dropna()

        # Determine range, clipping extreme outliers for visualization
        cost_clip = cost_data.quantile([0.01, 0.99])
        cost_range_plot = [cost_clip.iloc[0], cost_clip.iloc[1]] if pd.notna(cost_clip.iloc[0]) and pd.notna(cost_clip.iloc[1]) and cost_clip.iloc[0] < cost_clip.iloc[1] else None


        fig_cost_hist = go.Figure()
        fig_cost_hist.add_trace(go.Histogram(
            x=cost_data,
            name='Hourly Cost Freq.',
            marker_color='rgb(100,100,180)',
            xbins=dict(size=max(100, int(cost_data.std() / 4))) if pd.notna(cost_data.std()) and cost_data.std() > 0 else None # Auto-bin size based on std dev, min $100
        ))

        mean_cost = cost_data.mean()
        median_cost = cost_data.median()

        fig_cost_hist.add_vline(x=0, line_width=2, line_dash="dash", line_color="grey",
                                annotation_text="Zero Cost", annotation_position="top left")
        if pd.notna(mean_cost):
            fig_cost_hist.add_vline(x=mean_cost, line_width=2, line_dash="dot", line_color="orange",
                                    annotation_text=f"Mean: ${mean_cost:,.0f}",
                                    annotation_position="top right" if mean_cost >= 0 else "bottom right")

        fig_cost_hist.update_layout(
            title="Frequency Distribution of Hourly Cost of Forecast Error ($)",
            xaxis_title=f"Hourly Cost of Error ({HOURLY_COST_COL})",
            yaxis_title="Frequency (Count of Hours)",
            height=400,
            bargap=0.1,
            xaxis=dict(range=cost_range_plot) if cost_range_plot else {}, # Apply clipped range
        )
        st.plotly_chart(fig_cost_hist, use_container_width=True)
        st.caption(f"Total Cost of Error for period: ${cost_data.sum():,.0f}")

    except Exception as e:
        st.error(f"Error generating Cost of Error histogram: {e}")
        st.exception(e)
else:
    st.info(f"Cannot generate Cost of Error histogram: Requires '{HOURLY_COST_COL}' calculated from error and price difference data.")

st.markdown("---")

# --- Section 6: Hour-of-day heatmap (optional toggle) ---
show_hourly_heatmap = st.toggle("Show Hourly Price Difference Heatmap", value=False)

if show_hourly_heatmap:
    st.subheader("Average Price Difference by Hour and Day of Week")
    st.markdown(f"""
    Shows the typical RT–DA Price Difference (`{PRICE_DIFF_COL}`) for each hour across the days of the week.
    Helps identify systematic time-based pricing patterns relevant to forecasting or trading strategies.
    """)
    if PRICE_DIFF_COL in df_analysis.columns and df_analysis[PRICE_DIFF_COL].notna().any() and HOUR_COL in df_analysis.columns and DOW_COL in df_analysis.columns:
        try:
            hourly_pivot = df_analysis.pivot_table(
                index=HOUR_COL,
                columns=DOW_COL,
                values=PRICE_DIFF_COL,
                aggfunc='mean'
            )
            # Ensure all hours 0-23 and days 0-6 are present
            hourly_pivot = hourly_pivot.reindex(index=range(24), columns=range(7))

            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            fig_hour_heatmap = go.Figure(data=go.Heatmap(
                   z=hourly_pivot.values,
                   x=day_names, # Use names for columns
                   y=hourly_pivot.index, # Hours 0-23 for rows
                   colorscale='RdBu', # Red-Blue diverging scale is good for price diff
                   zmid=0, # Center color scale at zero
                   hoverongaps = False,
                   hovertemplate = "<b>Day:</b> %{x}<br><b>Hour:</b> %{y}:00<br><b>Avg PriceΔ:</b> %{z:.2f} $/MWh<extra></extra>"
                   ))
            fig_hour_heatmap.update_layout(
                title='Average RT-DA Price Difference ($/MWh) by Hour and Day',
                xaxis_title='Day of Week',
                yaxis_title='Hour of Day',
                yaxis=dict(tickmode='linear', dtick=2), # Show every 2nd hour label
                height=550
            )
            st.plotly_chart(fig_hour_heatmap, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating hourly price difference heatmap: {e}")
            st.exception(e)
    else:
        st.info(f"Cannot generate hourly price heatmap: Requires '{PRICE_DIFF_COL}', '{HOUR_COL}', and '{DOW_COL}'.")


# --- Advanced Diagnostics Expander ---
with st.expander("Advanced Diagnostics & Original Plots"):
    st.markdown("Detailed plots for deeper analysis. These were part of the original dashboard or provide additional context.")

    # --- Plot: Load vs Forecast and Forecast Error (Original Plot 1) ---
    st.subheader("Load vs. Forecast & Error Time Series (Detailed)")
    st.markdown(f"Shows actual load (`{ACTUAL_LOAD_COL}`) vs. forecast (`{FORECAST_LOAD_COL}`), and the resulting error (`{FORECAST_ERROR_COL}`).")
    try:
        plot1_cols_adv = [ACTUAL_LOAD_COL, FORECAST_LOAD_COL, FORECAST_ERROR_COL]
        if all(col in df_analysis.columns for col in plot1_cols_adv) and df_analysis[plot1_cols_adv].notna().any().any():
            fig1_adv = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                row_heights=[0.6, 0.4],
                subplot_titles=(
                    f"Actual Load vs. Forecast Load",
                    f"Forecast Error"
                )
            )

            # --- Subplot 1: Actual Load vs. Forecast ---
            fig1_adv.add_trace(go.Scatter(
                x=df_analysis.index, y=df_analysis[ACTUAL_LOAD_COL], name='Actual Load',
                mode='lines', line=dict(color='rgba(0,100,80,1)', width=1.5), fill='tozeroy', fillcolor='rgba(0,100,80,0.2)', connectgaps=True
            ), row=1, col=1)
            fig1_adv.add_trace(go.Scatter(
                x=df_analysis.index, y=df_analysis[FORECAST_LOAD_COL], name='Forecast Load',
                mode='lines', line=dict(color='rgba(0,0,255,0.8)', width=1), connectgaps=True
            ), row=1, col=1)
            fig1_adv.update_yaxes(title_text="Load (MW)", row=1, col=1)

            # --- Subplot 2: Forecast Error ---
            error_pos_adv = df_analysis[FORECAST_ERROR_COL].clip(lower=0)
            error_neg_adv = df_analysis[FORECAST_ERROR_COL].clip(upper=0)
            fig1_adv.add_trace(go.Scatter(x=df_analysis.index, y=error_pos_adv, name='Over-forecast',
                                        mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.4)', connectgaps=False), row=2, col=1)
            fig1_adv.add_trace(go.Scatter(x=df_analysis.index, y=error_neg_adv, name='Under-forecast',
                                        mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.4)', connectgaps=False), row=2, col=1)
            # Add lines over fill for better definition at zero crossing
            fig1_adv.add_trace(go.Scatter(x=df_analysis.index, y=error_pos_adv.replace(0, np.nan), showlegend=False,
                                        mode='lines', line=dict(color='rgba(255,0,0,0.6)', width=1), connectgaps=False), row=2, col=1)
            fig1_adv.add_trace(go.Scatter(x=df_analysis.index, y=error_neg_adv.replace(0, np.nan), showlegend=False,
                                        mode='lines', line=dict(color='rgba(0,0,255,0.6)', width=1), connectgaps=False), row=2, col=1)

            fig1_adv.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=2, col=1)

            # Add 7-day MA if enough data
            window_7d = 24 * 7
            if len(df_analysis[FORECAST_ERROR_COL].dropna()) >= window_7d:
                df_analysis['Error_MA_7D'] = df_analysis[FORECAST_ERROR_COL].rolling(window=window_7d, min_periods=24).mean()
                fig1_adv.add_trace(go.Scatter(
                    x=df_analysis.index, y=df_analysis['Error_MA_7D'], name='7-Day Avg Error',
                    mode='lines', line=dict(color='rgba(80,80,80,0.9)', width=1.5, dash='dot'), connectgaps=True
                ), row=2, col=1)

            fig1_adv.update_yaxes(title_text="Forecast Error (MW)", row=2, col=1)
            fig1_adv.update_xaxes(title_text="Date / Time", row=2, col=1)
            fig1_adv.update_layout(height=600, hovermode='x unified',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            title_text="Detailed Load vs. Forecast and Error Analysis")
            st.plotly_chart(fig1_adv, use_container_width=True)
        else:
            st.info("Required columns for detailed Load/Forecast plot not available or contain only NaNs.")
    except Exception as e:
        st.error(f"Error generating detailed Load/Forecast plot: {e}")
        st.exception(e)

    # --- Plot: Cumulative Forecast Bias (Original Plot 7) ---
    st.subheader("Cumulative Forecast Bias Over Time")
    st.markdown("Shows the running total of forecast error, highlighting persistent bias trends.")
    try:
        if FORECAST_ERROR_COL in df_analysis.columns and df_analysis[FORECAST_ERROR_COL].notna().any():
            # Calculate cumulative error safely, filling potential NaNs in error with 0 before summing
            df_analysis['Cumulative Error'] = df_analysis[FORECAST_ERROR_COL].fillna(0).cumsum()

            fig_cumul = go.Figure()
            fig_cumul.add_trace(go.Scatter(
                x=df_analysis.index, y=df_analysis['Cumulative Error'], mode='lines',
                name='Cumulative Error', line=dict(color='purple', width=2), connectgaps=True
            ))
            fig_cumul.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
            fig_cumul.update_layout(
                title="Cumulative Forecast Error (Bias Trend)", xaxis_title="Date / Time",
                yaxis_title="Cumulative Error (MW)", height=400, hovermode='x unified'
            )
            st.plotly_chart(fig_cumul, use_container_width=True)
        else:
            st.info(f"Cumulative bias plot requires '{FORECAST_ERROR_COL}'.")
    except Exception as e:
        st.error(f"Error generating cumulative error plot: {e}")
        st.exception(e)

    # --- Plot: Error vs. Actual Load Scatter (Original Plot 6) ---
    st.subheader("Forecast Error vs. Actual Load Level")
    st.markdown("Shows if forecast accuracy changes depending on the system load level.")
    try:
        scatter_cols_adv = [FORECAST_ERROR_COL, ACTUAL_LOAD_COL]
        if all(col in df_analysis.columns for col in scatter_cols_adv) and df_analysis[scatter_cols_adv].notna().all(axis=1).any():
            df_scatter_adv = df_analysis[scatter_cols_adv].dropna().copy()

            fig_scatter_adv = go.Figure()
            fig_scatter_adv.add_trace(go.Scattergl(
                x=df_scatter_adv[ACTUAL_LOAD_COL], y=df_scatter_adv[FORECAST_ERROR_COL], mode='markers', name='Hourly Error',
                marker=dict(color='rgba(0, 128, 128, 0.5)', size=4),
                customdata=df_scatter_adv.index,
                hovertemplate=(f"<b>Time:</b> %{{customdata|%Y-%m-%d %H:%M}}<br>"
                               f"<b>Actual Load:</b> %{{x:.0f}} MW<br>"
                               f"<b>Forecast Error:</b> %{{y:.0f}} MW<extra></extra>")
            ))
            fig_scatter_adv.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
            fig_scatter_adv.update_layout(
                title="Forecast Error vs. Actual Load", xaxis_title=f"Actual Load (MW)", yaxis_title=f"Forecast Error (MW)",
                height=450, hovermode='closest'
            )
            st.plotly_chart(fig_scatter_adv, use_container_width=True)
        else:
             st.info(f"Error vs Load scatter requires '{FORECAST_ERROR_COL}' and '{ACTUAL_LOAD_COL}'.")
    except Exception as e:
        st.error(f"Error generating error vs. load scatter plot: {e}")
        st.exception(e)

    # --- Other plots previously included can be added here as needed ---
    # e.g., APE plot, Hourly magnitude/count plots, original binned plots if desired by users.


# --- Tab 2: ISO Comparison (Placeholder - Keep as is) ---
# (No changes needed to Tab 2 based on the request)
# tab1, tab2 = st.tabs(["ERCOT Analysis", "ISO Comparison"]) # Use if needed
# with tab2:
#     st.header("ISO Comparison (Placeholder)")
#     st.markdown("This section is intended for comparing metrics across different ISOs.")
#     st.warning("ISO comparison functionality requires significant data harmonization and is not yet implemented.")