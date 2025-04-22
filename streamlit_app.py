import sys # Keep for potential future use, though not strictly necessary now
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from collections import defaultdict
# import scipy.stats # No longer explicitly required by this version's plots

# --- Module Imports with Error Handling ---
# Keep necessary error reporting during app startup
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

try:
    from metrics_calculation import compute_iso_metrics
except ImportError as e:
    st.error(f"Error importing from 'metrics_calculation.py': {e}")
    st.error("Please ensure 'metrics_calculation.py' is in the same directory and contains 'compute_iso_metrics'.")
    st.stop()

# --- Configuration ---
# Keep necessary error reporting during app startup
try:
    ISO_CONFIG = load_config()
    if ISO_CONFIG is None:
        raise ValueError("load_config returned None")
except Exception as e:
    st.error(f"Failed to load configuration from config.json: {e}. Cannot proceed.")
    st.stop()

TARGET_ISO_KEY = "ERCOT_Load_From_ISO"

# Define standard column names expected by analysis functions
ACTUAL_LOAD_COL = 'TOTAL Actual Load (MW)'
FORECAST_LOAD_COL = 'SystemTotal Forecast Load (MW)'
FORECAST_ERROR_COL = 'Forecast Error (MW)'
PRICE_DIFF_COL = "LMP Difference (USD)" # Assumed column name for price difference

REQUIRED_COLS_ERCOT = [ACTUAL_LOAD_COL, FORECAST_LOAD_COL, FORECAST_ERROR_COL]

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
def get_global_date_range(iso_data_dict):
    """Calculates the minimum start and maximum end date across all loaded DataFrames."""
    valid_dates = []
    for key, df in iso_data_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                 try:
                     # Attempt conversion, skip if fails badly
                     df_index_conv = pd.to_datetime(df.index, errors='coerce')
                     df_temp = df[~df_index_conv.isna()]
                     if not df_temp.empty:
                         df.index = df_index_conv[~df_index_conv.isna()]
                     else:
                         st.warning(f"Could not convert index to datetime for ISO '{key}'. Skipping date range calculation for this ISO.")
                         continue
                 except Exception as conv_err:
                     st.warning(f"Error converting index for ISO '{key}': {conv_err}. Skipping.")
                     continue

            # Now we know it's a DatetimeIndex (or skipped)
            if isinstance(df.index, pd.DatetimeIndex) and pd.api.types.is_datetime64_any_dtype(df.index):
                current_index = df.index
                if current_index.tz is None:
                    # Try to localize using config, default to UTC
                    tz_info_str = ISO_CONFIG.get(key, {}).get('timezone', 'UTC')
                    try:
                        current_index = current_index.tz_localize(tz_info_str)
                    except Exception as tz_err:
                        # st.warning(f"Could not localize index for {key} to {tz_info_str} ({tz_err}). Assuming UTC.")
                        try:
                           current_index = current_index.tz_localize('UTC')
                        except Exception:
                           # st.warning(f"Could not localize index for {key} to UTC either. Skipping this ISO for date range.")
                           continue # Skip if localization fails completely
                if not current_index.empty:
                    valid_dates.append(current_index.min())
                    valid_dates.append(current_index.max())

    if not valid_dates:
        st.warning("No valid datetime indices found across loaded data to determine range.")
        fallback_end = datetime.date.today()
        fallback_start = fallback_end - datetime.timedelta(days=30)
        return fallback_start, fallback_end

    # Ensure all dates are timezone-aware (UTC) for comparison
    valid_dates_utc = [d.tz_convert('UTC') if d.tz is not None else d.tz_localize('UTC') for d in valid_dates]

    global_min_utc = min(valid_dates_utc)
    global_max_utc = max(valid_dates_utc)

    # Return as date objects
    return global_min_utc.date(), global_max_utc.date()


# --- Helper: Format Metrics ---
def format_metric_value(value, metric_name):
    """Formats a metric value based on its name for display."""
    if pd.isna(value):
        return 'N/A'
    metric_name_lower = metric_name.lower()
    # Prioritize % formatting
    if 'mape' in metric_name_lower or '%' in metric_name:
        return f"{value:.2f}%"
    elif 'mw' in metric_name_lower or 'load' in metric_name_lower or 'error' in metric_name_lower :
        return f"{value:.1f}"
    elif 'price' in metric_name_lower or 'usd' in metric_name_lower:
         return f"{value:,.2f}" # Add comma for thousands
    elif isinstance(value, (int, float)):
        # General numeric formatting if no specific keyword matches
        return f"{value:.2f}"
    else:
        # Return as string if not numeric or handled above
        return str(value)

# --- Streamlit App Setup ---
st.set_page_config(layout="wide")
st.title(f"ERCOT Load Forecast Analysis")

# --- Load Data ---
all_iso_data = load_all_data_cached()

# --- Check Data Loading Status ---
if not all_iso_data:
    st.error("Data loading failed or returned no data. Aborting application.")
    st.stop()

# --- Get ERCOT Data and Validate ---
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

# --- Validate ERCOT DataFrame (Index, Timezone, Columns) ---
try:
    # 1. Ensure DatetimeIndex
    if not isinstance(df_ercot_raw.index, pd.DatetimeIndex):
        st.info("Attempting to convert ERCOT DataFrame index to DatetimeIndex.")
        original_index_name = df_ercot_raw.index.name # Preserve name if exists
        df_ercot_raw.index = pd.to_datetime(df_ercot_raw.index, errors='coerce')
        df_ercot_raw = df_ercot_raw.dropna(axis=0, subset=[df_ercot_raw.index.name] if original_index_name else None) # Drop rows where conversion failed
        if not isinstance(df_ercot_raw.index, pd.DatetimeIndex) or df_ercot_raw.empty:
            raise ValueError("Index conversion to DatetimeIndex failed or resulted in an empty DataFrame.")
        if original_index_name:
             df_ercot_raw.index.name = original_index_name

    # 2. Ensure Timezone Awareness
    if df_ercot_raw.index.tz is None:
        tz_info = ISO_CONFIG.get(TARGET_ISO_KEY, {}).get('timezone', 'UTC') # Default to UTC
        st.info(f"Localizing ERCOT data timezone to '{tz_info}'.")
        try:
            # Use tz_localize for naive -> aware
            df_ercot_raw = df_ercot_raw.tz_localize(tz_info)
        except TypeError:
             # If already aware but different, convert
             st.warning(f"Index already timezone-aware. Attempting conversion to '{tz_info}'.")
             try:
                 df_ercot_raw = df_ercot_raw.tz_convert(tz_info)
             except Exception as tz_convert_err:
                 st.error(f"Failed conversion to {tz_info}: {tz_convert_err}. Falling back to UTC.")
                 try:
                     df_ercot_raw = df_ercot_raw.tz_convert('UTC')
                 except Exception as utc_convert_err:
                     st.error(f"Fallback conversion to UTC also failed: {utc_convert_err}. Cannot ensure timezone consistency.")
                     # Decide whether to stop or proceed with potentially incorrect TZ
                     # st.stop() # Option: Stop if TZ is critical
        except Exception as tz_localize_err:
             st.error(f"Failed to localize timezone to {tz_info}: {tz_localize_err}. Attempting UTC localization.")
             try:
                 df_ercot_raw = df_ercot_raw.tz_localize('UTC')
             except Exception as utc_localize_err:
                  st.error(f"Fallback localization to UTC also failed: {utc_localize_err}. Cannot ensure timezone consistency.")
                  # st.stop() # Option: Stop if TZ is critical

    # 3. Check Required Columns
    missing_cols = [col for col in REQUIRED_COLS_ERCOT if col not in df_ercot_raw.columns]
    if missing_cols:
        st.error(f"ERCOT data is missing required columns: {missing_cols}")
        st.info(f"Available columns: {df_ercot_raw.columns.tolist()}")
        st.stop()

except Exception as e:
    st.error(f"Critical error during ERCOT data validation: {e}")
    st.exception(e)
    st.stop()


# --- Sidebar Date Range Selection ---
st.sidebar.header("Date Range Selection")
global_min_date, global_max_date = get_global_date_range(all_iso_data)

if global_min_date is None or global_max_date is None:
    st.sidebar.error("Could not determine a valid global date range from the loaded data. Using fallback.")
    # Provide some default fallbacks if global range fails
    ercot_min_date_safe = df_ercot_raw.index.min().date() if not df_ercot_raw.empty else datetime.date.today() - datetime.timedelta(days=365)
    ercot_max_date_safe = df_ercot_raw.index.max().date() if not df_ercot_raw.empty else datetime.date.today()
    global_min_date = ercot_min_date_safe
    global_max_date = ercot_max_date_safe


# Use ERCOT's specific range if available and valid, otherwise use global
ercot_min_date = df_ercot_raw.index.min().date()
ercot_max_date = df_ercot_raw.index.max().date()

# Sensible defaults: last 30 days within available ERCOT range, clamped by global range
default_end = min(ercot_max_date, global_max_date)
default_start = max(global_min_date, ercot_min_date, default_end - datetime.timedelta(days=30))

# Ensure start is not after end
if default_start > default_end:
    default_start = min(global_min_date, ercot_min_date) # Fallback to earliest available

start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=global_min_date, max_value=global_max_date)
end_date = st.sidebar.date_input("End Date", value=default_end, min_value=start_date, max_value=global_max_date)

# --- Filter ERCOT Data based on Selection ---
df_ercot_filtered = pd.DataFrame() # Initialize empty
try:
    if start_date and end_date:
        tz = df_ercot_raw.index.tz
        # Create timezone-aware timestamps for filtering
        start_dt = pd.Timestamp(datetime.datetime.combine(start_date, datetime.time.min), tz=tz)
        # End date is inclusive, so filter up to the start of the next day
        end_dt = pd.Timestamp(datetime.datetime.combine(end_date + datetime.timedelta(days=1), datetime.time.min), tz=tz)

        mask = (df_ercot_raw.index >= start_dt) & (df_ercot_raw.index < end_dt)
        # Use .copy() to avoid SettingWithCopyWarning later
        df_ercot_filtered = df_ercot_raw.loc[mask].copy()

        if df_ercot_filtered.empty:
            st.warning(f"No ERCOT data available between {start_date} and {end_date}. Adjust the date range or check raw data.")
            # Don't stop here, let the rest of the app handle the empty DataFrame gracefully

except Exception as e:
     st.error(f"Error during date filtering: {e}")
     st.exception(e)
     st.stop() # Stop if filtering fails critically

# --- Standardize Index (Hourly) ---
df_ercot = pd.DataFrame() # Initialize empty
if not df_ercot_filtered.empty:
    try:
        df_ercot = ensure_uniform_hourly_index(df_ercot_filtered, TARGET_ISO_KEY)
        if df_ercot.empty:
            st.warning("ERCOT data became empty after ensuring uniform hourly index. Check for large gaps or resampling issues.")
    except Exception as e:
        st.error(f"Error during index standardization for ERCOT: {e}")
        st.exception(e)
        # df_ercot remains empty
else:
    # If df_ercot_filtered was already empty, df_ercot should also be empty
    st.info("Skipping index standardization as filtered ERCOT data is empty.")

# --- Add Price Data (if available for ERCOT) ---
# Perform this *after* standardization to ensure index alignment
if not df_ercot.empty:
    try:
        # Pass a copy to potentially avoid modifying the original if the function does it in-place
        df_ercot = add_price_data_to_existing_df(df_ercot.copy(), TARGET_ISO_KEY, target_column=PRICE_DIFF_COL)
        if PRICE_DIFF_COL not in df_ercot.columns:
            st.info(f"Could not find or calculate price difference data ('{PRICE_DIFF_COL}') for ERCOT in the selected range.")
        else:
            st.success(f"Successfully added/found '{PRICE_DIFF_COL}' column.")
    except KeyError as ke:
        st.warning(f"Could not calculate '{PRICE_DIFF_COL}'. Missing required input column(s): {ke}. Price analysis will be limited.")
    except FileNotFoundError as fnf_err:
        st.warning(f"Price data file not found: {fnf_err}. Price analysis will be limited.")
    except Exception as e:
        st.error(f"An unexpected error occurred while adding price data for ERCOT: {e}")
        st.exception(e)

st.markdown("---") # Add a visual separator

# --- Main Application Area with Tabs ---
tab1, tab2 = st.tabs(["ERCOT Detailed Analysis", "ISO Comparison (Optional)"])

# =============================
# Tab 1: ERCOT Detailed Analysis
# =============================
with tab1:
    st.header(f"Detailed Analysis for ERCOT ({start_date} to {end_date})")

    if df_ercot.empty:
        st.warning("No ERCOT data available for analysis in the selected range after processing. Please adjust the date range or check data sources.")
        st.stop() # Stop this tab if no data
    else:
        # --- Calculate and Display Metrics ---
        st.subheader("Performance Metrics")
        try:
            # Ensure required columns exist before calling metrics calculation
            if all(col in df_ercot.columns for col in REQUIRED_COLS_ERCOT):
                ercot_metrics = compute_iso_metrics(df_ercot)

                if ercot_metrics and isinstance(ercot_metrics, dict):
                    # Create DataFrame with raw values first
                    metrics_df = pd.DataFrame([ercot_metrics]).T.rename(columns={0: "Raw Value"})
                    # Create the formatted display column
                    metrics_df['Value'] = metrics_df.apply(lambda row: format_metric_value(row['Raw Value'], row.name), axis=1)

                    # Display only the formatted 'Value' column
                    st.dataframe(metrics_df[['Value']], use_container_width=True)
                elif not ercot_metrics:
                     st.warning("Metrics calculation returned no results (e.g., empty dict or None).")
                else:
                     st.warning(f"Metrics calculation returned unexpected type: {type(ercot_metrics)}. Expected a dictionary.")
            else:
                missing_metric_cols = [col for col in REQUIRED_COLS_ERCOT if col not in df_ercot.columns]
                st.warning(f"Skipping metrics calculation because required columns are missing: {missing_metric_cols}")

        except TypeError as te:
             st.error(f"TypeError during metrics calculation: {te}")
             st.error("This often means `compute_iso_metrics` received incompatible data or arguments.")
             st.info(f"Data columns available for metrics: {df_ercot.columns.tolist()}")
             st.exception(te) # Show traceback for debugging
        except KeyError as ke:
             st.error(f"KeyError during metrics calculation: Missing expected column '{ke}'.")
             st.info(f"Data columns available for metrics: {df_ercot.columns.tolist()}")
             st.exception(ke)
        except Exception as e:
            st.error("An unexpected error occurred during metrics calculation:")
            st.exception(e)

        # --- Plot 1: Load vs Forecast and Forecast Error ---
        st.subheader("Load vs. Forecast & Error Time Series")
        st.markdown("""
        **Narrative:** This fundamental plot is the starting point for monitoring forecast performance.
        *   **Top Panel:** Shows the actual load (green line and fill) against the forecasted load (blue line). Visually tracking how closely the forecast follows the actual demand is crucial for identifying immediate discrepancies. The green filled area emphasizes the magnitude of the actual load we need to serve.
        *   **Bottom Panel:** Breaks down the forecast error (Actual - Forecast). Red filled areas highlight **Over-forecasting** (predicting too much load), while blue filled areas show **Under-forecasting** (predicting too little load). Consistently large errors, or errors predominantly of one type, signal potential issues with the forecast model or input assumptions. The grey dotted line represents the 7-day moving average error, smoothing out noise to reveal underlying trends or biases.
        **Why Monitor?** Accurate forecasts minimize costs associated with procuring too much or too little power and ensure grid reliability.
        """)
        try:
            # Check if required columns exist
            if ACTUAL_LOAD_COL in df_ercot.columns and FORECAST_LOAD_COL in df_ercot.columns and FORECAST_ERROR_COL in df_ercot.columns:
                fig1 = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=(
                        "Actual Load (Green) vs. Forecast Load (Blue)",
                        "Forecast Error (Red = Over-forecast, Blue = Under-forecast)"
                    )
                )

                # --- Subplot 1: Actual Load vs. Forecast ---
                # Calculate dynamic range with buffer, handle potential NaNs
                min_load = min(df_ercot[ACTUAL_LOAD_COL].min(skipna=True), df_ercot[FORECAST_LOAD_COL].min(skipna=True))
                max_load = max(df_ercot[ACTUAL_LOAD_COL].max(skipna=True), df_ercot[FORECAST_LOAD_COL].max(skipna=True))
                load_buffer = (max_load - min_load) * 0.05 if pd.notna(min_load) and pd.notna(max_load) and max_load > min_load else 100 # 5% buffer or fixed

                fig1.add_trace(go.Scatter(
                    x=df_ercot.index, y=df_ercot[ACTUAL_LOAD_COL], name='Actual Load',
                    mode='lines',
                    line=dict(color='rgba(0,100,80,1)', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,100,80,0.2)',
                    connectgaps=True # Connect across missing data points
                ), row=1, col=1)

                fig1.add_trace(go.Scatter(
                    x=df_ercot.index, y=df_ercot[FORECAST_LOAD_COL], name='Forecast Load',
                    mode='lines',
                    line=dict(color='rgba(0,0,255,0.8)', width=1.5),
                    connectgaps=True # Connect across missing data points
                ), row=1, col=1)

                # Set y-axis range only if min/max are valid
                if pd.notna(min_load) and pd.notna(max_load):
                    fig1.update_yaxes(title_text="Load (MW)", range=[min_load - load_buffer, max_load + load_buffer], row=1, col=1)
                else:
                    fig1.update_yaxes(title_text="Load (MW)", row=1, col=1)


                # --- Subplot 2: Forecast Error ---
                # Use clip and replace 0 with NaN for separate line/fill traces
                # Fill traces should NOT connect gaps where error is zero
                df_ercot['Overforecast_Fill'] = df_ercot[FORECAST_ERROR_COL].clip(lower=0)
                df_ercot['Underforecast_Fill'] = df_ercot[FORECAST_ERROR_COL].clip(upper=0)
                df_ercot['Overforecast_Line'] = df_ercot['Overforecast_Fill'].replace(0, np.nan)
                df_ercot['Underforecast_Line'] = df_ercot['Underforecast_Fill'].replace(0, np.nan)

                fig1.add_trace(go.Scatter(
                    x=df_ercot.index, y=df_ercot['Overforecast_Fill'],
                    name='Over-forecast Error',
                    mode='lines',
                    line=dict(width=0), # Hide line for fill trace
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.4)',
                    connectgaps=False # Don't connect across zero error
                ), row=2, col=1)
                fig1.add_trace(go.Scatter(
                    x=df_ercot.index, y=df_ercot['Overforecast_Line'],
                    name='Over-forecast Line',
                    showlegend=False,
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.6)', width=1),
                    connectgaps=False # Don't connect across zero error
                ), row=2, col=1)

                fig1.add_trace(go.Scatter(
                    x=df_ercot.index, y=df_ercot['Underforecast_Fill'],
                    name='Under-forecast Error',
                    mode='lines',
                    line=dict(width=0), # Hide line for fill trace
                    fill='tozeroy',
                    fillcolor='rgba(0, 0, 255, 0.4)',
                    connectgaps=False # Don't connect across zero error
                ), row=2, col=1)
                fig1.add_trace(go.Scatter(
                    x=df_ercot.index, y=df_ercot['Underforecast_Line'],
                    name='Under-forecast Line',
                    showlegend=False,
                    mode='lines',
                    line=dict(color='rgba(0,0,255,0.6)', width=1),
                    connectgaps=False # Don't connect across zero error
                ), row=2, col=1)

                fig1.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=2, col=1)

                # Calculate and plot 7-day moving average if enough data
                window_7d = 24 * 7
                if len(df_ercot) >= window_7d:
                    # Calculate MA if not already present or needs recalculation
                    df_ercot['Error_MA_7D'] = df_ercot[FORECAST_ERROR_COL].rolling(window=window_7d, min_periods=24).mean()

                    fig1.add_trace(go.Scatter(
                        x=df_ercot.index, y=df_ercot['Error_MA_7D'], name='7-Day Avg Error',
                        mode='lines', line=dict(color='rgba(80,80,80,0.9)', width=1.5, dash='dot'),
                        connectgaps=True # MA should connect gaps
                    ), row=2, col=1)
                else:
                    st.info("Not enough data points for a 7-day moving average error calculation.")


                fig1.update_yaxes(title_text="Forecast Error (MW)", row=2, col=1)
                fig1.update_xaxes(title_text="Date / Time", row=2, col=1)
                fig1.update_layout(height=700, hovermode='x unified',
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                title="ERCOT Load: Actual vs. Forecast and Error Analysis")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning("Skipping Load/Forecast plot: Missing one or more required columns "
                           f"({ACTUAL_LOAD_COL}, {FORECAST_LOAD_COL}, {FORECAST_ERROR_COL}).")
        except Exception as e:
            st.error(f"Error generating Load vs. Forecast plot: {e}")
            st.exception(e)

        # --- Plot 2: RT vs DA Price Difference Time Series (Two Panel) ---
        st.subheader("Real-Time vs. Day-Ahead Price Difference Over Time")
        st.markdown(f"""
        **Narrative:** These two panels show the hourly RT–DA price difference (`{PRICE_DIFF_COL}`, typically $/MWh):
        1. **Overview (full range):** Useful for spotting extreme price spikes or overall trends.
        2. **Zoomed (±100):** Focuses on the typical range of price differences, making smaller variations easier to see.

        * **Blue lines:** Hourly RT–DA price difference.
        * **Orange lines:** 7‑day moving average of the price difference (if enough data).
        * **Grey dashed line:** Zero difference reference line.
        **Why Monitor?** Large or persistent differences between RT and DA prices can indicate forecast inaccuracies (impacting load/generation balance), transmission congestion, or generator bidding behavior. Positive values (RT > DA) often occur during unexpected high demand or low supply.
        """)
        try:
            if PRICE_DIFF_COL in df_ercot.columns and not df_ercot[PRICE_DIFF_COL].isnull().all():
                # Prepare the series, drop NaNs specific to this column
                df_price_ts = df_ercot[[PRICE_DIFF_COL]].dropna().copy()

                if not df_price_ts.empty:
                    # Calculate 7‑day MA if enough data
                    ma_window = 24 * 7
                    if len(df_price_ts) >= ma_window:
                        df_price_ts['Price_Diff_MA'] = (
                            df_price_ts[PRICE_DIFF_COL]
                            .rolling(window=ma_window, min_periods=24)
                            .mean()
                        )
                    else:
                        df_price_ts['Price_Diff_MA'] = np.nan
                        st.info("Not enough data points for a 7‑day moving average on price difference.")

                    # Build the two‑panel figure
                    fig_price_ts = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.3, 0.7], # Give more space to zoomed view
                        vertical_spacing=0.05,
                        subplot_titles=("Full Range Overview", "Zoomed View (±100 $/MWh)")
                    )

                    # Panel 1: Full range
                    fig_price_ts.add_trace(
                        go.Scatter(
                            x=df_price_ts.index,
                            y=df_price_ts[PRICE_DIFF_COL],
                            mode="lines",
                            name="Hourly Δ (Full Range)",
                            line=dict(color='cornflowerblue'),
                            connectgaps=True # Connect gaps for the primary line
                        ),
                        row=1, col=1
                    )
                    # Add MA to Panel 1 if exists
                    if 'Price_Diff_MA' in df_price_ts and not df_price_ts['Price_Diff_MA'].isnull().all():
                        fig_price_ts.add_trace(
                            go.Scatter(
                                x=df_price_ts.index,
                                y=df_price_ts['Price_Diff_MA'],
                                mode="lines",
                                name="7‑day MA (Full Range)",
                                line=dict(color='darkorange', width=2, dash="dash"),
                                connectgaps=True
                            ),
                            row=1, col=1
                        )
                    fig_price_ts.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=1, col=1)
                    fig_price_ts.update_yaxes(title_text="Δ Price ($/MWh)", row=1, col=1)

                    # Panel 2: Zoomed ±100
                    # Use the original data for the line, let y-axis range clip it visually
                    fig_price_ts.add_trace(
                        go.Scatter(
                            x=df_price_ts.index,
                            y=df_price_ts[PRICE_DIFF_COL],
                            mode="lines",
                            name="Hourly Δ (Zoomed)",
                            line=dict(color='cornflowerblue'),
                            connectgaps=True
                        ),
                        row=2, col=1
                    )
                    # Add MA to Panel 2 if exists
                    if 'Price_Diff_MA' in df_price_ts and not df_price_ts['Price_Diff_MA'].isnull().all():
                         fig_price_ts.add_trace(
                            go.Scatter(
                                x=df_price_ts.index,
                                y=df_price_ts['Price_Diff_MA'],
                                mode="lines",
                                name="7‑day MA (Zoomed)",
                                line=dict(color='darkorange', width=2, dash="dash"),
                                connectgaps=True
                            ),
                            row=2, col=1
                        )
                    fig_price_ts.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=2, col=1)
                    fig_price_ts.update_yaxes(title_text="Δ Price ($/MWh)", range=[-100, 100], row=2, col=1)
                    fig_price_ts.update_xaxes(title_text="Date / Time", row=2, col=1)

                    # Global layout tweaks
                    fig_price_ts.update_layout(
                        height=700,
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        #showlegend=False, # Keep legend for clarity between panels if needed
                        title_text="RT–DA Price Difference: Full vs. Zoomed View"
                    )

                    st.plotly_chart(fig_price_ts, use_container_width=True)
                    st.caption(f"Price Difference = {PRICE_DIFF_COL} (Real-Time - Day-Ahead assumed).")
                else:
                    st.info(f"No valid price difference data ('{PRICE_DIFF_COL}') available in the selected range to plot.")
            else:
                st.info(f"Price difference column '{PRICE_DIFF_COL}' not found or is empty. Skipping price difference time series plot.")
        except Exception as e:
            st.error(f"Error generating two-panel price difference plot: {e}")
            st.exception(e)


        # --- Plot 3: Price Diff + Forecast Error Context + Heatmap ---
        st.subheader("Clipped Price Difference & Forecast Error Context")
        st.markdown(f"""
        **Narrative:** This section explores the relationship between forecast errors and price differences.
        *   **Top Plot (Time Series):** Shows the RT–DA price difference clipped to ±100 $/MWh (blue line) to focus on typical variations. The background is shaded based on the *sign* of the forecast error:
            *   **Red Shading:** Indicates periods of **Over-forecasting** (Forecast Error > 0).
            *   **Blue Shading:** Indicates periods of **Under-forecasting** (Forecast Error < 0).
            This helps visually correlate forecast misses with price outcomes. For example, do periods of under-forecasting (blue shade) tend to coincide with high price differences (blue line spikes)?
        *   **Bottom Plot (Heatmap):** Shows the 2D density distribution of **Forecast Error (MW)** vs. **RT–DA Price Difference ($/MWh)**. Each point represents an hour. Colors indicate the density of observations: warmer colors (e.g., yellow) show combinations that occur frequently, while cooler colors (e.g., purple/blue) show less frequent combinations. The center of the densest region reveals the most typical relationship between error and price difference. Outliers will appear in low-density areas.
        **Why Monitor?** Understanding this relationship is key to valuing forecast accuracy. Large errors that consistently coincide with significant price differences have a higher financial impact.
        """)
        try:
            # Check required columns first
            required_cols_context = [PRICE_DIFF_COL, FORECAST_ERROR_COL]
            if all(col in df_ercot.columns for col in required_cols_context):
                # Prepare data, drop NaNs for rows where *both* columns are needed
                df_price_context = df_ercot[required_cols_context].dropna().copy()

                if not df_price_context.empty:
                    # Clipped price difference
                    df_price_context['Clipped_Price_Diff'] = df_price_context[PRICE_DIFF_COL].clip(-100, 100)

                    # Shading heights based on forecast error sign (use small tolerance for float comparison)
                    tolerance = 1e-6
                    df_price_context['Overforecast_BG'] = np.where(df_price_context[FORECAST_ERROR_COL] > tolerance, 100, 0)
                    df_price_context['Underforecast_BG'] = np.where(df_price_context[FORECAST_ERROR_COL] < -tolerance, -100, 0)


                    # --- Plot 3a: Clipped time series with BG shading ---
                    fig_ts_context = go.Figure()

                    # Background Over-forecast shading (Red)
                    fig_ts_context.add_trace(go.Bar(
                        x=df_price_context.index,
                        y=df_price_context['Overforecast_BG'],
                        base=0,
                        marker_color='red',
                        opacity=0.15,
                        width=3600000 * 1.05,  # ms for 1 hour + small overlap
                        name='Over-forecast Period',
                        showlegend=True # Optional: Add legend item for shading
                    ))
                    # Background Under-forecast shading (Blue)
                    fig_ts_context.add_trace(go.Bar(
                        x=df_price_context.index,
                        y=df_price_context['Underforecast_BG'],
                        base=0,
                        marker_color='blue',
                        opacity=0.15,
                        width=3600000 * 1.05, # ms for 1 hour + small overlap
                        name='Under-forecast Period',
                        showlegend=True # Optional: Add legend item for shading
                    ))

                    # Clipped price diff line (plot last to be on top)
                    fig_ts_context.add_trace(go.Scatter(
                        x=df_price_context.index,
                        y=df_price_context['Clipped_Price_Diff'],
                        mode='lines',
                        name='RT-DA Δ (Clipped ±100)',
                        line=dict(width=1.5, color='rgba(0,100,150,0.9)'), # Teal/Blue line
                        connectgaps=True
                    ))

                    # Zero line
                    fig_ts_context.add_hline(y=0, line_dash='dash', line_color='grey')

                    fig_ts_context.update_layout(
                        title="RT–DA Price Difference (Clipped ±100) with Forecast Error Shading",
                        xaxis_title="Date / Time",
                        yaxis_title="Price Δ ($/MWh)",
                        yaxis_range=[-105, 105], # Slightly larger range for clipped view
                        height=450,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        bargap=0 # Ensure bars touch for continuous shading
                    )
                    st.plotly_chart(fig_ts_context, use_container_width=True)

                    # --- Plot 3b: 2D density heatmap of Error vs. Price Difference ---
                    st.subheader("Forecast Error vs. RT–DA Price Difference Density")
                    st.markdown("Each point represents an hour; density contours show where most observations lie. Helps identify typical relationships and outliers.")

                    fig_heat = go.Figure(go.Histogram2dContour(
                        x=df_price_context[FORECAST_ERROR_COL],
                        y=df_price_context[PRICE_DIFF_COL],
                        colorscale="Viridis", # Common colorscale for density
                        contours=dict(
                            coloring='heatmap',
                            showlabels=False, # Labels can clutter the plot
                            showlines=False
                        ),
                        ncontours=20, # Adjust number of contours for detail
                        name='Density',
                        hoverinfo='z', # Show density value on hover
                        colorbar=dict(title='Density') # Add color bar label
                    ))

                    # Add faint scatter points underneath for context (optional, can be slow for large data)
                    # fig_heat.add_trace(go.Scatter(
                    #    x=df_price_context[FORECAST_ERROR_COL],
                    #    y=df_price_context[PRICE_DIFF_COL],
                    #    mode='markers',
                    #    marker=dict(color='black', size=2, opacity=0.1),
                    #    hoverinfo='none',
                    #    showlegend=False
                    # ))


                    fig_heat.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
                    fig_heat.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")

                    fig_heat.update_layout(
                        title="2D Density: Forecast Error vs. RT–DA Price Difference",
                        xaxis_title=f"Forecast Error (MW) [{FORECAST_ERROR_COL}]",
                        yaxis_title=f"RT–DA Price Difference ($/MWh) [{PRICE_DIFF_COL}]",
                        height=500,
                        hovermode='closest',
                        xaxis=dict(zeroline=False), # Keep axis lines, not zero lines if using vline/hline
                        yaxis=dict(zeroline=False)
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                else:
                    st.info("No overlapping valid data points found for Price Difference and Forecast Error. Skipping context plots.")
            else:
                missing_context_cols = [col for col in required_cols_context if col not in df_ercot.columns]
                st.info(f"Skipping Price Difference/Error context plots: Missing required columns: {missing_context_cols}.")

        except Exception as e:
            st.error(f"Error generating Price Difference/Error context plots: {e}")
            st.exception(e)


        # --- Plot 4: MAPE vs Time ---
        st.subheader("Absolute Percentage Error (APE) Over Time")
        st.markdown("""
        **Narrative:** While MW error shows the magnitude, Absolute Percentage Error (APE) contextualizes the error relative to the actual load. A 500 MW error is significant during low load periods but less so during peak demand. This plot shows the 7-day moving average of APE (%).
        **Why Monitor?** Tracking APE helps assess forecast quality consistently across different load levels. A stable, low APE indicates a robust forecast. Spikes often correlate with unexpected weather events or load changes the model didn't capture. Setting performance thresholds (like the red dashed line) allows for quick identification of periods where forecast accuracy degraded significantly.
        """)
        try:
            # Required columns: FORECAST_ERROR_COL, ACTUAL_LOAD_COL
            if FORECAST_ERROR_COL in df_ercot.columns and ACTUAL_LOAD_COL in df_ercot.columns:
                # Create a temporary series for calculation, handling potential division by zero/small numbers
                # Use a small positive threshold for the denominator
                load_threshold = 1.0 # MW - Avoid division by zero or near-zero load
                actual_load_safe = df_ercot[ACTUAL_LOAD_COL].where(df_ercot[ACTUAL_LOAD_COL] >= load_threshold, np.nan)

                # Calculate APE only where actual load is safe
                df_ercot['APE'] = np.abs(df_ercot[FORECAST_ERROR_COL]) / actual_load_safe * 100
                # APE might still have NaNs if error or load was NaN

                # Calculate 7-day Moving Average of APE if enough valid APE data exists
                window_7d = 24 * 7
                if len(df_ercot['APE'].dropna()) >= window_7d:
                     # Ensure APE_MA_7D calculation happens if needed
                     df_ercot['APE_MA_7D'] = df_ercot['APE'].rolling(window=window_7d, min_periods=24).mean()

                     if 'APE_MA_7D' in df_ercot.columns and not df_ercot['APE_MA_7D'].isnull().all():
                         fig_ape = go.Figure()
                         fig_ape.add_trace(go.Scatter(
                            x=df_ercot.index, y=df_ercot['APE_MA_7D'], mode='lines',
                            name='7-Day Moving Avg APE (%)', line=dict(color='steelblue', width=2),
                            connectgaps=True # MA should connect gaps
                         ))

                         threshold_mape = 4.0 # Example threshold
                         fig_ape.add_hline(y=threshold_mape, line_width=1.5, line_dash="dash", line_color="red",
                                   annotation_text=f"Threshold: {threshold_mape}%", annotation_position="top right")

                         fig_ape.update_layout(
                            title=f"7-Day Moving Average Absolute Percentage Error (APE) vs. Time (Threshold = {threshold_mape}%)",
                            xaxis_title="Date / Time", yaxis_title="APE (%)", height=450, hovermode='x unified',
                            yaxis=dict(range=[0, max(df_ercot['APE_MA_7D'].max()*1.1, threshold_mape*1.5)]) # Dynamic y-axis range
                         )
                         st.plotly_chart(fig_ape, use_container_width=True)
                     else:
                         st.warning("Could not plot 7-Day Moving Average APE (result might be all NaN).")
                else:
                    st.info("Not enough valid data points to calculate a 7-Day Moving Average APE.")
            else:
                 st.warning("Skipping APE plot: Missing required columns "
                           f"({FORECAST_ERROR_COL}, {ACTUAL_LOAD_COL}).")

        except Exception as e:
             st.error(f"Error generating APE plot: {e}")
             st.exception(e)

        # --- Plot 5: Error Distribution Histogram ---
        st.subheader("Distribution of Forecast Errors")
        st.markdown("""
        **Narrative:** How often is the forecast 'off', and by how much? This histogram shows the frequency of different forecast error sizes (in MW).
        *   **Shape:** A tall, narrow peak centered close to zero indicates a generally accurate and unbiased forecast.
        *   **Spread:** A wider spread means errors are frequently large, suggesting inconsistency.
        *   **Skewness:** If the histogram leans heavily to one side, it reveals a systematic bias (e.g., consistently under-forecasting if skewed left, or over-forecasting if skewed right, given Error = Actual - Forecast).
        **Why Monitor?** Understanding the distribution helps quantify forecast reliability and identify systematic biases that need correction in the model. It answers: "Are large errors rare exceptions or common occurrences?"
        """)
        try:
            if FORECAST_ERROR_COL in df_ercot.columns and not df_ercot[FORECAST_ERROR_COL].isnull().all():
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=df_ercot[FORECAST_ERROR_COL].dropna(), # Drop NaNs before histogram
                    name='Error Frequency',
                    marker_color='rgb(100,100,100)', # Neutral grey color
                    # nbinsx=50 # Optional: Control number of bins
                ))

                mean_err = df_ercot[FORECAST_ERROR_COL].mean()
                median_err = df_ercot[FORECAST_ERROR_COL].median()

                fig_hist.add_vline(x=0, line_width=2, line_dash="dash", line_color="grey",
                                annotation_text="Zero Error", annotation_position="top left")
                if pd.notna(mean_err):
                    fig_hist.add_vline(x=mean_err, line_width=2, line_dash="dot", line_color="orange",
                                    annotation_text=f"Mean Err: {mean_err:.1f} MW", annotation_position="top right")
                # Optionally add median line as well
                # if pd.notna(median_err):
                #     fig_hist.add_vline(x=median_err, line_width=2, line_dash="dot", line_color="skyblue",
                #                     annotation_text=f"Median Err: {median_err:.1f} MW", annotation_position="bottom right")


                fig_hist.update_layout(
                    title="Frequency Distribution of Forecast Error (MW)",
                    xaxis_title=f"Forecast Error (MW) [{FORECAST_ERROR_COL}]",
                    yaxis_title="Frequency (Count of Hours)",
                    height=450,
                    bargap=0.1
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning(f"Forecast Error column '{FORECAST_ERROR_COL}' not found or is empty. Skipping histogram.")

        except Exception as e:
            st.error(f"Error generating forecast error histogram: {e}")
            st.exception(e)


        # --- Plot 6: Error vs. Actual Load Scatter ---
        st.subheader("Forecast Error vs. Actual Load Level")
        st.markdown("""
        **Narrative:** Does forecast accuracy change depending on the system load? This scatter plot investigates the relationship between the actual load level (horizontal axis) and the corresponding forecast error (vertical axis).
        *   **Ideal Pattern:** Points should be randomly scattered around the zero error line (horizontal grey line) across all load levels, forming a roughly horizontal band.
        *   **Problematic Patterns:**
            *   **Fanning Out (Heteroscedasticity):** Errors increasing in magnitude (wider spread) at higher or lower loads suggests the model's performance varies with load level.
            *   **Curved Trend:** A non-linear pattern might indicate the model misrepresents load dynamics at certain levels (e.g., under-forecasting peaks consistently).
            *   **Slanted Trend:** A diagonal pattern indicates a load-dependent bias (e.g., error increases as load increases).
        **Why Monitor?** This helps diagnose *when* the forecast model is weakest (e.g., during peaks, ramps, or low load periods), guiding targeted improvements.
        """)
        try:
            required_scatter_cols = [FORECAST_ERROR_COL, ACTUAL_LOAD_COL]
            if all(col in df_ercot.columns for col in required_scatter_cols):
                # Drop rows where *either* column is NaN for a fair comparison
                df_scatter_valid = df_ercot[required_scatter_cols].dropna().copy()

                if not df_scatter_valid.empty:
                    # --- Optional: Outlier Filtering (Visual Clarity) ---
                    # Filter based on Error standard deviations
                    err_mean = df_scatter_valid[FORECAST_ERROR_COL].mean()
                    err_std = df_scatter_valid[FORECAST_ERROR_COL].std()
                    scatter_caption = ""

                    # Apply filtering only if std is valid and positive
                    if pd.notna(err_std) and err_std > 0:
                        sigma_threshold = 3.0
                        lower_bound = err_mean - sigma_threshold * err_std
                        upper_bound = err_mean + sigma_threshold * err_std

                        df_scatter_plot = df_scatter_valid[
                            (df_scatter_valid[FORECAST_ERROR_COL] >= lower_bound) &
                            (df_scatter_valid[FORECAST_ERROR_COL] <= upper_bound)
                        ].copy()

                        num_filtered = len(df_scatter_valid) - len(df_scatter_plot)
                        if num_filtered > 0:
                             scatter_caption = (f"Note: {num_filtered} points with forecast error outside "
                                               f"±{sigma_threshold} standard deviations from the mean are excluded "
                                               f"from this plot for visual clarity.")
                        else:
                             scatter_caption = "Note: All data points are within ±3 standard deviations of the mean error."

                    else:
                        df_scatter_plot = df_scatter_valid.copy() # Plot all data if std is invalid
                        scatter_caption = "Note: Standard deviation of error could not be calculated; showing all points."


                    if not df_scatter_plot.empty:
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scattergl( # Use Scattergl for potentially better performance with many points
                            x=df_scatter_plot[ACTUAL_LOAD_COL],
                            y=df_scatter_plot[FORECAST_ERROR_COL],
                            mode='markers',
                            name='Hourly Error',
                            marker=dict(
                                color='rgba(0, 128, 128, 0.5)', # Teal color with transparency
                                size=5,
                                line=dict(width=0) # No marker outline
                            ),
                            # Create hover text using the index from df_scatter_plot
                            hovertext=[f"Time: {idx.strftime('%Y-%m-%d %H:%M')}<br>Actual: {act:.0f} MW<br>Error: {err:.0f} MW"
                                    for idx, act, err in zip(df_scatter_plot.index, df_scatter_plot[ACTUAL_LOAD_COL], df_scatter_plot[FORECAST_ERROR_COL])],
                            hoverinfo='text'
                        ))

                        fig_scatter.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")

                        fig_scatter.update_layout(
                            title="Forecast Error vs. Actual Load",
                            xaxis_title=f"Actual Load (MW) [{ACTUAL_LOAD_COL}]",
                            yaxis_title=f"Forecast Error (MW) [{FORECAST_ERROR_COL}]",
                            height=500,
                            hovermode='closest' # Show hover for individual points
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        if scatter_caption:
                            st.caption(scatter_caption)
                    else:
                        st.warning("No data remaining for scatter plot after applying filters (if any).")
                else:
                    st.warning("No overlapping valid data points found for Forecast Error and Actual Load. Skipping scatter plot.")
            else:
                missing_scatter_cols = [col for col in required_scatter_cols if col not in df_ercot.columns]
                st.warning(f"Skipping Error vs. Load scatter plot: Missing required columns: {missing_scatter_cols}.")

        except Exception as e:
            st.error(f"Error generating error vs. load scatter plot: {e}")
            st.exception(e)


        # --- Plot 7: Cumulative Forecast Bias ---
        st.subheader("Cumulative Forecast Bias Over Time")
        st.markdown("""
        **Narrative:** Is our forecast consistently predicting too high or too low over the selected period? This plot shows the running total (cumulative sum) of the forecast error (MW).
        *   **Upward Trend:** A line steadily increasing indicates a persistent **Over-forecasting** bias (accumulating positive error, assuming Error = Actual - Forecast).
        *   **Downward Trend:** A line steadily decreasing indicates a persistent **Under-forecasting** bias (accumulating negative error).
        *   **Fluctuating Around Zero:** A line staying relatively close to zero suggests the forecast is **unbiased** over the long run, even if individual hours have errors. Steeper slopes indicate periods of larger bias.
        **Why Monitor?** Uncorrected long-term bias can lead to systematic inefficiencies in resource scheduling, capacity planning, and financial settlements. Detecting the trend early allows for model recalibration.
        """)
        try:
            if FORECAST_ERROR_COL in df_ercot.columns and not df_ercot[FORECAST_ERROR_COL].isnull().all():
                # Calculate cumulative error safely, filling potential NaNs in error with 0 before summing
                df_ercot['Cumulative Error'] = df_ercot[FORECAST_ERROR_COL].fillna(0).cumsum()

                fig_cumul = go.Figure()
                fig_cumul.add_trace(go.Scatter(
                    x=df_ercot.index,
                    y=df_ercot['Cumulative Error'],
                    mode='lines',
                    name='Cumulative Error',
                    line=dict(color='purple', width=2),
                    connectgaps=True # Should connect gaps here as it's a cumulative sum
                ))

                fig_cumul.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")

                fig_cumul.update_layout(
                    title="Cumulative Forecast Error (Bias Trend)",
                    xaxis_title="Date / Time",
                    yaxis_title="Cumulative Error (MW)",
                    height=450,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_cumul, use_container_width=True)
            else:
                st.warning(f"Forecast Error column '{FORECAST_ERROR_COL}' not found or is empty. Skipping cumulative bias plot.")

        except Exception as e:
            st.error(f"Error generating cumulative error plot: {e}")
            st.exception(e)


        # --- Hourly Pattern Plots ---
        st.subheader("Hourly Patterns Analysis")
        st.markdown("""
        **Narrative:** Do forecast errors exhibit specific patterns depending on the hour of the day? These plots aggregate error data by hour.
        *   **Top Plot (Average Magnitude):** Shows the average size of over-forecast errors (Red) and under-forecast errors (Blue) for each hour. This helps identify hours where the *magnitude* of misses is typically larger.
        *   **Bottom Plot (Frequency Count):** Shows the total number of times over-forecasting (Red) and under-forecasting (Blue) occurred for each hour. This identifies hours where the forecast is more *likely* to be wrong in a particular direction.
        *   **Day Type Filter:** Use the radio buttons to focus the analysis on Weekdays, Weekends, or All Days, as patterns can differ significantly.
        **Why Monitor?** Identifying hourly biases or weaknesses (e.g., consistently under-forecasting morning ramps, over-forecasting overnight lows) provides actionable insights for tuning the forecast model's diurnal profile adjustments.
        """)
        try:
            if not df_ercot.empty and FORECAST_ERROR_COL in df_ercot.columns:
                # Add 'Hour' column if not present
                if 'Hour' not in df_ercot.columns:
                     df_ercot['Hour'] = df_ercot.index.hour
                # Add 'DayOfWeek' column if not present
                if 'DayOfWeek' not in df_ercot.columns:
                    df_ercot['DayOfWeek'] = df_ercot.index.dayofweek # Monday=0, Sunday=6

                day_filter = st.radio(
                    "Select Day Type for Hourly Aggregations:",
                    options=["All Days", "Weekdays", "Weekends"],
                    index=0, horizontal=True, key="hourly_day_filter"
                )

                df_hourly_filtered = pd.DataFrame() # Initialize
                filter_desc = ""
                if day_filter == "Weekdays":
                    # Monday=0, Tuesday=1, ..., Friday=4
                    df_hourly_filtered = df_ercot[df_ercot['DayOfWeek'] < 5].copy()
                    filter_desc = " (Weekdays Only)"
                elif day_filter == "Weekends":
                    # Saturday=5, Sunday=6
                    df_hourly_filtered = df_ercot[df_ercot['DayOfWeek'] >= 5].copy()
                    filter_desc = " (Weekends Only)"
                else: # All Days
                    df_hourly_filtered = df_ercot.copy()
                    filter_desc = " (All Days)"

                if df_hourly_filtered.empty or df_hourly_filtered[FORECAST_ERROR_COL].isnull().all():
                    st.warning(f"No valid forecast error data available for the selected day type '{day_filter}'. Hourly plots skipped.")
                else:
                    # Proceed with plots using df_hourly_filtered

                    # --- Plot 8: Avg Error Magnitude by Hour ---
                    df_hourly_filtered['Overforecast_MW'] = df_hourly_filtered[FORECAST_ERROR_COL].clip(lower=0)
                    # Calculate absolute value of underforecast for magnitude comparison
                    df_hourly_filtered['Underforecast_MW_abs'] = df_hourly_filtered[FORECAST_ERROR_COL].clip(upper=0).abs()

                    grouped_mw = df_hourly_filtered.groupby('Hour').agg(
                        Avg_Overforecast_MW=('Overforecast_MW', 'mean'),
                        Avg_Underforecast_MW=('Underforecast_MW_abs', 'mean')
                        # Optional: Add count or std dev if needed
                        # Count = ('Forecast Error (MW)', 'count')
                    ).reset_index()

                    # Ensure all hours 0-23 are present, filling missing with 0 or NaN
                    all_hours = pd.DataFrame({'Hour': range(24)})
                    grouped_mw = pd.merge(all_hours, grouped_mw, on='Hour', how='left').fillna(0)


                    fig_hourly_mw = go.Figure()
                    fig_hourly_mw.add_trace(go.Bar(
                        x=grouped_mw['Hour'], y=grouped_mw['Avg_Overforecast_MW'],
                        name='Avg. Over-forecast MW', marker_color='rgba(255,0,0,0.7)'
                    ))
                    fig_hourly_mw.add_trace(go.Bar(
                         x=grouped_mw['Hour'], y=grouped_mw['Avg_Underforecast_MW'],
                         name='Avg. Under-forecast MW', marker_color='rgba(0,0,255,0.7)'
                    ))
                    fig_hourly_mw.update_layout(
                        title=f"Average Forecast Error Magnitude by Hour{filter_desc}",
                        xaxis_title="Hour of Day (0-23)", yaxis_title="Average Error Magnitude (MW)",
                        barmode='group', height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(tickmode='linear', dtick=1) # Ensure all hours are labeled
                    )
                    st.plotly_chart(fig_hourly_mw, use_container_width=True)

                    # --- Plot 9: Count of Errors by Hour ---
                    # Use a small tolerance for zero comparison with floating point numbers
                    tolerance = 1e-6
                    df_hourly_filtered['Is_Overforecast'] = (df_hourly_filtered[FORECAST_ERROR_COL] > tolerance).astype(int)
                    df_hourly_filtered['Is_Underforecast'] = (df_hourly_filtered[FORECAST_ERROR_COL] < -tolerance).astype(int)

                    grouped_count = df_hourly_filtered.groupby('Hour').agg(
                        Count_Overforecast=('Is_Overforecast', 'sum'),
                        Count_Underforecast=('Is_Underforecast', 'sum')
                    ).reset_index()

                    # Ensure all hours 0-23 are present
                    grouped_count = pd.merge(all_hours, grouped_count, on='Hour', how='left').fillna(0)

                    fig_hourly_count = go.Figure()
                    fig_hourly_count.add_trace(go.Bar(
                        x=grouped_count['Hour'], y=grouped_count['Count_Overforecast'],
                        name='Over-forecast Count', marker_color='rgba(255,0,0,0.7)'
                    ))
                    fig_hourly_count.add_trace(go.Bar(
                        x=grouped_count['Hour'], y=grouped_count['Count_Underforecast'],
                        name='Under-forecast Count', marker_color='rgba(0,0,255,0.7)'
                    ))
                    fig_hourly_count.update_layout(
                        title=f"Count of Over/Under Forecast Occurrences by Hour{filter_desc}",
                        xaxis_title="Hour of Day (0-23)", yaxis_title="Number of Occurrences",
                        barmode='group', height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(tickmode='linear', dtick=1) # Ensure all hours are labeled
                    )
                    st.plotly_chart(fig_hourly_count, use_container_width=True)

            elif df_ercot.empty:
                 # This case is already handled by the check at the start of the tab
                 pass
            else: # df_ercot not empty, but missing forecast error col
                st.warning(f"Skipping Hourly Pattern plots: Missing required column '{FORECAST_ERROR_COL}'.")

        except Exception as e:
            st.error(f"Error generating hourly pattern plots: {e}")
            st.exception(e)


        # --- Plot 10: Price Impact Analysis (Binned % Error vs Avg Price Diff) ---
        st.subheader("Price Impact Analysis (Binned % Error)")
        st.markdown(f"""
        **Narrative:** How does the magnitude of the *percentage* forecast error relate to the RT-DA price difference? This plot groups hours into bins based on their Forecast % Error (Error MW / Actual Load MW * 100) and shows the average `'{PRICE_DIFF_COL}'` within each bin.
        *   **X-axis:** Bins representing ranges of forecast percentage error.
        *   **Y-axis:** The average price difference observed for all hours falling into that error bin.
        *   **Day Type:** This analysis currently focuses on **Weekdays Only** (Mon-Fri), as weekday price dynamics often differ from weekends.
        **Why Monitor?** This directly links forecast performance (in relative terms) to potential financial implications. A strong correlation (e.g., large under-forecast % errors consistently coinciding with high positive price differences) highlights the value of improving accuracy, especially during certain error ranges. The "-0.1% to 0.1%" bin represents hours with very accurate forecasts.
        """)
        try:
            required_price_impact_cols = [PRICE_DIFF_COL, ACTUAL_LOAD_COL, FORECAST_ERROR_COL]
            if all(col in df_ercot.columns for col in required_price_impact_cols):
                # Ensure DayOfWeek column exists
                if 'DayOfWeek' not in df_ercot.columns:
                    df_ercot['DayOfWeek'] = df_ercot.index.dayofweek

                # 1. Filter for Weekdays
                weekday_data = df_ercot[df_ercot['DayOfWeek'] < 5].copy()

                if not weekday_data.empty:
                    # 2. Drop rows where any of the essential columns for calculation are NaN
                    valid_price_data = weekday_data[required_price_impact_cols].dropna()

                    if not valid_price_data.empty:
                        # 3. Filter out rows with non-positive actual load for safe % Error calculation
                        load_threshold = 1.0 # Use the same threshold as APE plot
                        valid_price_data = valid_price_data[valid_price_data[ACTUAL_LOAD_COL] >= load_threshold].copy()

                        if not valid_price_data.empty:
                            # 4. Calculate % Error
                            valid_price_data['% Error'] = (
                                valid_price_data[FORECAST_ERROR_COL] / valid_price_data[ACTUAL_LOAD_COL] * 100
                            )

                            # 5. Define bins and labels
                            bins = [-np.inf, -4, -2, -1, -0.1, 0.1, 1, 2, 4, np.inf]
                            bin_labels = ["<-4%", "-4% to -2%", "-2% to -1%", "-1% to -0.1%",
                                        "~0%", # Use a more descriptive label for the near-zero bin
                                        "0.1% to 1%", "1% to 2%", "2% to 4%", ">4%"]
                            # Ensure labels match number of intervals (len(bins)-1)
                            if len(bin_labels) != len(bins) - 1:
                                raise ValueError("Number of bin labels must be one less than the number of bin edges.")


                            valid_price_data['Error Bin'] = pd.cut(
                                valid_price_data['% Error'], bins=bins, labels=bin_labels, right=False, include_lowest=True
                            )

                            # 6. Group by bin and calculate mean price difference and count
                            # Use observed=False for categorical index if needed (depends on pandas version)
                            avg_price_diff_by_bin = valid_price_data.groupby('Error Bin', observed=True)[PRICE_DIFF_COL].agg(['mean', 'count']).reset_index()


                            if not avg_price_diff_by_bin.empty:
                                # 7. Create the bar plot
                                fig_price_impact = go.Figure()
                                fig_price_impact.add_trace(go.Bar(
                                    x=avg_price_diff_by_bin['Error Bin'],
                                    y=avg_price_diff_by_bin['mean'],
                                    name='Avg. Price Difference',
                                    marker_color='seagreen',
                                    # Add count to hover text
                                    hovertext=[f"Count: {count:,}" for count in avg_price_diff_by_bin['count']],
                                    hoverinfo='x+y+text'
                                ))
                                fig_price_impact.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
                                fig_price_impact.update_layout(
                                    title="Average RT-DA Price Difference vs Binned Forecast % Error (Weekdays Only)",
                                    xaxis_title="Forecast % Error Bin (Actual - Forecast) / Actual",
                                    yaxis_title=f"Average Price Difference ({PRICE_DIFF_COL})",
                                    height=450,
                                    xaxis={'categoryorder':'array', 'categoryarray':bin_labels} # Ensure correct bin order
                                )
                                st.plotly_chart(fig_price_impact, use_container_width=True)
                                st.caption(f"Analysis based on {len(valid_price_data):,} valid weekday hours with positive actual load.")
                            else:
                                st.warning("No data available after grouping by error bins for the price impact plot.")
                        else:
                             st.warning("Insufficient valid weekday data after filtering for positive actual load. Skipping price impact plot.")
                    else:
                        st.warning("Insufficient valid weekday data after removing NaNs. Skipping price impact plot.")
                else:
                    st.warning("No weekday data found in the selected date range. Skipping price impact plot.")
            else:
                missing_pi_cols = [col for col in required_price_impact_cols if col not in df_ercot.columns]
                st.info(f"Required columns for price impact analysis not available ({missing_pi_cols}). Skipping plot.")

        except Exception as e:
            st.error(f"Error generating price impact plot: {e}")
            st.exception(e)


        # --- Optional Data Preview ---
        st.subheader("Data Preview")
        try:
            if st.checkbox("Show Processed ERCOT Data Table (First 1000 Rows)", value=False):
                # Define core columns plus any calculated columns that might exist
                potential_cols = [
                    ACTUAL_LOAD_COL, FORECAST_LOAD_COL, FORECAST_ERROR_COL, PRICE_DIFF_COL,
                    'APE', 'Error_MA_7D', 'APE_MA_7D', 'Cumulative Error',
                    'Overforecast_Fill', 'Underforecast_Fill', # From Plot 1 helpers
                    'Hour', 'DayOfWeek' # From Hourly plots helpers
                ]
                # Select only columns that actually exist in the dataframe
                cols_to_show = [col for col in potential_cols if col in df_ercot.columns]

                if cols_to_show:
                    # Display with formatted numbers for better readability
                    st.dataframe(df_ercot[cols_to_show].head(1000).style.format(precision=2, na_rep="NaN"), use_container_width=True)
                else:
                    st.warning("No relevant columns available for data preview.")
        except Exception as e:
             st.error(f"Error displaying data preview table: {e}")
             st.exception(e)

# =============================
# Tab 2: ISO Comparison (Optional)
# =============================
with tab2:
    st.header("ISO Comparison (Placeholder)")
    st.markdown("""
    This section is intended for comparing forecast performance metrics and visualizations across different ISOs/regions available in the loaded data.

    **Future Implementation Ideas:**

    *   **ISO Selection:** Allow users to select multiple ISOs from the `all_iso_data` dictionary.
    *   **Metric Comparison:** Display key performance metrics (like MAPE, Bias, RMSE) side-by-side in a table for the selected ISOs and date range.
    *   **Combined Plots:** Generate comparative plots, such as:
        *   Overlaying 7-day Moving Average APE for selected ISOs.
        *   Side-by-side error distribution histograms.
        *   Comparative hourly error pattern plots.
    *   **Data Harmonization:** Ensure consistent column naming and units (e.g., timezone conversion to UTC) before comparison. Handle cases where data might be missing for certain ISOs or metrics cannot be calculated (e.g., price difference if price columns are absent).

    *(Current implementation focuses only on ERCOT detailed analysis in Tab 1)*
    """)

    # Example: Show available ISO keys
    st.subheader("Available ISOs/Regions in Loaded Data")
    available_keys = list(all_iso_data.keys())
    if available_keys:
        st.write(available_keys)
    else:
        st.warning("No ISO data keys found in the loaded data.")

    # Placeholder for future comparison logic
    # selected_isos = st.multiselect("Select ISOs to Compare:", available_keys)
    # if selected_isos:
        # Add logic to filter data for selected ISOs based on date range
        # Calculate common metrics
        # Display comparison table/plots
        # Handle potential errors like missing columns (e.g., 'LMP Difference (USD)' might require 'RT Bus average LMP' and 'DA Bus average LMP' which might not exist in all ISO dataframes)
        # st.warning("Comparison logic not yet implemented.")
        # pass
    st.warning("ISO comparison functionality is not yet implemented.")