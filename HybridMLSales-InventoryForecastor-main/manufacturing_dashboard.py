"""
Business Trend Analysis and Inventory Forecasting Dashboard
For Industrial Manufacturing Company

This dashboard provides:
1. Sales trend analysis
2. Demand forecasting visualization (Prophet, ARIMA, SARIMA, Random Forest, XGBoost)
3. Inventory intelligence and reorder point calculations
4. Risk alerts and executive summary

Author: Analytics Team
Version: 2.0 - ML Integration
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Business Trend Analysis and Sales Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

def apply_custom_styling(dark_mode=True):
    """Load external CSS stylesheet"""
    import os
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            css = f.read()
            if not dark_mode:
                css += """
                :root {
                    --bg:          #f8fafc;
                    --surface:     #ffffff;
                    --surface-2:   #f1f5f9;
                    --border:      #e2e8f0;
                    --accent:      #0f172a;
                    --success:     #059669;
                    --warning:     #d97706;
                    --danger:      #dc2626;
                    --text-1:      #0f172a;
                    --text-2:      #475569;
                    --text-3:      #64748b;
                }
                .kpi-container:hover {
                    border-color: rgba(0,0,0,0.1);
                }
                .alert-box {
                    background: var(--surface);
                }
                """
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# ============================================================================
# DATA LOADING & SAMPLE DATA GENERATION
# ============================================================================

@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """
    Load sales and inventory data from processed TC Strips data
    """
    import os
    
    # Try multiple possible file locations
    possible_paths = [
        'processed_tc_strips_data.csv',
        './processed_tc_strips_data.csv',
        'AcquiredData.xlsx',
        './AcquiredData.xlsx',
        'sales_inventory_dataset.csv',
        './sales_inventory_dataset.csv',
        '/mnt/user-data/uploads/sales_inventory_dataset.csv'
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        st.error("""
        ❌ **Data file not found!**
        
        Please make sure one of these files is in the same folder as manufacturing_dashboard.py:
        - **processed_tc_strips_data.csv** (recommended - run process_acquired_data.py first)
        - **AcquiredData.xlsx** (original Excel file)
        - **sales_inventory_dataset.csv** (alternative dataset)
        
        Your folder should look like:
        ```
        D:\\Santusht\\Minor Project\\Frontend\\
        ├── manufacturing_dashboard.py
        └── processed_tc_strips_data.csv  ← Put the data file here!
        ```
        """)
        st.stop()
    
    if csv_path.endswith('.csv'):
        df_monthly = pd.read_csv(csv_path)
        df_monthly['Date'] = pd.to_datetime(df_monthly['Date'])
        
        if 'Product_Display' in df_monthly.columns and 'TC STRIPS' in df_monthly['Product_Display'].iloc[0]:
            return df_monthly
        
        df = df_monthly
        df['timestamp'] = df_monthly['Date']
        df['transaction_type'] = 'sell'
        df['quantity'] = df_monthly['Quantity_Sold']
        df['unit_price'] = df_monthly['Sales_Value'] / df_monthly['Quantity_Sold'].replace(0, 1)
        df['sales_value'] = df_monthly['Sales_Value']
        
        df_sales = df[df['transaction_type'] == 'sell'].copy()
        
        df_result = df_sales.groupby([
            pd.Grouper(key='timestamp', freq='MS'),
            'Product_Name'
        ]).agg({
            'quantity': 'sum',
            'sales_value': 'sum',
            'current_inventory': 'last' if 'current_inventory' in df_sales.columns else lambda x: 0
        }).reset_index()
        
        df_result = df_result.rename(columns={
            'timestamp': 'Date',
            'quantity': 'Quantity_Sold',
            'sales_value': 'Sales_Value'
        })
        
        df_result['Month_Year'] = df_result['Date'].dt.strftime('%b-%Y')
        df_result['Product_Display'] = df_result['Product_Name']
        df_result['product_category'] = 'General'
        
        return df_result
    
    elif csv_path.endswith('.xlsx'):
        st.info("📊 Processing Excel file... This may take a moment.")
        
        import re
        
        def clean_price(price_str):
            if pd.isna(price_str) or price_str == '-' or price_str == '':
                return 0.0
            cleaned = str(price_str).replace('₹', '').replace(',', '').strip()
            try:
                return float(cleaned)
            except:
                return 0.0
        
        xl = pd.ExcelFile(csv_path)
        all_data = []
        
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(csv_path, sheet_name=sheet_name)
            df_clean = df.iloc[1:].copy()
            
            if df.shape[1] == 5:
                df_clean.columns = ['Serial', 'Type', 'Date', 'Quantity', 'Price']
            elif df.shape[1] == 4:
                df_clean.columns = ['Serial', 'Type', 'Date', 'Quantity']
                df_clean['Price'] = 0
            else:
                continue
            
            df_clean = df_clean[df_clean['Type'].notna()].copy()
            df_clean = df_clean[df_clean['Type'] != ''].copy()
            
            try:
                df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d/%m/%Y', errors='coerce')
            except:
                df_clean['Date'] = pd.to_datetime(df_clean['Date'], dayfirst=True, errors='coerce')
            
            df_clean = df_clean[df_clean['Date'].notna()].copy()
            df_clean['Quantity'] = pd.to_numeric(df_clean['Quantity'], errors='coerce').fillna(0).astype(int)
            df_clean['Price'] = df_clean['Price'].apply(clean_price)
            df_clean['Product_Name'] = sheet_name.strip()
            df_clean['Type'] = df_clean['Type'].str.strip()
            df_clean['Sales_Value'] = df_clean['Quantity'] * df_clean['Price']
            
            df_result = df_clean[['Date', 'Product_Name', 'Type', 'Quantity', 'Price', 'Sales_Value']].copy()
            df_result = df_result.rename(columns={'Type': 'Transaction_Type', 'Price': 'Unit_Price'})
            
            all_data.append(df_result)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        
        df_sales = combined_df[combined_df['Transaction_Type'].str.contains('Sale', case=False, na=False)].copy()
        
        df_monthly = df_sales.groupby([
            pd.Grouper(key='Date', freq='MS'),
            'Product_Name'
        ]).agg({
            'Quantity': 'sum',
            'Sales_Value': 'sum',
            'Unit_Price': 'mean'
        }).reset_index()
        
        df_monthly = df_monthly.rename(columns={'Quantity': 'Quantity_Sold'})
        df_monthly['Month_Year'] = df_monthly['Date'].dt.strftime('%b-%Y')
        
        def categorize_product(product_name):
            match = re.match(r'(\d+\.?\d*)', product_name)
            if match:
                width = float(match.group(1))
                if width < 3.0:
                    return 'Small TC Strips'
                elif width < 5.0:
                    return 'Medium TC Strips'
                else:
                    return 'Large TC Strips'
            return 'TC Strips'
        
        df_monthly['product_category'] = df_monthly['Product_Name'].apply(categorize_product)
        df_monthly['Product_Display'] = 'TC STRIPS ' + df_monthly['Product_Name']
        
        inventory = {}
        for product in combined_df['Product_Name'].unique():
            df_product = combined_df[combined_df['Product_Name'] == product].sort_values('Date')
            stock = 0
            for _, row in df_product.iterrows():
                trans_type = row['Transaction_Type'].lower()
                qty = row['Quantity']
                if 'sale' in trans_type or 'reduce' in trans_type:
                    stock -= qty
                elif 'purchase' in trans_type or 'add' in trans_type or 'opening' in trans_type:
                    stock += qty
            inventory[product] = max(stock, 0)
        
        df_monthly['current_inventory'] = df_monthly['Product_Name'].map(inventory)
        
        return df_monthly
    
    else:
        st.error("Unsupported file format!")
        st.stop()


# ============================================================================
# ML FORECASTING ENGINE
# ============================================================================

def _make_ts_features(dates: pd.Series) -> pd.DataFrame:
    """Create time-series calendar features for tree-based models."""
    df = pd.DataFrame({'ds': pd.to_datetime(dates)})
    df['month']      = df['ds'].dt.month
    df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)
    df['year']       = df['ds'].dt.year
    df['trend']      = (df['ds'] - df['ds'].min()).dt.days
    df['quarter']    = df['ds'].dt.quarter
    return df.drop(columns=['ds'])


def _forecast_prophet(series: pd.Series, dates: pd.DatetimeIndex,
                       horizon: int = 3) -> Dict:
    """
    Fit Facebook/Meta Prophet and return 3-month forecast + in-sample fitted values.
    Returns: dict with keys forecast_values, lower_bound, upper_bound,
             forecast_dates, fitted_values, model_name
    """
    from prophet import Prophet

    df_prophet = pd.DataFrame({'ds': dates, 'y': series.values})
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        interval_width=0.95,
        changepoint_prior_scale=0.1
    )
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=horizon, freq='MS')
    forecast = m.predict(future)

    # Separate historical fitted values and future forecasts
    hist_forecast = forecast.iloc[:len(series)]
    fut_forecast  = forecast.iloc[len(series):]

    fitted_values = hist_forecast['yhat'].values.tolist()
    forecast_vals = [max(int(round(v)), 0) for v in fut_forecast['yhat'].values]
    lower_bound   = [max(int(round(v)), 0) for v in fut_forecast['yhat_lower'].values]
    upper_bound   = [max(int(round(v)), 0) for v in fut_forecast['yhat_upper'].values]
    forecast_dates = [d.strftime('%b-%Y') for d in fut_forecast['ds']]

    return {
        'forecast_values': forecast_vals,
        'lower_bound':     lower_bound,
        'upper_bound':     upper_bound,
        'forecast_dates':  forecast_dates,
        'fitted_values':   fitted_values,
        'model_name':      'Prophet'
    }


def _forecast_arima(series: pd.Series, dates: pd.DatetimeIndex,
                     horizon: int = 3) -> Dict:
    """
    Fit ARIMA with auto-order selection via AIC and return forecast.
    """
    from statsmodels.tsa.arima.model import ARIMA
    import itertools

    best_aic  = np.inf
    best_order = (1, 1, 1)
    
    # Grid search over small (p,d,q) space
    for p, d, q in itertools.product(range(3), range(2), range(3)):
        try:
            res = ARIMA(series.values, order=(p, d, q)).fit()
            if res.aic < best_aic:
                best_aic   = res.aic
                best_order = (p, d, q)
        except Exception:
            continue

    model  = ARIMA(series.values, order=best_order)
    result = model.fit()

    forecast_obj   = result.get_forecast(steps=horizon)
    pred_mean      = forecast_obj.predicted_mean
    conf_int       = forecast_obj.conf_int(alpha=0.05)
    fitted_values  = result.fittedvalues.tolist()

    last_date      = dates[-1]
    fut_dates      = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                    periods=horizon, freq='MS')

    forecast_vals  = [max(int(round(v)), 0) for v in pred_mean]
    lower_bound    = [max(int(round(v)), 0) for v in conf_int[:, 0]]
    upper_bound    = [max(int(round(v)), 0) for v in conf_int[:, 1]]

    return {
        'forecast_values': forecast_vals,
        'lower_bound':     lower_bound,
        'upper_bound':     upper_bound,
        'forecast_dates':  [d.strftime('%b-%Y') for d in fut_dates],
        'fitted_values':   fitted_values,
        'model_name':      f'ARIMA{best_order}'
    }


def _forecast_sarima(series: pd.Series, dates: pd.DatetimeIndex,
                      horizon: int = 3) -> Dict:
    """
    Fit SARIMA (seasonal ARIMA) with seasonal period = 12 months.
    Auto-selects best (p,d,q)(P,D,Q)[12] via AIC.
    Falls back to ARIMA if data is too short.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import itertools

    if len(series) < 24:
        # Not enough data for seasonal fitting; fall back to ARIMA
        result = _forecast_arima(series, dates, horizon)
        result['model_name'] = 'SARIMA→ARIMA (short series)'
        return result

    best_aic   = np.inf
    best_order = ((1, 1, 1), (1, 0, 1, 12))

    for p, d, q in itertools.product(range(2), range(2), range(2)):
        for P, D, Q in itertools.product(range(2), range(1), range(2)):
            try:
                res = SARIMAX(
                    series.values,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                if res.aic < best_aic:
                    best_aic   = res.aic
                    best_order = ((p, d, q), (P, D, Q, 12))
            except Exception:
                continue

    model  = SARIMAX(
        series.values,
        order=best_order[0],
        seasonal_order=best_order[1],
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)

    forecast_obj  = result.get_forecast(steps=horizon)
    pred_mean     = forecast_obj.predicted_mean
    conf_int      = forecast_obj.conf_int(alpha=0.05)
    fitted_values = result.fittedvalues.tolist()

    last_date  = dates[-1]
    fut_dates  = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                periods=horizon, freq='MS')

    return {
        'forecast_values': [max(int(round(v)), 0) for v in pred_mean],
        'lower_bound':     [max(int(round(v)), 0) for v in conf_int.iloc[:, 0]],
        'upper_bound':     [max(int(round(v)), 0) for v in conf_int.iloc[:, 1]],
        'forecast_dates':  [d.strftime('%b-%Y') for d in fut_dates],
        'fitted_values':   fitted_values,
        'model_name':      f'SARIMA{best_order[0]}x{best_order[1]}'
    }


def _forecast_random_forest(series: pd.Series, dates: pd.DatetimeIndex,
                              horizon: int = 3) -> Dict:
    """
    Random Forest with lag features + calendar features.
    Uses walk-forward CV for CI estimation.
    """
    from sklearn.ensemble import RandomForestRegressor

    N_LAGS = min(6, len(series) - 2)

    def build_features(vals, future_dates=None, last_dates=None):
        rows = []
        for i in range(N_LAGS, len(vals)):
            lag_feats = list(vals[i - N_LAGS:i])
            date_feats = _make_ts_features(pd.Series([last_dates[i]])).iloc[0].tolist()
            rows.append(lag_feats + date_feats)
        return np.array(rows)

    vals = series.values.astype(float)
    X    = build_features(vals, last_dates=dates)
    y    = vals[N_LAGS:]

    rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                                min_samples_leaf=2, random_state=42)
    rf.fit(X, y)

    # Walk-forward forecast
    history    = list(vals)
    hist_dates = list(dates)
    last_date  = dates[-1]
    fut_dates  = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                periods=horizon, freq='MS')

    forecast_vals  = []
    all_tree_preds = []

    for fd in fut_dates:
        lag_feats  = history[-N_LAGS:]
        date_feats = _make_ts_features(pd.Series([fd])).iloc[0].tolist()
        row        = np.array([lag_feats + date_feats])
        
        # Per-tree predictions for CI
        tree_preds = np.array([tree.predict(row)[0] for tree in rf.estimators_])
        all_tree_preds.append(tree_preds)
        pred = float(np.mean(tree_preds))
        history.append(pred)
        hist_dates.append(fd)
        forecast_vals.append(max(int(round(pred)), 0))

    lower_bound = [max(int(round(np.percentile(tp, 2.5))), 0)  for tp in all_tree_preds]
    upper_bound = [max(int(round(np.percentile(tp, 97.5))), 0) for tp in all_tree_preds]

    # In-sample fitted values
    fitted_raw    = rf.predict(X)
    fitted_values = [float(v) for v in fitted_raw]
    # Prepend NaN for the lag warm-up period
    fitted_values = [np.nan] * N_LAGS + fitted_values

    return {
        'forecast_values': forecast_vals,
        'lower_bound':     lower_bound,
        'upper_bound':     upper_bound,
        'forecast_dates':  [d.strftime('%b-%Y') for d in fut_dates],
        'fitted_values':   fitted_values,
        'model_name':      'Random Forest'
    }


def _forecast_xgboost(series: pd.Series, dates: pd.DatetimeIndex,
                       horizon: int = 3) -> Dict:
    """
    XGBoost with lag features + calendar features.
    """
    from xgboost import XGBRegressor

    N_LAGS = min(6, len(series) - 2)

    def build_features(vals, date_series):
        rows = []
        for i in range(N_LAGS, len(vals)):
            lag_feats  = list(vals[i - N_LAGS:i])
            date_feats = _make_ts_features(pd.Series([date_series[i]])).iloc[0].tolist()
            rows.append(lag_feats + date_feats)
        return np.array(rows)

    vals = series.values.astype(float)
    X    = build_features(vals, dates)
    y    = vals[N_LAGS:]

    xgb = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0
    )
    xgb.fit(X, y)

    history   = list(vals)
    last_date = dates[-1]
    fut_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                               periods=horizon, freq='MS')

    forecast_vals = []
    for fd in fut_dates:
        lag_feats  = history[-N_LAGS:]
        date_feats = _make_ts_features(pd.Series([fd])).iloc[0].tolist()
        row        = np.array([lag_feats + date_feats])
        pred       = float(xgb.predict(row)[0])
        history.append(pred)
        forecast_vals.append(max(int(round(pred)), 0))

    # Bootstrap CI from in-sample residuals
    fitted_raw     = xgb.predict(X)
    residuals      = y - fitted_raw
    std_res        = np.std(residuals)
    lower_bound    = [max(int(round(v - 1.96 * std_res)), 0) for v in forecast_vals]
    upper_bound    = [max(int(round(v + 1.96 * std_res)), 0) for v in forecast_vals]
    fitted_values  = [np.nan] * N_LAGS + [float(v) for v in fitted_raw]

    return {
        'forecast_values': forecast_vals,
        'lower_bound':     lower_bound,
        'upper_bound':     upper_bound,
        'forecast_dates':  [d.strftime('%b-%Y') for d in fut_dates],
        'fitted_values':   fitted_values,
        'model_name':      'XGBoost'
    }


# ─── Model Evaluation ────────────────────────────────────────────────────────

def _evaluate_model(name: str, forecast_fn, series: pd.Series,
                     dates: pd.DatetimeIndex, n_test: int = 6) -> Tuple[float, float]:
    """
    Walk-forward cross-validation: train on all but last n_test points,
    predict n_test steps, compare against actuals.
    Returns (RMSE, MAPE).
    """
    if len(series) < n_test + 12:
        # Too few data points for this n_test; use smaller window
        n_test = max(3, len(series) // 4)

    train_series = series.iloc[:-n_test]
    train_dates  = dates[:-n_test]
    actuals      = series.iloc[-n_test:].values

    try:
        result = forecast_fn(train_series, train_dates, horizon=n_test)
        preds  = np.array(result['forecast_values'], dtype=float)
        # Align lengths (some models may return fewer steps)
        min_len  = min(len(preds), len(actuals))
        preds    = preds[:min_len]
        actuals_ = actuals[:min_len]

        rmse = float(np.sqrt(np.mean((preds - actuals_) ** 2)))
        # MAPE: avoid zero division
        mask  = actuals_ != 0
        mape  = float(np.mean(np.abs((actuals_[mask] - preds[mask]) / actuals_[mask])) * 100) \
                if mask.any() else np.inf
        return rmse, mape
    except Exception as e:
        return np.inf, np.inf


@st.cache_data(show_spinner=False)
def select_best_model_and_forecast(product_name: str,
                                    series_values: tuple,
                                    dates_str: tuple,
                                    metric: str) -> Dict:
    """
    Evaluate all 5 models via walk-forward CV, select the best by RMSE,
    then refit on the full series and return the 3-month forecast.

    Parameters are hashable (tuples) so Streamlit can cache them.
    """
    series = pd.Series(list(series_values), dtype=float)
    dates  = pd.DatetimeIndex(list(dates_str))

    MODEL_FNS = {
        'Prophet':       _forecast_prophet,
        'ARIMA':         _forecast_arima,
        'SARIMA':        _forecast_sarima,
        'Random Forest': _forecast_random_forest,
        'XGBoost':       _forecast_xgboost,
    }

    evaluation_results = {}
    for name, fn in MODEL_FNS.items():
        rmse, mape = _evaluate_model(name, fn, series, dates)
        evaluation_results[name] = {'rmse': rmse, 'mape': mape}

    # Pick best model by RMSE (exclude inf)
    valid   = {k: v for k, v in evaluation_results.items() if v['rmse'] < np.inf}
    if valid:
        best_name = min(valid, key=lambda k: valid[k]['rmse'])
    else:
        best_name = 'Prophet'   # safe fallback

    # Refit best model on full data
    best_forecast = MODEL_FNS[best_name](series, dates, horizon=3)

    # Attach evaluation table to result
    best_forecast['evaluation']  = evaluation_results
    best_forecast['best_model']  = best_name
    best_forecast['model_name']  = best_name

    return best_forecast


# ─── Public interface (same signature as original) ───────────────────────────

def get_ml_forecast(product_name: str, historical_data: pd.DataFrame,
                    metric: str = 'Quantity_Sold') -> Dict:
    """
    Main forecast entry-point.  Matches the original function signature exactly.

    Runs model selection + best-model forecasting.  Results are cached by
    Streamlit so re-renders don't re-run the heavy computation.
    """
    series_values = tuple(historical_data[metric].values.tolist())
    dates_str     = tuple(historical_data['Date'].astype(str).tolist())

    return select_best_model_and_forecast(product_name, series_values, dates_str, metric)


# ============================================================================
# INVENTORY CALCULATIONS
# ============================================================================

def calculate_inventory_metrics(historical_data: pd.DataFrame, 
                                metric: str,
                                current_stock: int,
                                lead_time_months: int = 3) -> Dict:
    avg_monthly_demand = historical_data.tail(6)[metric].mean()
    std_demand         = historical_data.tail(6)[metric].std()
    service_level_factor = 1.65
    safety_stock       = service_level_factor * std_demand * np.sqrt(lead_time_months)
    reorder_point      = (avg_monthly_demand * lead_time_months) + safety_stock
    stock_coverage     = current_stock / avg_monthly_demand if avg_monthly_demand > 0 else 0

    return {
        'avg_monthly_demand':    int(avg_monthly_demand),
        'std_demand':            int(std_demand),
        'safety_stock':          int(safety_stock),
        'reorder_point':         int(reorder_point),
        'stock_coverage_months': round(stock_coverage, 1),
        'lead_time_demand':      int(avg_monthly_demand * lead_time_months)
    }


def determine_risk_level(current_stock: int, reorder_point: int,
                         safety_stock: int) -> Tuple[str, str, str]:
    if current_stock <= safety_stock:
        return "CRITICAL - Stockout Risk", "alert-red",    "ORDER IMMEDIATELY"
    elif current_stock <= reorder_point:
        return "WARNING - Reorder Soon",   "alert-yellow", "PLACE ORDER"
    elif current_stock > reorder_point * 1.5:
        return "HEALTHY - Stock Safe",     "alert-green",  "MONITOR"
    else:
        return "ADEQUATE - Stock Safe",    "alert-green",  "HOLD"


# ============================================================================
# CHART HELPERS
# ============================================================================

PLOT_LAYOUT = dict(
    font=dict(family='Inter, -apple-system, sans-serif', size=12, color='#9494a0'),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(t=64, b=48, l=52, r=24),
)
PLOT_GRID      = dict(showgrid=True,  gridwidth=1, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#5a5a68'))
PLOT_GRID_NONE = dict(showgrid=False, tickfont=dict(color='#5a5a68'))

COLOR_ACTUAL   = '#e2e8f0'
COLOR_FORECAST = '#94a3b8'
COLOR_CI       = 'rgba(148, 163, 184, 0.1)'
COLOR_BARS     = ['#e2e8f0', '#64748b', '#475569', '#334155']
COLOR_FITTED   = 'rgba(250, 204, 21, 0.6)'   # soft gold for fitted line
TITLE_COLOR    = '#f0f0f2'


def create_sales_trend_chart(data: pd.DataFrame, product: str,
                              metric: str, forecast_data: Dict = None) -> go.Figure:
    """Create interactive sales trend line chart with optional forecast and fitted line."""
    fig = go.Figure()
    metric_label_hover = metric.replace("_", " ")

    # Actual line
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data[metric],
        mode='lines+markers', name='Actual',
        line=dict(color=COLOR_ACTUAL, width=2),
        marker=dict(size=5),
        hovertemplate='<b>%{x|%b %Y}</b><br>' + metric_label_hover + ': %{y:,.0f}<extra></extra>'
    ))

    # Fitted values (in-sample model fit)
    if forecast_data and 'fitted_values' in forecast_data:
        fitted = forecast_data['fitted_values']
        fitted_clean = [v if not (isinstance(v, float) and np.isnan(v)) else None
                        for v in fitted]
        fig.add_trace(go.Scatter(
            x=data['Date'], y=fitted_clean,
            mode='lines', name=f"Fitted ({forecast_data.get('model_name','Model')})",
            line=dict(color=COLOR_FITTED, width=1.5, dash='dot'),
            hovertemplate='<b>%{x|%b %Y}</b><br>Fitted: %{y:,.0f}<extra></extra>'
        ))

    if forecast_data:
        forecast_dates = pd.to_datetime(forecast_data['forecast_dates'], format='%b-%Y')

        # Connector
        fig.add_trace(go.Scatter(
            x=[data['Date'].iloc[-1], forecast_dates[0]],
            y=[data[metric].iloc[-1], forecast_data['forecast_values'][0]],
            mode='lines',
            line=dict(color=COLOR_FORECAST, width=1.5, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_data['forecast_values'],
            mode='lines+markers', name='Forecast',
            line=dict(color=COLOR_FORECAST, width=2, dash='dash'),
            marker=dict(size=5, symbol='diamond'),
            hovertemplate='<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>'
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
            y=forecast_data['upper_bound'] + forecast_data['lower_bound'][::-1],
            fill='toself', fillcolor=COLOR_CI,
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI', hoverinfo='skip'
        ))

    metric_label = 'Quantity Sold' if metric == 'Quantity_Sold' else 'Sales Value (₹)'
    layout = dict(**PLOT_LAYOUT)
    layout.update(
        title=dict(text=f'{product} — {metric_label}',
                   font=dict(color=TITLE_COLOR, size=15, family='Inter, sans-serif')),
        hovermode='x unified', height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                    font=dict(color='#9494a0', size=11))
    )
    fig.update_layout(**layout)
    fig.update_xaxes(**PLOT_GRID)
    fig.update_yaxes(**PLOT_GRID)
    return fig


def create_inventory_comparison_chart(current_stock: int, reorder_point: int,
                                       safety_stock: int, lead_time_demand: int) -> go.Figure:
    categories = ['Current Stock', 'Reorder Point', 'Safety Stock', 'Lead Time Demand']
    values     = [current_stock, reorder_point, safety_stock, lead_time_demand]

    fig = go.Figure(data=[go.Bar(
        x=categories, y=values,
        marker_color=COLOR_BARS,
        text=[f'{v:,}' for v in values], textposition='outside',
        textfont=dict(color='#9494a0', size=11),
        hovertemplate='<b>%{x}</b><br>%{y:,.0f}<extra></extra>'
    )])

    layout = dict(**PLOT_LAYOUT)
    layout.update(
        title=dict(text='Inventory Levels',
                   font=dict(color=TITLE_COLOR, size=15, family='Inter, sans-serif')),
        height=340, showlegend=False
    )
    fig.update_layout(**layout)
    fig.update_xaxes(**PLOT_GRID_NONE)
    fig.update_yaxes(**PLOT_GRID)
    return fig


def create_forecast_comparison_chart(forecast_data: Dict, metric: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=forecast_data['forecast_dates'],
        y=forecast_data['forecast_values'],
        name='Forecast', marker_color=COLOR_ACTUAL,
        text=[f'{v:,}' for v in forecast_data['forecast_values']],
        textposition='outside', textfont=dict(color='#9494a0', size=11),
        hovertemplate='<b>%{x}</b><br>Forecast: %{y:,.0f}<extra></extra>'
    ))

    error_y = {
        'type': 'data', 'symmetric': False,
        'array':      [u - f for u, f in zip(forecast_data['upper_bound'], forecast_data['forecast_values'])],
        'arrayminus': [f - l for f, l in zip(forecast_data['forecast_values'], forecast_data['lower_bound'])],
        'color': '#5a5a68', 'thickness': 1.5, 'width': 8
    }
    fig.update_traces(error_y=error_y)

    layout = dict(**PLOT_LAYOUT)
    layout.update(
        title=dict(text='3-Month Demand Forecast',
                   font=dict(color=TITLE_COLOR, size=15, family='Inter, sans-serif')),
        height=340, showlegend=False
    )
    fig.update_layout(**layout)
    fig.update_xaxes(**PLOT_GRID_NONE)
    fig.update_yaxes(**PLOT_GRID)
    return fig


def create_model_evaluation_chart(evaluation: Dict) -> go.Figure:
    """Bar chart showing RMSE for each candidate model."""
    models = list(evaluation.keys())
    rmse_vals = [evaluation[m]['rmse'] if evaluation[m]['rmse'] < np.inf else 0
                 for m in models]
    mape_vals = [evaluation[m]['mape'] if evaluation[m]['mape'] < np.inf else 0
                 for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='RMSE', x=models, y=rmse_vals,
        marker_color=COLOR_BARS[0],
        text=[f'{v:,.1f}' for v in rmse_vals], textposition='outside',
        textfont=dict(color='#9494a0', size=11),
        hovertemplate='<b>%{x}</b><br>RMSE: %{y:,.1f}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='MAPE (%)', x=models, y=mape_vals,
        marker_color=COLOR_BARS[1],
        text=[f'{v:.1f}%' for v in mape_vals], textposition='outside',
        textfont=dict(color='#9494a0', size=11),
        hovertemplate='<b>%{x}</b><br>MAPE: %{y:.1f}%<extra></extra>'
    ))

    layout = dict(**PLOT_LAYOUT)
    layout.update(
        title=dict(text='Model Evaluation — Walk-Forward CV',
                   font=dict(color=TITLE_COLOR, size=15, family='Inter, sans-serif')),
        barmode='group', height=340,
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                    font=dict(color='#9494a0', size=11))
    )
    fig.update_layout(**layout)
    fig.update_xaxes(**PLOT_GRID_NONE)
    fig.update_yaxes(**PLOT_GRID)
    return fig


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    st.sidebar.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h3 style="font-size: 1.1rem; font-weight: 600; color: var(--text-1); margin: 0; line-height: 1.3;">Business Trend Analysis and Sales Prediction System</h3>
        <p style="font-size: 0.85rem; color: var(--text-2); margin-top: 0.3rem;">Praj Engineers</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("## Appearance")
    dark_mode = st.sidebar.toggle("Dark Mode", value=True)
    apply_custom_styling(dark_mode)

    global PLOT_LAYOUT, PLOT_GRID, PLOT_GRID_NONE, COLOR_ACTUAL, COLOR_FORECAST, COLOR_CI, COLOR_BARS, COLOR_FITTED, TITLE_COLOR
    if not dark_mode:
        PLOT_LAYOUT['font']['color'] = '#475569'
        PLOT_GRID['gridcolor'] = 'rgba(0,0,0,0.05)'
        PLOT_GRID['tickfont']['color'] = '#64748b'
        PLOT_GRID_NONE['tickfont']['color'] = '#64748b'
        COLOR_ACTUAL = '#2563eb'
        COLOR_FORECAST = '#60a5fa'
        COLOR_CI = 'rgba(96, 165, 250, 0.15)'
        COLOR_BARS = ['#1d4ed8', '#3b82f6', '#93c5fd', '#dbeafe']
        COLOR_FITTED = 'rgba(245, 158, 11, 0.8)'
        TITLE_COLOR = '#0f172a'
    else:
        PLOT_LAYOUT['font']['color'] = '#9494a0'
        PLOT_GRID['gridcolor'] = 'rgba(255,255,255,0.05)'
        PLOT_GRID['tickfont']['color'] = '#5a5a68'
        PLOT_GRID_NONE['tickfont']['color'] = '#5a5a68'
        COLOR_ACTUAL = '#e2e8f0'
        COLOR_FORECAST = '#94a3b8'
        COLOR_CI = 'rgba(148, 163, 184, 0.1)'
        COLOR_BARS = ['#e2e8f0', '#64748b', '#475569', '#334155']
        COLOR_FITTED = 'rgba(250, 204, 21, 0.6)'
        TITLE_COLOR = '#f0f0f2'

    st.markdown("""
    <div class="main-header">
        <h1>Business Trend Analysis and Sales Prediction System</h1>
        <p>Praj Engineers</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_sample_data()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("## Dashboard Controls")
    st.sidebar.markdown("---")

    products = sorted(df['Product_Display'].unique())
    selected_product_display = st.sidebar.selectbox(" Select Product", products, index=0)
    selected_product = df[df['Product_Display'] == selected_product_display]['Product_Name'].iloc[0]

    st.sidebar.markdown("###  Date Range")
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    date_range = st.sidebar.date_input(
        "Select Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    metric_option = st.sidebar.radio(
        " View Metric",
        ["Quantity Sold", "Sales Value"],
        horizontal=True
    )
    metric = 'Quantity_Sold' if metric_option == "Quantity Sold" else 'Sales_Value'

    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Inventory Parameters")

    latest_inventory = df[df['Product_Name'] == selected_product]['current_inventory'].iloc[-1] \
                       if len(df[df['Product_Name'] == selected_product]) > 0 else 150

    current_stock = st.sidebar.number_input(
        "Current Stock Level",
        min_value=0, value=int(latest_inventory), step=10,
        help="Enter current inventory level for the selected product"
    )

    st.sidebar.markdown(f"**Lead Time:** 3 months (Fixed)")
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Use the product selector and date range to analyze different products and time periods.")

    # ── FILTER DATA ───────────────────────────────────────────────────────────
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    filtered_df = df[
        (df['Product_Name'] == selected_product) &
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))
    ].sort_values('Date')

    if filtered_df.empty:
        st.error("⚠️ No data available for the selected filters. Please adjust your selection.")
        return

    if 'current_inventory' in filtered_df.columns:
        actual_inventory = filtered_df['current_inventory'].iloc[-1]
        st.sidebar.info(f" **Latest Inventory from Data:** {actual_inventory:,} units")

    # ── SECTION 1: SALES & TREND ANALYSIS ────────────────────────────────────
    st.markdown('<div class="section-header">Sales &amp; Trend Analysis</div>', unsafe_allow_html=True)

    if len(filtered_df) >= 2:
        last_month_value = filtered_df.iloc[-1][metric]
        prev_month_value = filtered_df.iloc[-2][metric]
        mom_growth = ((last_month_value - prev_month_value) / prev_month_value * 100) \
                     if prev_month_value != 0 else 0
    else:
        mom_growth = 0

    # ── ML FORECAST (with spinner while computing) ────────────────────────────
    with st.spinner(" Running model selection & forecasting… (cached after first run)"):
        forecast_data = get_ml_forecast(selected_product, filtered_df, metric)

    best_model_name = forecast_data.get('best_model', forecast_data.get('model_name', 'ML Model'))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=f"Latest Month {metric_option}",
                  value=f"{filtered_df.iloc[-1][metric]:,.0f}")
    with col2:
        st.metric(label="Month-over-Month Growth",
                  value=f"{abs(mom_growth):.1f}%",
                  delta=f"{mom_growth:+.1f}%")
    with col3:
        avg_value = filtered_df[metric].mean()
        st.metric(label=f"Average Monthly {metric_option}",
                  value=f"{avg_value:,.0f}")
    with col4:
        total_forecast = sum(forecast_data['forecast_values'])
        st.metric(label="3-Month Forecast Total",
                  value=f"{total_forecast:,.0f}",
                  delta=f"Model: {best_model_name}")

    st.plotly_chart(
        create_sales_trend_chart(filtered_df, selected_product_display, metric, forecast_data),
        use_container_width=True
    )

    # ── SECTION 2: FORECASTING OUTPUT ────────────────────────────────────────
    st.markdown('<div class="section-header">Demand Forecasting — Next 3 Months</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(create_forecast_comparison_chart(forecast_data, metric),
                        use_container_width=True)
    with col2:
        st.markdown("#### Forecast Details")
        forecast_df = pd.DataFrame({
            'Month':       forecast_data['forecast_dates'],
            'Forecast':    forecast_data['forecast_values'],
            'Lower Bound': forecast_data['lower_bound'],
            'Upper Bound': forecast_data['upper_bound']
        })
        st.dataframe(
            forecast_df.style.format({
                'Forecast': '{:,.0f}', 'Lower Bound': '{:,.0f}', 'Upper Bound': '{:,.0f}'
            }),
            hide_index=True, use_container_width=True
        )
        st.info(f"""
        **Model Selected:** {best_model_name}  
        Chosen via walk-forward cross-validation (RMSE).  
        Confidence intervals represent 95% prediction bounds.
        """)

    # ── SECTION 2b: MODEL EVALUATION ─────────────────────────────────────────
    st.markdown('<div class="section-header">Model Evaluation &amp; Selection</div>',
                unsafe_allow_html=True)

    evaluation = forecast_data.get('evaluation', {})
    if evaluation:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(create_model_evaluation_chart(evaluation),
                            use_container_width=True)
        with col2:
            st.markdown("#### Model Rankings")
            eval_rows = []
            for m_name, scores in evaluation.items():
                eval_rows.append({
                    'Model': m_name,
                    'RMSE':  f"{scores['rmse']:.2f}" if scores['rmse'] < np.inf else '—',
                    'MAPE (%)': f"{scores['mape']:.2f}" if scores['mape'] < np.inf else '—',
                    'Selected': '✅' if m_name == best_model_name else ''
                })
            eval_df = pd.DataFrame(eval_rows)
            # Sort by RMSE ascending (put inf at bottom)
            eval_df['_sort'] = eval_df['RMSE'].apply(
                lambda x: float(x) if x != '—' else np.inf
            )
            eval_df = eval_df.sort_values('_sort').drop(columns=['_sort'])
            st.dataframe(eval_df, hide_index=True, use_container_width=True)

            st.info("""
            **Evaluation method:** Walk-forward CV  
            Train on all data except last 6 months,  
            predict 6 steps ahead, compare to actuals.  
            Best RMSE model is used for live forecast.
            """)

    # ── SECTION 3: INVENTORY INTELLIGENCE ────────────────────────────────────
    st.markdown('<div class="section-header">Inventory Intelligence</div>',
                unsafe_allow_html=True)

    inventory_metrics = calculate_inventory_metrics(
        filtered_df, metric, current_stock, lead_time_months=3
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Avg Monthly Demand",
                  value=f"{inventory_metrics['avg_monthly_demand']:,}")
    with col2:
        st.metric(label="Safety Stock",
                  value=f"{inventory_metrics['safety_stock']:,}")
    with col3:
        st.metric(label="Reorder Point (ROP)",
                  value=f"{inventory_metrics['reorder_point']:,}")
    with col4:
        st.metric(label="Stock Coverage",
                  value=f"{inventory_metrics['stock_coverage_months']:.1f} months")

    st.plotly_chart(
        create_inventory_comparison_chart(
            current_stock,
            inventory_metrics['reorder_point'],
            inventory_metrics['safety_stock'],
            inventory_metrics['lead_time_demand']
        ),
        use_container_width=True
    )

    with st.expander(" View Calculation Details"):
        st.markdown(f"""
        **Inventory Calculations:**

        - **Average Monthly Demand:** Calculated from last 6 months of sales data
          - Value: {inventory_metrics['avg_monthly_demand']:,} units

        - **Safety Stock:** Buffer inventory to protect against demand variability
          - Formula: 1.65 × σ × √(Lead Time)
          - Standard Deviation (σ): {inventory_metrics['std_demand']:,}
          - Lead Time: 3 months
          - Value: {inventory_metrics['safety_stock']:,} units

        - **Reorder Point (ROP):** Inventory level that triggers new order
          - Formula: (Avg Demand × Lead Time) + Safety Stock
          - Lead Time Demand: {inventory_metrics['lead_time_demand']:,} units
          - Value: {inventory_metrics['reorder_point']:,} units

        - **Current Stock:** {current_stock:,} units
        - **Stock Coverage:** {inventory_metrics['stock_coverage_months']:.1f} months at current demand rate
        """)

    # ── SECTION 4: ALERT & RISK SYSTEM ───────────────────────────────────────
    st.markdown('<div class="section-header">Risk Alerts</div>', unsafe_allow_html=True)

    risk_level, alert_class, recommended_action = determine_risk_level(
        current_stock, inventory_metrics['reorder_point'], inventory_metrics['safety_stock']
    )

    if "CRITICAL" in risk_level:
        icon = "🔴"
        message = f"""
        **{icon} CRITICAL STOCKOUT RISK**

        Current stock ({current_stock:,} units) is at or below safety stock level
        ({inventory_metrics['safety_stock']:,} units). Immediate action required to prevent stockout.

        **Recommended Action:** {recommended_action}

        **Impact:** High risk of production stoppage and customer dissatisfaction.
        """
    elif "WARNING" in risk_level:
        icon = "🟡"
        message = f"""
        **{icon} REORDER RECOMMENDED**

        Current stock ({current_stock:,} units) has reached the reorder point
        ({inventory_metrics['reorder_point']:,} units). Consider placing an order soon.

        **Recommended Action:** {recommended_action}

        **Impact:** Given the 3-month lead time, delaying may lead to stockout risk.
        """
    else:
        icon = "🟢"
        message = f"""
        **{icon} STOCK LEVELS HEALTHY**

        Current stock ({current_stock:,} units) is above the reorder point
        ({inventory_metrics['reorder_point']:,} units). Inventory levels are adequate.

        **Recommended Action:** {recommended_action}

        **Impact:** Continue monitoring. Stock should cover demand for approximately
        {inventory_metrics['stock_coverage_months']:.1f} months.
        """

    st.markdown(f'<div class="alert-box {alert_class}">{message}</div>',
                unsafe_allow_html=True)

    # ── SECTION 5: EXECUTIVE SUMMARY ─────────────────────────────────────────
    st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">3-Month Forecasted Demand</div>
            <div class="kpi-value">{sum(forecast_data['forecast_values']):,}</div>
            <div class="kpi-sub">avg {sum(forecast_data['forecast_values']) / 3:,.0f} / month</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        text_color = "#ef4444" if "CRITICAL" in risk_level else \
                     ("#f59e0b" if "WARNING" in risk_level else "#10b981")
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Stock Risk Level</div>
            <div class="kpi-value" style="color: {text_color}; font-size: 1.75rem; letter-spacing: -0.5px;">{risk_level.split('-')[0].strip()}</div>
            <div class="kpi-sub">{inventory_metrics['stock_coverage_months']:.1f} months coverage</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        text_color = "#ef4444" if "ORDER" in recommended_action else \
                     ("#f59e0b" if "PLACE" in recommended_action else "#10b981")
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Recommended Action</div>
            <div class="kpi-value" style="color: {text_color}; font-size: 1.75rem; letter-spacing: -0.5px;">{recommended_action}</div>
            <div class="kpi-sub">Lead time: 3 months · Model: {best_model_name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 🔍 Key Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        ** Sales Performance:**
        - Latest month sales: {filtered_df.iloc[-1][metric]:,}
        - Month-over-month change: {mom_growth:+.1f}%
        - 6-month average: {filtered_df.tail(6)[metric].mean():,.0f}
        - Trend: {" Increasing" if mom_growth > 5 else ("📉 Decreasing" if mom_growth < -5 else "➡️ Stable")}
        """)

    with col2:
        gap = current_stock - inventory_metrics['reorder_point']
        st.markdown(f"""
        ** Inventory Status:**
        - Current stock: {current_stock:,} units
        - Stock vs ROP: {gap:+,} units
        - Safety buffer: {current_stock - inventory_metrics['safety_stock']:,} units
        - Action needed: {recommended_action}
        """)

    st.markdown("---")
    st.markdown("""
    <div class="dashboard-footer">
        Version 2.0 &nbsp;·&nbsp; {} &nbsp;·&nbsp; Data: {} to {} &nbsp;·&nbsp; Forecast: {}
    </div>
    """.format(
        datetime.now().strftime('%Y-%m-%d %H:%M'),
        filtered_df['Date'].min().strftime('%b %Y'),
        filtered_df['Date'].max().strftime('%b %Y'),
        best_model_name
    ), unsafe_allow_html=True)


if __name__ == "__main__":
    main()