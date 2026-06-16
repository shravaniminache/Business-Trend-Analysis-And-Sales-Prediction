"""
ML Model Integration Guide for Manufacturing Dashboard

This module provides complete examples for integrating various ML forecasting
models into the dashboard.

Supported Models:
1. ARIMA/SARIMA (Statistical)
2. Prophet (Facebook's forecasting library)
3. LSTM (Deep Learning)
4. Ensemble Methods (Combining multiple models)

Author: Analytics Team
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# OPTION 1: ARIMA/SARIMA MODEL
# ============================================================================

class ARIMAForecaster:
    """
    ARIMA/SARIMA-based demand forecasting
    
    Pros:
    - Excellent for capturing trends and seasonality
    - Well-established statistical method
    - Provides confidence intervals naturally
    
    Cons:
    - Requires sufficient historical data (24+ months recommended)
    - Can be computationally expensive for parameter tuning
    """
    
    def __init__(self):
        """Initialize ARIMA forecaster"""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self.SARIMAX = SARIMAX
        except ImportError:
            raise ImportError(
                "statsmodels not installed. Run: pip install statsmodels"
            )
    
    def forecast(self, product_name: str, historical_data: pd.DataFrame,
                metric: str = 'Quantity_Sold', 
                periods: int = 3) -> Dict:
        """
        Generate ARIMA forecast
        
        Parameters:
        -----------
        product_name : str
            Product identifier
        historical_data : pd.DataFrame
            Historical sales data
        metric : str
            Column to forecast ('Quantity_Sold' or 'Sales_Value')
        periods : int
            Number of periods to forecast (default: 3 months)
            
        Returns:
        --------
        Dict with forecast_values, lower_bound, upper_bound, forecast_dates
        """
        
        # Prepare time series
        ts_data = historical_data.set_index('Date')[metric]
        ts_data = ts_data.asfreq('MS')  # Set monthly frequency
        
        # Fill any missing values
        ts_data = ts_data.fillna(method='ffill')
        
        # Determine SARIMA parameters
        # For demonstration, using fixed parameters
        # In production, use auto_arima for parameter selection
        order = (1, 1, 1)  # (p, d, q)
        seasonal_order = (1, 1, 1, 12)  # (P, D, Q, s) - s=12 for monthly
        
        try:
            # Fit SARIMA model
            model = self.SARIMAX(
                ts_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False)
            
            # Generate forecast
            forecast_result = fitted_model.get_forecast(steps=periods)
            forecast_values = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.05)  # 95% CI
            
            # Generate forecast dates
            last_date = historical_data['Date'].max()
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='MS'
            )
            
            return {
                'forecast_values': [int(max(v, 0)) for v in forecast_values],
                'lower_bound': [int(max(v, 0)) for v in conf_int.iloc[:, 0]],
                'upper_bound': [int(max(v, 0)) for v in conf_int.iloc[:, 1]],
                'forecast_dates': [d.strftime('%b-%Y') for d in forecast_dates],
                'model_type': 'SARIMA',
                'parameters': f"Order: {order}, Seasonal: {seasonal_order}"
            }
            
        except Exception as e:
            print(f"ARIMA forecasting error: {e}")
            # Fallback to simple trend-based forecast
            return self._fallback_forecast(historical_data, metric, periods)
    
    def _fallback_forecast(self, historical_data: pd.DataFrame,
                          metric: str, periods: int) -> Dict:
        """Fallback forecast using simple trend extrapolation"""
        
        recent_values = historical_data.tail(6)[metric].values
        avg_value = np.mean(recent_values)
        trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
        
        forecast_values = []
        for i in range(periods):
            forecast_values.append(int(max(avg_value + trend * (i + 1), 0)))
        
        last_date = historical_data['Date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='MS'
        )
        
        return {
            'forecast_values': forecast_values,
            'lower_bound': [int(v * 0.85) for v in forecast_values],
            'upper_bound': [int(v * 1.15) for v in forecast_values],
            'forecast_dates': [d.strftime('%b-%Y') for d in forecast_dates],
            'model_type': 'Trend-based (Fallback)'
        }

# ============================================================================
# OPTION 2: PROPHET MODEL
# ============================================================================

class ProphetForecaster:
    """
    Facebook Prophet-based demand forecasting
    
    Pros:
    - Handles missing data and outliers well
    - Easy to incorporate holidays and special events
    - Fast and robust
    
    Cons:
    - Less flexible than ARIMA for some patterns
    - Requires Prophet library installation
    """
    
    def __init__(self):
        """Initialize Prophet forecaster"""
        try:
            from prophet import Prophet
            self.Prophet = Prophet
        except ImportError:
            raise ImportError(
                "Prophet not installed. Run: pip install prophet"
            )
    
    def forecast(self, product_name: str, historical_data: pd.DataFrame,
                metric: str = 'Quantity_Sold',
                periods: int = 3) -> Dict:
        """
        Generate Prophet forecast
        
        Parameters same as ARIMAForecaster.forecast()
        """
        
        # Prepare data for Prophet
        df_prophet = historical_data[['Date', metric]].copy()
        df_prophet.columns = ['ds', 'y']
        
        # Initialize and configure Prophet
        model = self.Prophet(
            interval_width=0.95,  # 95% confidence interval
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'  # Better for sales data
        )
        
        # Add custom seasonality if enough data
        if len(df_prophet) >= 24:
            model.add_seasonality(
                name='quarterly',
                period=91.25,
                fourier_order=5
            )
        
        try:
            # Fit model
            model.fit(df_prophet)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq='MS')
            
            # Generate predictions
            forecast = model.predict(future)
            
            # Extract future predictions
            future_forecast = forecast.tail(periods)
            
            return {
                'forecast_values': [int(max(v, 0)) for v in future_forecast['yhat']],
                'lower_bound': [int(max(v, 0)) for v in future_forecast['yhat_lower']],
                'upper_bound': [int(max(v, 0)) for v in future_forecast['yhat_upper']],
                'forecast_dates': [d.strftime('%b-%Y') for d in future_forecast['ds']],
                'model_type': 'Prophet',
                'trend': 'Multiplicative Seasonality'
            }
            
        except Exception as e:
            print(f"Prophet forecasting error: {e}")
            return self._simple_forecast(historical_data, metric, periods)
    
    def _simple_forecast(self, historical_data: pd.DataFrame,
                        metric: str, periods: int) -> Dict:
        """Simple moving average forecast"""
        
        avg_value = historical_data.tail(6)[metric].mean()
        std_value = historical_data.tail(6)[metric].std()
        
        forecast_values = [int(avg_value)] * periods
        
        last_date = historical_data['Date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='MS'
        )
        
        return {
            'forecast_values': forecast_values,
            'lower_bound': [int(avg_value - 1.96 * std_value)] * periods,
            'upper_bound': [int(avg_value + 1.96 * std_value)] * periods,
            'forecast_dates': [d.strftime('%b-%Y') for d in forecast_dates],
            'model_type': 'Moving Average (Fallback)'
        }

# ============================================================================
# OPTION 3: LSTM DEEP LEARNING MODEL
# ============================================================================

class LSTMForecaster:
    """
    LSTM Neural Network-based demand forecasting
    
    Pros:
    - Can capture complex non-linear patterns
    - Excellent for long-term dependencies
    
    Cons:
    - Requires significant historical data
    - Longer training time
    - Less interpretable than statistical models
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize LSTM forecaster
        
        Parameters:
        -----------
        model_path : str, optional
            Path to pre-trained model file (.h5 or .keras)
        """
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            self.tf = tf
            self.MinMaxScaler = MinMaxScaler
            self.model_path = model_path
        except ImportError:
            raise ImportError(
                "TensorFlow and scikit-learn not installed. "
                "Run: pip install tensorflow scikit-learn"
            )
    
    def forecast(self, product_name: str, historical_data: pd.DataFrame,
                metric: str = 'Quantity_Sold',
                periods: int = 3,
                lookback: int = 12) -> Dict:
        """
        Generate LSTM forecast
        
        Parameters:
        -----------
        product_name : str
            Product identifier
        historical_data : pd.DataFrame
            Historical sales data
        metric : str
            Column to forecast
        periods : int
            Number of periods to forecast
        lookback : int
            Number of past months to consider (sequence length)
            
        Returns:
        --------
        Dict with forecast data
        """
        
        # Extract time series
        ts_data = historical_data[metric].values.reshape(-1, 1)
        
        # Check if we have enough data
        if len(ts_data) < lookback + periods:
            print(f"Insufficient data for LSTM. Need at least {lookback + periods} points.")
            return self._exponential_smoothing_forecast(historical_data, metric, periods)
        
        # Scale data
        scaler = self.MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(ts_data)
        
        try:
            # Load or create model
            if self.model_path and os.path.exists(self.model_path):
                model = self.tf.keras.models.load_model(self.model_path)
            else:
                model = self._build_lstm_model(lookback)
                # In production, you would train the model here
                # For now, we'll use a simple approach
            
            # Prepare input sequence
            input_seq = scaled_data[-lookback:].reshape(1, lookback, 1)
            
            # Generate predictions
            predictions = []
            for _ in range(periods):
                pred = model.predict(input_seq, verbose=0)
                predictions.append(pred[0, 0])
                
                # Update sequence with prediction
                input_seq = np.append(
                    input_seq[:, 1:, :],
                    pred.reshape(1, 1, 1),
                    axis=1
                )
            
            # Inverse transform predictions
            predictions_array = np.array(predictions).reshape(-1, 1)
            forecast_values = scaler.inverse_transform(predictions_array).flatten()
            
            # Calculate confidence intervals (using prediction variance)
            std = np.std(forecast_values) * 0.15
            
            last_date = historical_data['Date'].max()
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='MS'
            )
            
            return {
                'forecast_values': [int(max(v, 0)) for v in forecast_values],
                'lower_bound': [int(max(v - 1.96*std, 0)) for v in forecast_values],
                'upper_bound': [int(max(v + 1.96*std, 0)) for v in forecast_values],
                'forecast_dates': [d.strftime('%b-%Y') for d in forecast_dates],
                'model_type': 'LSTM',
                'sequence_length': lookback
            }
            
        except Exception as e:
            print(f"LSTM forecasting error: {e}")
            return self._exponential_smoothing_forecast(historical_data, metric, periods)
    
    def _build_lstm_model(self, lookback: int):
        """Build simple LSTM model architecture"""
        
        model = self.tf.keras.Sequential([
            self.tf.keras.layers.LSTM(50, activation='relu', 
                                     return_sequences=True,
                                     input_shape=(lookback, 1)),
            self.tf.keras.layers.Dropout(0.2),
            self.tf.keras.layers.LSTM(50, activation='relu'),
            self.tf.keras.layers.Dropout(0.2),
            self.tf.keras.layers.Dense(25, activation='relu'),
            self.tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def _exponential_smoothing_forecast(self, historical_data: pd.DataFrame,
                                       metric: str, periods: int) -> Dict:
        """Exponential smoothing as fallback"""
        
        values = historical_data[metric].values
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter
        smoothed = [values[0]]
        
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
        
        # Forecast
        last_smoothed = smoothed[-1]
        trend = (values[-1] - values[-6]) / 6
        
        forecast_values = []
        for i in range(periods):
            forecast_values.append(int(max(last_smoothed + trend * (i + 1), 0)))
        
        last_date = historical_data['Date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='MS'
        )
        
        return {
            'forecast_values': forecast_values,
            'lower_bound': [int(v * 0.9) for v in forecast_values],
            'upper_bound': [int(v * 1.1) for v in forecast_values],
            'forecast_dates': [d.strftime('%b-%Y') for d in forecast_dates],
            'model_type': 'Exponential Smoothing (Fallback)'
        }

# ============================================================================
# OPTION 4: ENSEMBLE FORECASTER
# ============================================================================

class EnsembleForecaster:
    """
    Combine multiple forecasting models for improved accuracy
    
    Pros:
    - More robust than single models
    - Reduces prediction variance
    - Can handle different data patterns
    
    Cons:
    - More complex to maintain
    - Slower computation
    """
    
    def __init__(self, models: list = None):
        """
        Initialize ensemble forecaster
        
        Parameters:
        -----------
        models : list, optional
            List of forecaster instances to ensemble
        """
        if models is None:
            # Default: Use ARIMA and Prophet
            self.models = []
            try:
                self.models.append(ARIMAForecaster())
            except ImportError:
                pass
            
            try:
                self.models.append(ProphetForecaster())
            except ImportError:
                pass
        else:
            self.models = models
    
    def forecast(self, product_name: str, historical_data: pd.DataFrame,
                metric: str = 'Quantity_Sold',
                periods: int = 3,
                weights: Optional[list] = None) -> Dict:
        """
        Generate ensemble forecast by combining multiple models
        
        Parameters:
        -----------
        weights : list, optional
            Weights for each model. If None, uses equal weights.
        """
        
        if not self.models:
            raise ValueError("No models available for ensemble")
        
        # Get forecasts from each model
        forecasts = []
        successful_models = []
        
        for model in self.models:
            try:
                forecast = model.forecast(product_name, historical_data, metric, periods)
                forecasts.append(forecast)
                successful_models.append(model.__class__.__name__)
            except Exception as e:
                print(f"Model {model.__class__.__name__} failed: {e}")
                continue
        
        if not forecasts:
            # Fallback if all models fail
            return self._simple_average_forecast(historical_data, metric, periods)
        
        # Set weights (equal if not provided)
        if weights is None:
            weights = [1.0 / len(forecasts)] * len(forecasts)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Combine forecasts
        ensemble_values = np.zeros(periods)
        ensemble_lower = np.zeros(periods)
        ensemble_upper = np.zeros(periods)
        
        for forecast, weight in zip(forecasts, weights):
            ensemble_values += np.array(forecast['forecast_values']) * weight
            ensemble_lower += np.array(forecast['lower_bound']) * weight
            ensemble_upper += np.array(forecast['upper_bound']) * weight
        
        return {
            'forecast_values': [int(v) for v in ensemble_values],
            'lower_bound': [int(v) for v in ensemble_lower],
            'upper_bound': [int(v) for v in ensemble_upper],
            'forecast_dates': forecasts[0]['forecast_dates'],
            'model_type': f'Ensemble ({", ".join(successful_models)})',
            'weights': dict(zip(successful_models, weights))
        }
    
    def _simple_average_forecast(self, historical_data: pd.DataFrame,
                                 metric: str, periods: int) -> Dict:
        """Simple moving average as ultimate fallback"""
        
        avg = historical_data.tail(6)[metric].mean()
        
        last_date = historical_data['Date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='MS'
        )
        
        return {
            'forecast_values': [int(avg)] * periods,
            'lower_bound': [int(avg * 0.85)] * periods,
            'upper_bound': [int(avg * 1.15)] * periods,
            'forecast_dates': [d.strftime('%b-%Y') for d in forecast_dates],
            'model_type': 'Simple Average (Fallback)'
        }

# ============================================================================
# INTEGRATION WITH DASHBOARD
# ============================================================================

def get_ml_forecast(product_name: str, historical_data: pd.DataFrame,
                   metric: str = 'Quantity_Sold',
                   model_type: str = 'auto') -> Dict:
    """
    Main function to integrate with dashboard
    
    Replace the placeholder function in manufacturing_dashboard.py with this
    
    Parameters:
    -----------
    product_name : str
        Product identifier
    historical_data : pd.DataFrame
        Historical sales data
    metric : str
        'Quantity_Sold' or 'Sales_Value'
    model_type : str
        'arima', 'prophet', 'lstm', 'ensemble', or 'auto'
        'auto' will try models in order and use first successful one
        
    Returns:
    --------
    Dict with forecast_values, lower_bound, upper_bound, forecast_dates
    """
    
    if model_type == 'auto':
        # Try models in order of preference
        try:
            forecaster = ProphetForecaster()
            return forecaster.forecast(product_name, historical_data, metric)
        except:
            pass
        
        try:
            forecaster = ARIMAForecaster()
            return forecaster.forecast(product_name, historical_data, metric)
        except:
            pass
        
        # Ultimate fallback
        return _fallback_forecast(historical_data, metric)
    
    elif model_type == 'arima':
        forecaster = ARIMAForecaster()
        return forecaster.forecast(product_name, historical_data, metric)
    
    elif model_type == 'prophet':
        forecaster = ProphetForecaster()
        return forecaster.forecast(product_name, historical_data, metric)
    
    elif model_type == 'lstm':
        forecaster = LSTMForecaster()
        return forecaster.forecast(product_name, historical_data, metric)
    
    elif model_type == 'ensemble':
        forecaster = EnsembleForecaster()
        return forecaster.forecast(product_name, historical_data, metric)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def _fallback_forecast(historical_data: pd.DataFrame, metric: str) -> Dict:
    """Ultimate fallback using moving average"""
    
    avg = historical_data.tail(6)[metric].mean()
    std = historical_data.tail(6)[metric].std()
    
    last_date = historical_data['Date'].max()
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=3,
        freq='MS'
    )
    
    return {
        'forecast_values': [int(avg)] * 3,
        'lower_bound': [int(avg - 1.96 * std)] * 3,
        'upper_bound': [int(avg + 1.96 * std)] * 3,
        'forecast_dates': [d.strftime('%b-%Y') for d in forecast_dates],
        'model_type': 'Moving Average (Ultimate Fallback)'
    }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: How to use the forecasters
    """
    
    import os
    
    # Sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')
    data = {
        'Date': dates,
        'Product_Name': ['Product A'] * len(dates),
        'Quantity_Sold': [100 + i*5 + np.random.randint(-10, 10) 
                         for i in range(len(dates))],
        'Sales_Value': [5000 + i*200 + np.random.randint(-500, 500) 
                       for i in range(len(dates))]
    }
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("ML Forecasting Examples")
    print("=" * 60)
    
    # Test each model
    models_to_test = ['prophet', 'arima', 'ensemble']
    
    for model_type in models_to_test:
        try:
            print(f"\nTesting {model_type.upper()} Model:")
            print("-" * 60)
            
            forecast = get_ml_forecast('Product A', df, 'Quantity_Sold', model_type)
            
            print(f"Model Type: {forecast.get('model_type', 'Unknown')}")
            print("\nForecast Results:")
            for date, value, lower, upper in zip(
                forecast['forecast_dates'],
                forecast['forecast_values'],
                forecast['lower_bound'],
                forecast['upper_bound']
            ):
                print(f"  {date}: {value:,} (CI: {lower:,} - {upper:,})")
            
        except ImportError as e:
            print(f"  Skipped - {e}")
        except Exception as e:
            print(f"  Error - {e}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
