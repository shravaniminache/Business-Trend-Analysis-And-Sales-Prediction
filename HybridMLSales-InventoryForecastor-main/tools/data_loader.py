"""
Data Loading Module for Manufacturing Dashboard

This module demonstrates how to replace the sample data generation
with real data loading from various sources.

Usage:
    from data_loader import load_sales_data
    df = load_sales_data(source='csv', filepath='data/sales_data.csv')
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime

# ============================================================================
# CSV DATA LOADING
# ============================================================================

def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load sales data from CSV file
    
    Expected CSV format:
    Date,Product_Name,Quantity_Sold,Sales_Value
    2023-01-01,Carbide Drill Bit 10mm,245,12500
    2023-02-01,Carbide Drill Bit 10mm,267,13200
    ...
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame with columns: Date, Product_Name, Quantity_Sold, Sales_Value
    """
    
    df = pd.read_csv(filepath)
    
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add Month_Year column for display
    df['Month_Year'] = df['Date'].dt.strftime('%b-%Y')
    
    # Validate required columns
    required_cols = ['Date', 'Product_Name', 'Quantity_Sold', 'Sales_Value']
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

# ============================================================================
# DATABASE LOADING
# ============================================================================

def load_from_database(connection_string: str, 
                      query: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load sales data from database
    
    Parameters:
    -----------
    connection_string : str
        Database connection string
        Examples:
        - PostgreSQL: 'postgresql://user:pass@localhost:5432/manufacturing_db'
        - MySQL: 'mysql+pymysql://user:pass@localhost:3306/manufacturing_db'
        - SQLite: 'sqlite:///manufacturing.db'
    
    query : str, optional
        Custom SQL query. If None, uses default query.
    
    start_date : str, optional
        Filter start date (YYYY-MM-DD)
    
    end_date : str, optional
        Filter end date (YYYY-MM-DD)
        
    Returns:
    --------
    pd.DataFrame with sales data
    """
    
    try:
        from sqlalchemy import create_engine
    except ImportError:
        raise ImportError("SQLAlchemy not installed. Run: pip install sqlalchemy")
    
    # Create database engine
    engine = create_engine(connection_string)
    
    # Default query if none provided
    if query is None:
        query = """
        SELECT 
            date_column as Date,
            product_name as Product_Name,
            quantity_sold as Quantity_Sold,
            sales_value as Sales_Value
        FROM sales_transactions
        WHERE 1=1
        """
        
        if start_date:
            query += f" AND date_column >= '{start_date}'"
        if end_date:
            query += f" AND date_column <= '{end_date}'"
        
        query += " ORDER BY date_column, product_name"
    
    # Load data
    df = pd.read_sql(query, engine)
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month_Year'] = df['Date'].dt.strftime('%b-%Y')
    
    return df

# ============================================================================
# EXCEL LOADING
# ============================================================================

def load_from_excel(filepath: str, sheet_name: str = 'Sheet1') -> pd.DataFrame:
    """
    Load sales data from Excel file
    
    Parameters:
    -----------
    filepath : str
        Path to Excel file (.xlsx or .xls)
    
    sheet_name : str
        Name of the sheet containing data
        
    Returns:
    --------
    pd.DataFrame with sales data
    """
    
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except ImportError:
        raise ImportError("openpyxl not installed. Run: pip install openpyxl")
    
    # Data processing
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month_Year'] = df['Date'].dt.strftime('%b-%Y')
    
    return df

# ============================================================================
# API LOADING
# ============================================================================

def load_from_api(api_url: str, 
                 headers: Optional[Dict] = None,
                 params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Load sales data from REST API
    
    Parameters:
    -----------
    api_url : str
        API endpoint URL
    
    headers : dict, optional
        HTTP headers (e.g., authentication tokens)
    
    params : dict, optional
        Query parameters
        
    Returns:
    --------
    pd.DataFrame with sales data
    """
    
    import requests
    
    response = requests.get(api_url, headers=headers, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    # Assuming API returns list of records
    df = pd.DataFrame(data['records'])  # Adjust based on API structure
    
    # Data processing
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month_Year'] = df['Date'].dt.strftime('%b-%Y')
    
    return df

# ============================================================================
# MAIN LOADING FUNCTION
# ============================================================================

def load_sales_data(source: str = 'csv', **kwargs) -> pd.DataFrame:
    """
    Unified data loading function
    
    Parameters:
    -----------
    source : str
        Data source type: 'csv', 'database', 'excel', or 'api'
    
    **kwargs : 
        Source-specific parameters
        
    Returns:
    --------
    pd.DataFrame with standardized sales data
    
    Examples:
    ---------
    # Load from CSV
    df = load_sales_data(source='csv', filepath='data/sales.csv')
    
    # Load from database
    df = load_sales_data(
        source='database',
        connection_string='postgresql://user:pass@localhost/db',
        start_date='2023-01-01'
    )
    
    # Load from Excel
    df = load_sales_data(source='excel', filepath='data/sales.xlsx')
    """
    
    if source == 'csv':
        return load_from_csv(**kwargs)
    elif source == 'database':
        return load_from_database(**kwargs)
    elif source == 'excel':
        return load_from_excel(**kwargs)
    elif source == 'api':
        return load_from_api(**kwargs)
    else:
        raise ValueError(f"Unknown source: {source}")

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_sales_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate loaded sales data and return quality metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Sales data to validate
        
    Returns:
    --------
    Dict with validation results and quality metrics
    """
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'metrics': {}
    }
    
    # Check required columns
    required_cols = ['Date', 'Product_Name', 'Quantity_Sold', 'Sales_Value']
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Missing columns: {missing_cols}")
        return validation_results
    
    # Check for null values
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        validation_results['warnings'].append(
            f"Null values found: {null_counts[null_counts > 0].to_dict()}"
        )
    
    # Check for negative values
    if (df['Quantity_Sold'] < 0).any():
        validation_results['warnings'].append("Negative quantities found")
    
    if (df['Sales_Value'] < 0).any():
        validation_results['warnings'].append("Negative sales values found")
    
    # Calculate quality metrics
    validation_results['metrics'] = {
        'total_records': len(df),
        'unique_products': df['Product_Name'].nunique(),
        'date_range': (df['Date'].min(), df['Date'].max()),
        'total_quantity': df['Quantity_Sold'].sum(),
        'total_value': df['Sales_Value'].sum(),
        'avg_monthly_records': len(df) / df['Date'].nunique(),
        'completeness': (1 - df[required_cols].isnull().sum().sum() / 
                        (len(df) * len(required_cols))) * 100
    }
    
    return validation_results

# ============================================================================
# SAMPLE DATA GENERATOR (for testing)
# ============================================================================

def generate_sample_data(
    products: list = None,
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    frequency: str = 'MS'
) -> pd.DataFrame:
    """
    Generate sample sales data for testing
    
    Parameters:
    -----------
    products : list, optional
        List of product names. If None, uses default products.
    
    start_date : str
        Start date (YYYY-MM-DD)
    
    end_date : str
        End date (YYYY-MM-DD)
    
    frequency : str
        Date frequency ('MS' for month start, 'D' for daily)
        
    Returns:
    --------
    pd.DataFrame with sample sales data
    """
    
    if products is None:
        products = [
            "Carbide Drill Bit 10mm",
            "HSS Cutting Tool Set",
            "Carbon Steel End Mill",
            "Titanium Coated Drill",
            "Precision Boring Bar",
            "Threading Tap M8",
            "Reamer Tool 12mm",
            "Countersink Bit Set",
            "Indexable Insert CNMG",
            "Spiral Flute Tap"
        ]
    
    dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    data = []
    np.random.seed(42)
    
    for product in products:
        base_quantity = np.random.randint(100, 500)
        base_value = np.random.randint(5000, 25000)
        
        for date in dates:
            # Add seasonal variation and trend
            month_idx = date.month
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month_idx / 12)
            trend_factor = 1 + 0.05 * (dates.tolist().index(date) / len(dates))
            noise = np.random.normal(1, 0.1)
            
            quantity = int(base_quantity * seasonal_factor * trend_factor * noise)
            value = int(base_value * seasonal_factor * trend_factor * noise)
            
            data.append({
                'Date': date,
                'Product_Name': product,
                'Quantity_Sold': max(quantity, 0),
                'Sales_Value': max(value, 0)
            })
    
    df = pd.DataFrame(data)
    df['Month_Year'] = df['Date'].dt.strftime('%b-%Y')
    
    return df

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of data loading functions
    """
    
    # Example 1: Generate sample data
    print("Generating sample data...")
    df_sample = generate_sample_data()
    print(f"Generated {len(df_sample)} records")
    print(df_sample.head())
    
    # Validate data
    validation = validate_sales_data(df_sample)
    print(f"\nValidation Results:")
    print(f"Valid: {validation['valid']}")
    print(f"Metrics: {validation['metrics']}")
    
    # Example 2: Save sample data to CSV (for testing)
    # df_sample.to_csv('data/sales_data.csv', index=False)
    # print("\nSample data saved to data/sales_data.csv")
    
    # Example 3: Load from CSV (uncomment when you have real data)
    # df_csv = load_sales_data(source='csv', filepath='data/sales_data.csv')
    # print(f"\nLoaded {len(df_csv)} records from CSV")
    
    # Example 4: Load from database (uncomment when configured)
    # df_db = load_sales_data(
    #     source='database',
    #     connection_string='postgresql://user:pass@localhost/db',
    #     start_date='2023-01-01'
    # )
