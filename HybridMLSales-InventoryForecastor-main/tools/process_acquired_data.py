"""
AcquiredData.xlsx Processing Script

This script processes the TC STRIPS inventory data from multiple Excel sheets
and converts it into a unified format for the dashboard.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clean_price(price_str):
    """Convert price string like '₹ 1,078.00' or '₹1078' to float"""
    if pd.isna(price_str) or price_str == '-' or price_str == '':
        return 0.0
    
    # Remove currency symbol and commas
    cleaned = str(price_str).replace('₹', '').replace(',', '').strip()
    
    try:
        return float(cleaned)
    except:
        return 0.0

def extract_product_dimensions(sheet_name):
    """Extract product dimensions from sheet name"""
    # Example: "2.3 X 3.3 330" or "3.3 x 4.3 x 330 "
    return sheet_name.strip()

def process_excel_file(filepath):
    """
    Process all sheets in the Excel file and create unified dataset
    
    Returns:
        pd.DataFrame with columns: Date, Product_Name, Transaction_Type, 
                                   Quantity, Unit_Price, Sales_Value
    """
    
    # Load Excel file
    xl = pd.ExcelFile(filepath)
    
    all_data = []
    
    for sheet_name in xl.sheet_names:
        print(f"{Colors.OKBLUE}Processing sheet:{Colors.ENDC} {Colors.BOLD}{sheet_name}{Colors.ENDC}")
        
        # Read sheet
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        
        # Skip first row (header) and get data
        df_clean = df.iloc[1:].copy()
        
        # Rename columns based on structure
        if df.shape[1] == 5:
            df_clean.columns = ['Serial', 'Type', 'Date', 'Quantity', 'Price']
        elif df.shape[1] == 4:
            # Some sheets missing price column
            df_clean.columns = ['Serial', 'Type', 'Date', 'Quantity']
            df_clean['Price'] = 0
        else:
            print(f"  {Colors.WARNING}Warning:{Colors.ENDC} Unexpected column count ({df.shape[1]}) in {sheet_name}")
            continue
        
        # Filter out invalid rows
        df_clean = df_clean[df_clean['Type'].notna()].copy()
        df_clean = df_clean[df_clean['Type'] != ''].copy()
        
        # Convert date
        try:
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d/%m/%Y', errors='coerce')
        except:
            try:
                df_clean['Date'] = pd.to_datetime(df_clean['Date'], dayfirst=True, errors='coerce')
            except:
                print(f"  {Colors.WARNING}Warning:{Colors.ENDC} Could not parse dates in {sheet_name}")
                continue
        
        # Remove rows with invalid dates
        df_clean = df_clean[df_clean['Date'].notna()].copy()
        
        # Convert quantity
        df_clean['Quantity'] = pd.to_numeric(df_clean['Quantity'], errors='coerce').fillna(0).astype(int)
        
        # Convert price
        df_clean['Price'] = df_clean['Price'].apply(clean_price)
        
        # Add product name
        product_name = extract_product_dimensions(sheet_name)
        df_clean['Product_Name'] = product_name
        
        # Clean up type
        df_clean['Type'] = df_clean['Type'].str.strip()
        
        # Calculate sales value
        df_clean['Sales_Value'] = df_clean['Quantity'] * df_clean['Price']
        
        # Keep only relevant columns
        df_result = df_clean[['Date', 'Product_Name', 'Type', 'Quantity', 'Price', 'Sales_Value']].copy()
        df_result = df_result.rename(columns={'Type': 'Transaction_Type', 'Price': 'Unit_Price'})
        
        all_data.append(df_result)
        
        print(f"  {Colors.OKCYAN}→ Processed {len(df_result)} records{Colors.ENDC}")
    
    # Combine all sheets
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    print(f"\n{Colors.OKGREEN}✓ Total records processed:{Colors.ENDC} {Colors.BOLD}{len(combined_df)}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✓ Date range:{Colors.ENDC} {combined_df['Date'].min().strftime('%Y-%m-%d')} {Colors.OKCYAN}to{Colors.ENDC} {combined_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"{Colors.OKGREEN}✓ Products:{Colors.ENDC} {Colors.BOLD}{combined_df['Product_Name'].nunique()}{Colors.ENDC}")
    
    return combined_df

def aggregate_to_monthly(df):
    """
    Aggregate transaction data to monthly format for dashboard
    
    Returns:
        pd.DataFrame with monthly aggregated data
    """
    
    # Filter only sales transactions
    df_sales = df[df['Transaction_Type'].str.contains('Sale', case=False, na=False)].copy()
    
    # Group by product and month
    df_monthly = df_sales.groupby([
        pd.Grouper(key='Date', freq='MS'),
        'Product_Name'
    ]).agg({
        'Quantity': 'sum',
        'Sales_Value': 'sum',
        'Unit_Price': 'mean'  # Average price for the month
    }).reset_index()
    
    # Rename for dashboard compatibility
    df_monthly = df_monthly.rename(columns={
        'Quantity': 'Quantity_Sold'
    })
    
    # Add Month_Year for display
    df_monthly['Month_Year'] = df_monthly['Date'].dt.strftime('%b-%Y')
    
    # Add product category based on dimensions
    def categorize_product(product_name):
        """Categorize based on size"""
        # Extract first dimension (width)
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
    
    # Create display name
    df_monthly['Product_Display'] = df_monthly['Product_Name'] + ' mm'
    
    return df_monthly

def calculate_current_inventory(df):
    """
    Calculate current inventory for each product based on transactions
    
    Returns:
        dict of product_name: current_inventory
    """
    
    inventory = {}
    
    for product in df['Product_Name'].unique():
        df_product = df[df['Product_Name'] == product].sort_values('Date')
        
        stock = 0
        
        for _, row in df_product.iterrows():
            trans_type = row['Transaction_Type'].lower()
            qty = row['Quantity']
            
            if 'sale' in trans_type or 'reduce' in trans_type:
                stock -= qty
            elif 'purchase' in trans_type or 'add' in trans_type or 'opening' in trans_type:
                stock += qty
        
        # Ensure non-negative
        inventory[product] = max(stock, 0)
    
    return inventory

if __name__ == "__main__":
    """
    Run this script to process the Excel file and create a CSV
    """
    
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}🚀 PROCESSING ACQUIREDDATA.XLSX{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}")
    
    # Process the Excel file
    df_transactions = process_excel_file('AcquiredData.xlsx')
    
    # Aggregate to monthly
    df_monthly = aggregate_to_monthly(df_transactions)
    
    # Calculate current inventory
    inventory = calculate_current_inventory(df_transactions)
    
    # Add inventory to monthly data (use last known value for each month)
    def get_inventory_for_row(row):
        return inventory.get(row['Product_Name'], 0)
    
    df_monthly['current_inventory'] = df_monthly.apply(get_inventory_for_row, axis=1)
    
    # Save to CSV
    output_file = 'processed_tc_strips_data.csv'
    df_monthly.to_csv(output_file, index=False)
    
    print(f"\n{Colors.OKCYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}✅ PROCESSING COMPLETE{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✓ Output file:{Colors.ENDC} {Colors.BOLD}'{output_file}'{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✓ Total monthly records:{Colors.ENDC} {Colors.BOLD}{len(df_monthly)}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✓ Products:{Colors.ENDC} {Colors.BOLD}{df_monthly['Product_Name'].nunique()}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✓ Date range:{Colors.ENDC} {Colors.BOLD}{df_monthly['Date'].min().strftime('%B %Y')} {Colors.OKCYAN}to{Colors.ENDC} {Colors.BOLD}{df_monthly['Date'].max().strftime('%B %Y')}{Colors.ENDC}")
    
    print(f"\n{Colors.OKCYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}📊 SUMMARY BY PRODUCT{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}")
    
    for product in sorted(df_monthly['Product_Name'].unique()):
        total_sales = df_monthly[df_monthly['Product_Name'] == product]['Quantity_Sold'].sum()
        total_value = df_monthly[df_monthly['Product_Name'] == product]['Sales_Value'].sum()
        stock = inventory.get(product, 0)
        
        print(f"\n{Colors.OKBLUE}{Colors.BOLD}{product}:{Colors.ENDC}")
        print(f"  {Colors.OKCYAN}Total Sales:{Colors.ENDC} {total_sales:,} units")
        print(f"  {Colors.OKGREEN}Total Value:{Colors.ENDC} ₹{total_value:,.2f}")
        print(f"  {Colors.WARNING}Current Stock:{Colors.ENDC} {stock:,} units")
    
    print(f"\n{Colors.OKCYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}📋 Sample of processed data:{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}")
    print(df_monthly.head(10))
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ Ready to use with dashboard!{Colors.ENDC}")
    print(f"  Place {Colors.OKCYAN}'{output_file}'{Colors.ENDC} in the same folder as {Colors.WARNING}manufacturing_dashboard.py{Colors.ENDC}")
