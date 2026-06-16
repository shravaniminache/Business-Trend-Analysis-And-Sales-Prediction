"""
Data Validation Script for Sales Inventory Dataset

Run this to verify your dataset is loaded correctly
"""

import pandas as pd
import numpy as np

print("="*60)
print("SALES INVENTORY DATASET VALIDATION")
print("="*60)

# Load the dataset
try:
    df = pd.read_csv('sales_inventory_dataset.csv')
    print("✓ Dataset loaded successfully!")
    print(f"  Total records: {len(df):,}")
except FileNotFoundError:
    print("✗ File 'sales_inventory_dataset.csv' not found!")
    print("  Make sure the file is in the same folder as this script.")
    exit(1)

print("\n" + "-"*60)
print("DATASET STRUCTURE")
print("-"*60)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"\nColumn Names:")
for col in df.columns:
    print(f"  • {col}")

print("\n" + "-"*60)
print("PRODUCT INFORMATION")
print("-"*60)
print(f"Total Unique Products: {df['product_id'].nunique()}")
print(f"\nProducts:")
for product in sorted(df['product_id'].unique()):
    category = df[df['product_id'] == product]['product_category'].iloc[0]
    count = len(df[df['product_id'] == product])
    print(f"  • {product} ({category.replace('_', ' ').title()}) - {count} transactions")

print("\n" + "-"*60)
print("PRODUCT CATEGORIES")
print("-"*60)
for category in sorted(df['product_category'].unique()):
    count = len(df[df['product_category'] == category])
    products = df[df['product_category'] == category]['product_id'].nunique()
    print(f"  • {category.replace('_', ' ').title()}: {products} products, {count} transactions")

print("\n" + "-"*60)
print("DATE RANGE")
print("-"*60)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Start Date: {df['timestamp'].min().strftime('%B %d, %Y')}")
print(f"End Date: {df['timestamp'].max().strftime('%B %d, %Y')}")
print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
print(f"Total Months: {df['month'].nunique()}")

print("\n" + "-"*60)
print("TRANSACTION ANALYSIS")
print("-"*60)
print(f"Total Transactions: {len(df):,}")
print(f"\nBreakdown by Type:")
for trans_type, count in df['transaction_type'].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"  • {trans_type.title()}: {count:,} ({pct:.1f}%)")

print("\n" + "-"*60)
print("SALES METRICS (Sell Transactions Only)")
print("-"*60)
df_sales = df[df['transaction_type'] == 'sell'].copy()
df_sales['total_value'] = df_sales['quantity'] * df_sales['unit_price']

print(f"Total Quantity Sold: {df_sales['quantity'].sum():,} units")
print(f"Total Sales Value: ₹{df_sales['total_value'].sum():,.2f}")
print(f"Average Transaction Value: ₹{df_sales['total_value'].mean():,.2f}")
print(f"Average Quantity per Transaction: {df_sales['quantity'].mean():.1f} units")

print("\n" + "-"*60)
print("MONTHLY AGGREGATION PREVIEW")
print("-"*60)
df_monthly = df_sales.groupby([
    pd.Grouper(key='timestamp', freq='MS'),
    'product_id'
]).agg({
    'quantity': 'sum',
    'total_value': 'sum'
}).reset_index()

print(f"Total Monthly Records: {len(df_monthly):,}")
print(f"\nSample Monthly Data:")
print(df_monthly.head(10).to_string(index=False))

print("\n" + "-"*60)
print("INVENTORY LEVELS")
print("-"*60)
print("Current Inventory by Product:")
latest_inventory = df.groupby('product_id')['current_inventory'].last().sort_values(ascending=False)
for product, inventory in latest_inventory.items():
    category = df[df['product_id'] == product]['product_category'].iloc[0]
    print(f"  • {product} ({category.replace('_', ' ').title()}): {inventory:,} units")

print("\n" + "-"*60)
print("DATA QUALITY CHECK")
print("-"*60)
missing_values = df.isnull().sum()
if missing_values.any():
    print("⚠ Missing Values Found:")
    for col, count in missing_values[missing_values > 0].items():
        pct = (count / len(df)) * 100
        print(f"  • {col}: {count} ({pct:.1f}%)")
else:
    print("✓ No missing values in critical columns")

# Check for negative values
if (df_sales['quantity'] < 0).any():
    print("⚠ Negative quantities found")
else:
    print("✓ All quantities are positive")

if (df_sales['unit_price'] < 0).any():
    print("⚠ Negative prices found")
else:
    print("✓ All prices are positive")

print("\n" + "="*60)
print("VALIDATION COMPLETE!")
print("="*60)
print("\n✓ Your dataset is ready for the dashboard!")
print("  Run: streamlit run manufacturing_dashboard.py")
print("="*60)
