# Hybrid ML Sales & Inventory Forecastor

A Streamlit-based dashboard for business trend analysis, sales forecasting, and inventory management in manufacturing environments.

## Overview

This project provides an interactive analytics dashboard that helps businesses:

- Analyze sales trends
- Forecast future demand using machine learning models
- Monitor inventory levels
- Calculate reorder points
- Generate business insights and risk alerts
- Support executive decision-making with visual reports

## Features

### Sales Trend Analysis
- Historical sales visualization
- Revenue and quantity trend tracking
- Product-wise performance analysis

### Demand Forecasting
Supports multiple forecasting approaches:

- ARIMA / SARIMA
- Prophet
- Random Forest
- XGBoost
- LSTM (Extensible)

### Inventory Intelligence
- Stock monitoring
- Reorder point calculations
- Safety stock recommendations
- Demand-risk analysis

### Interactive Dashboard
- Built with Streamlit
- Plotly visualizations
- Responsive user interface
- Dark/Light mode support

---

## Project Structure

```text
HybridMLSales-InventoryForecastor-main/
│
├── manufacturing_dashboard.py      # Main Streamlit dashboard
├── processed_tc_strips_data.csv    # Processed dataset
├── style.css                       # Dashboard styling
├── requirements.txt                # Dependencies
│
└── tools/
    ├── data_loader.py
    ├── process_acquired_data.py
    ├── validate_data.py
    ├── ml_forecasting.py
    └── sales_inventory_dataset.csv
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/HybridMLSales-InventoryForecastor.git
cd HybridMLSales-InventoryForecastor
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/Mac**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

```bash
streamlit run manufacturing_dashboard.py
```

The dashboard will open automatically in your browser.

---

## Data Requirements

The dashboard expects one of the following files:

- `processed_tc_strips_data.csv` (recommended)
- `sales_inventory_dataset.csv`
- `AcquiredData.xlsx`

Required fields include:

- Date
- Product Name
- Quantity Sold
- Sales Value
- Inventory Level

---

## Forecasting Models

### Statistical Models
- ARIMA
- SARIMA

### Machine Learning Models
- Random Forest
- XGBoost

### Deep Learning
- LSTM (optional integration)

The forecasting framework is implemented in:

```text
tools/ml_forecasting.py
```

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-Learn
- StatsModels
- Prophet (optional)
- TensorFlow (optional)

---

## Business Applications

- Manufacturing Planning
- Demand Forecasting
- Inventory Optimization
- Supply Chain Management
- Executive Reporting

---

## Future Enhancements

- Real-time database integration
- Automated report generation
- Multi-location inventory management
- Advanced ML model comparison
- Cloud deployment support

---

## Author

Analytics Team

Version: 2.0 – ML Integration

---

## License

This project is intended for educational and research purposes.
