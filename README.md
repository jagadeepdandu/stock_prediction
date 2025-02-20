# Stock Market Prediction Project

This project implements a machine learning model to predict stock market prices using historical data. It uses Random Forest Regression to make predictions based on various technical indicators.

## Features

- Downloads historical stock data using Yahoo Finance API
- Implements data preprocessing and feature engineering
- Trains a Random Forest model for prediction
- Evaluates model performance using MSE and R² metrics
- Visualizes actual vs predicted prices
- Makes next-day price predictions

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the prediction script:
```bash
python stock_predictor.py
```

## Data Features

The model uses the following features for prediction:
- Closing Price
- Trading Volume
- Daily Returns
- 5-day Moving Average
- 20-day Moving Average
- 20-day Volatility

## Output

The script will:
1. Download historical stock data (default: AAPL)
2. Train the model and display performance metrics
3. Generate a visualization of actual vs predicted prices
4. Provide a prediction for the next day's price
