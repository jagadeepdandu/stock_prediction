import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_stock_data(days=252):  # 252 trading days in a year
   
    np.random.seed(42)
    
    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    # Generate price data with trend and volatility
    initial_price = 100
    daily_returns = np.random.normal(loc=0.0005, scale=0.02, size=days)  # Mean return 0.05% with 2% volatility
    price_multiplier = np.exp(np.cumsum(daily_returns))
    prices = initial_price * price_multiplier
    
    # Generate volume data
    volume = np.random.lognormal(mean=12, sigma=0.5, size=days)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volume
    })
    df.set_index('Date', inplace=True)
    
    return df

def prepare_data(df):
   
    print("Preparing data...")
    
    # Create features
    df['Returns'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Create target (next day's closing price)
    df['Target'] = df['Close'].shift(-1)
    
    # Drop NaN values
    df = df.dropna()
    
    # Select features
    features = ['Close', 'Volume', 'Returns', 'MA5', 'MA20', 'Volatility']
    X = df[features]
    y = df['Target']
    
    return X, y, df

def train_model(X, y):
    """Train the Random Forest model"""
    print("Training model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    return model, scaler, X_test, y_test, y_pred

def plot_predictions(y_test, y_pred, df):
    """Plot actual vs predicted values and stock price history"""
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Stock Prices')
    
    # Plot 2: Stock Price History with Moving Averages
    plt.subplot(1, 2, 2)
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['MA5'], label='5-day MA')
    plt.plot(df.index, df['MA20'], label='20-day MA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price History')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('stock_analysis.png')
    print("\nAnalysis plots have been saved to 'stock_analysis.png'")
    plt.close()

def main():
    # Generate synthetic data
    print("Generating synthetic stock data...")
    df = generate_stock_data()
    
    # Prepare data
    X, y, df = prepare_data(df)
    
    # Train model and get predictions
    model, scaler, X_test, y_test, y_pred = train_model(X, y)
    
    # Plot results
    plot_predictions(y_test, y_pred, df)
    
    # Make future prediction
    last_data = X.iloc[-1:].copy()
    last_data_scaled = scaler.transform(last_data)
    future_price = model.predict(last_data_scaled)[0]
    
    print(f"\nCurrent Price: ${X['Close'].iloc[-1]:.2f}")
    print(f"Predicted Next Price: ${future_price:.2f}")
    print(f"Predicted Change: {((future_price - X['Close'].iloc[-1]) / X['Close'].iloc[-1] * 100):.2f}%")

if __name__ == "__main__":
    main()
