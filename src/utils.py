import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        # "MAPE": mape,
        "R2": r2
    }

def plot_results(y_true, y_pred_lstm, y_pred_hybrid, title="Gold Price Prediction"):
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label="Actual", color='black')
    plt.plot(y_pred_lstm, label="LSTM Only", color='blue', linestyle='--')
    plt.plot(y_pred_hybrid, label="Hybrid LSTM-ARIMA", color='red', linestyle='-.')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price (Normalized)")
    plt.legend()
    import os
    os.makedirs('plots', exist_ok=True)
    filename = f"plots/{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

def get_steps_for_days(interval, days):
    """
    Calculate number of steps (candles) for a given number of days and timeframe.
    """
    # Hours per day = 24
    # Mins per day = 1440
    
    if interval == '1h':
        return days * 24
    elif interval == '30m':
        return days * 48
    elif interval == '15m':
        return days * 96
    elif interval == '1d':
        return days * 1
    elif interval == '4h':
        return days * 6
    else:
        # Default fallback: assume 1h if unknown or calculate roughly
        print(f"[WARN] Unknown interval {interval}. Defaulting to 24 steps/day (1h).")
        return days * 24
