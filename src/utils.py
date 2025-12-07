import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics as per the LSTM-ARIMA hybrid model paper.
    Metrics: MAE, MSE, RMSE, MAPE, SMAPE, R², DA (Directional Accuracy), Max Error
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE - Mean Absolute Percentage Error (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # SMAPE - Symmetric Mean Absolute Percentage Error
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    
    # DA - Directional Accuracy (% of correct direction predictions)
    if len(y_true) > 1:
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        da = np.mean(actual_direction == pred_direction) * 100
    else:
        da = 0.0
    
    # Max Error
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Accuracy (100 - MAPE)
    accuracy = 100 - mape
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "SMAPE": smape,
        "R2": r2,
        "DA": da,
        "Max_Error": max_error,
        "Accuracy": accuracy
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
