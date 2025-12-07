import argparse
import sys
import os
# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    # Configure GPU memory growth to prevent allocation errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] GPU(s) Detected: {len(gpus)} device(s). Using GPU for training.")
        except RuntimeError as e:
            print(f"[WARN] GPU error: {e}")
    else:
        print("[INFO] No GPU detected. Falling back to CPU. (Make sure CUDA/cuDNN are installed)")
except ImportError:
    print("\n[ERROR] TensorFlow is not installed or failed to import.")
    print("Please install it manually: pip install tensorflow")
    print("If on Windows, you may need to enable Long Paths or use a virtual environment.")
    import sys
    sys.exit(1)

from src.data_loader import DataLoader
from src.hybrid_model import HybridModel
from src.utils import calculate_metrics, plot_results, get_steps_for_days
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="XAUUSD Prediction Bot")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='Mode: train or predict')
    parser.add_argument('--interval', type=str, default='1h', help='Timeframe: 1h, 30m, 15m')
    parser.add_argument('--period', type=str, default='2y', help='Data period to fetch')
    parser.add_argument('--days', type=int, default=0, help='Days to predict into the future (e.g. 5)')
    
    args = parser.parse_args()
    
    print(f"Initializing DataLoader for {args.interval}...")
    # Defaulting to XAUUSD for MT5. If using yfinance fallback, loader handles GC=F mapping.
    loader = DataLoader(symbol='XAUUSD', interval=args.interval, period=args.period, data_path=f'data/gold_{args.interval}.csv')
    df = loader.fetch_data(source='mt5')
    
    # Preprocess
    look_back = 60
    X, y, scaled_data = loader.prepare_data_for_lstm(df, look_back=look_back)
    
    # Check if we have enough data
    if len(X) == 0:
        print("[ERROR] Not enough data to create sequences. Try a longer --period.")
        sys.exit(1)

    # Input shape: (look_back, 1)
    input_shape = (X.shape[1], X.shape[2])
    
    model = HybridModel(input_shape)
    
    # In 'train' mode, reasonable to perform a train-test split for validation
    # If users wants pure future prediction, better to retrain on ALL data.
    # For now, let's keep 80/20 split for validation, then if --days > 0, predict future from end of ALL data.
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    if args.mode == 'train':
        print(f"Starting training on {len(X_train)} samples...")
        model.train(X_train, y_train, epochs=args.epochs)
        
        # Evaluate
        print("Evaluating on Test set...")
        final_preds, lstm_preds, arima_preds = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, final_preds)
        print("Hybrid Metrics:", metrics)
        
        plot_results(y_test, lstm_preds, final_preds, title=f"Validation_{args.interval}")
        
        # Future Prediction
        if args.days > 0:
            # from src.utils import get_steps_for_days
            steps = get_steps_for_days(args.interval, args.days)
            
            # Use the VERY LAST sequence available in the entire dataset to predict forward
            # scaled_data shape: (Total, 1)
            # We need the last 'look_back' points
            last_sequence = scaled_data[-look_back:] 
            # Reshape to (1, look_back, 1)
            last_sequence = last_sequence.reshape(1, look_back, 1)
            
            future_final, future_lstm, future_arima = model.predict_future(last_sequence, steps)
            
            # Inverse transform to get actual prices
            future_prices = loader.inverse_transform(future_final)
            
            # Print/Save
            print(f"\n--- Future Prediction for next {args.days} days ({steps} steps) ---")
            print(f"Last Known Price: {loader.inverse_transform(y[-1].reshape(-1,1))[0,0]:.2f}")
            print(f"Next 5 Steps: {future_prices[:5].flatten()}")
            print(f"Final Predicted Price: {future_prices[-1,0]:.2f}")
            
            # Plot Future
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(future_prices)), future_prices, label='Future Forecast', color='green')
            plt.title(f"Future {args.days} Days Prediction ({args.interval})")
            plt.xlabel("Steps Ahead")
            plt.ylabel("Price")
            plt.legend()
            
            import os
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f"plots/Future_{args.days}days_{args.interval}.png")
            print(f"Future plot saved to plots/Future_{args.days}days_{args.interval}.png")
            
            # Save CSV
            import pandas as pd
            future_df = pd.DataFrame(future_prices, columns=['Predicted_Close'])
            future_df.to_csv(f"data/future_pred_{args.interval}.csv", index_label='Step')
            print(f"Future values saved to data/future_pred_{args.interval}.csv")

    elif args.mode == 'predict':
        print("Prediction mode not fully implemented with saved models yet.")
        # Load logic would go here
        pass

if __name__ == "__main__":
    main()
