
import numpy as np
import pickle
import os
import tensorflow as tf
from src.data_loader import DataLoader
from src.hybrid_bigru_model import HybridBiGRUModel

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_hybrid_model(input_shape, model_dir='models'):
    """
    Load the trained Hybrid Bi-GRU + ARIMA model.
    """
    # Initialize model structure
    model = HybridBiGRUModel(input_shape=input_shape)
    
    # Load Bi-GRU Keras model
    bigru_path = os.path.join(model_dir, 'bigru_final.keras')
    if os.path.exists(bigru_path):
        print(f"Loading Bi-GRU from {bigru_path}...")
        model.bigru.model = tf.keras.models.load_model(bigru_path)
    else:
        raise FileNotFoundError(f"Bi-GRU model not found at {bigru_path}")
        
    # Load ARIMA model
    arima_path = os.path.join(model_dir, 'arima_model.pkl')
    if os.path.exists(arima_path):
        print(f"Loading ARIMA from {arima_path}...")
        with open(arima_path, 'rb') as f:
            model.arima.model_fit = pickle.load(f)
    else:
        raise FileNotFoundError(f"ARIMA model not found at {arima_path}")
        
    return model

def main():
    print("=== XAUUSD Bi-GRU + ARIMA Inference ===")
    
    # Configuration
    SYMBOL = 'XAUUSD'
    TIMEFRAME = '15m'
    PERIOD = '60d' # Load enough data for lookback
    LOOK_BACK = 20
    
    # 1. Load Data
    print("\nFetching latest data...")
    loader = DataLoader(symbol=SYMBOL, interval=TIMEFRAME, period=PERIOD)
    df = loader.fetch_data(source='auto')
    
    if df is None or df.empty:
        print("Error: Could not fetch data.")
        return

    # 2. Preprocess Data
    # We need the last LOOK_BACK steps + scaling
    # Note: We must use the SAME scaler parameters as training. 
    # Ideally scaler should be saved, but for now we refit on recent data 
    # assuming the distribution hasn't drifted wildly or we use a large enough window.
    # TODO: Save/Load scaler object for production robustness.
    
    print("Preprocessing...")
    scaled_data = loader.scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    if len(scaled_data) < LOOK_BACK:
        print(f"Error: Not enough data points. Need {LOOK_BACK}, got {len(scaled_data)}")
        return
        
    # Prepare input sequence (last 20 steps)
    last_sequence = scaled_data[-LOOK_BACK:]
    last_sequence = last_sequence.reshape(1, LOOK_BACK, 1)
    
    # 3. Load Model
    try:
        model = load_hybrid_model(input_shape=(LOOK_BACK, 1))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Predict Next Step
    print("\nPredicting next price...")
    prediction_scaled, _, _ = model.predict(last_sequence)
    
    # Inverse transform
    prediction_price = loader.scaler.inverse_transform(prediction_scaled)[0][0]
    current_price = df['Close'].iloc[-1]
    
    print("\n" + "="*30)
    print(f"Current Price:   ${current_price:.2f}")
    print(f"Predicted Price: ${prediction_price:.2f}")
    
    direction = "UP" if prediction_price > current_price else "DOWN"
    print(f"Direction:       {direction}")
    print("="*30)

if __name__ == "__main__":
    main()
