import numpy as np
from src.bigru_model import BiGRUModel
from src.arima_model import ARIMAModel


class HybridBiGRUModel:
    """
    Hybrid Bi-GRU + ARIMA model for gold price prediction.
    
    Bi-GRU captures nonlinear patterns, ARIMA corrects residual linear trends.
    Based on paper methodology with Bi-GRU improvements.
    """
    
    def __init__(self, input_shape, arima_order=(5, 1, 0), 
                 units=64, dropout_rate=0.2, learning_rate=0.0001):
        """
        Initialize Hybrid model.
        
        Args:
            input_shape: (time_steps, features)
            arima_order: ARIMA (p, d, q) order
            units: Bi-GRU units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.bigru = BiGRUModel(input_shape, units, dropout_rate, learning_rate)
        self.arima = ARIMAModel(arima_order)
        self.look_back = input_shape[0]
        
    def train(self, X_train, y_train, epochs=100, batch_size=8):
        """
        Train hybrid model: Bi-GRU first, then ARIMA on residuals.
        """
        print("Training Bi-GRU model...")
        self.bigru.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        print("\nGenerating residuals for ARIMA...")
        bigru_preds = self.bigru.predict(X_train)
        
        y_flat = y_train.flatten()
        preds_flat = bigru_preds.flatten()
        residuals = y_flat - preds_flat
        
        print("Training ARIMA on residuals...")
        self.arima.train(residuals)
        print("Hybrid Bi-GRU + ARIMA training complete!")
        
    def predict(self, X_sample):
        """
        Make predictions combining Bi-GRU and ARIMA.
        
        Returns: (final_pred, bigru_pred, arima_pred)
        """
        bigru_pred = self.bigru.predict(X_sample)
        arima_pred = self.arima.predict_sequence(steps=len(X_sample)).reshape(-1, 1)
        
        final_pred = bigru_pred + arima_pred
        return final_pred, bigru_pred, arima_pred
    
    def predict_future(self, initial_sequence, steps):
        """
        Recursive future prediction.
        
        Args:
            initial_sequence: (1, look_back, 1) - last available data window
            steps: Number of steps to predict ahead
            
        Returns: (final_future, bigru_future, arima_future)
        """
        print(f"Generating {steps}-step future prediction...")
        curr_seq = initial_sequence.copy()
        bigru_future = []
        
        for i in range(steps):
            pred = self.bigru.model.predict(curr_seq, verbose=0)
            val = pred[0, 0]
            bigru_future.append(val)
            
            # Update sequence
            new_step = np.array([[[val]]])
            curr_seq = np.concatenate([curr_seq[:, 1:, :], new_step], axis=1)
        
        bigru_future = np.array(bigru_future).reshape(-1, 1)
        arima_future = self.arima.predict_sequence(steps=steps).reshape(-1, 1)
        
        final_future = bigru_future + arima_future
        return final_future, bigru_future, arima_future
