import numpy as np
from src.lstm_model import LSTMModel
from src.arima_model import ARIMAModel

class HybridModel:
    def __init__(self, input_shape, arima_order=(5, 1, 0)):
        """
        input_shape: (look_back, features) - supports multivariate
        """
        self.lstm = LSTMModel(input_shape)
        self.arima = ARIMAModel(arima_order)
        self.look_back = input_shape[0]
        self.n_features = input_shape[1]

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        print(f"Training LSTM part (input: {self.look_back} steps x {self.n_features} features)...")
        self.lstm.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        print("Model trained. Generating residuals...")
        # Predict on training set to get residuals
        lstm_preds = self.lstm.predict(X_train)
        
        # Flatten for arithmetic
        y_train_flat = y_train.flatten()
        lstm_preds_flat = lstm_preds.flatten()
        
        # Residual = Actual - Predicted
        residuals = y_train_flat - lstm_preds_flat
        
        print("Training ARIMA part on residuals...")
        self.arima.train(residuals)
        print("Hybrid Model Training Complete.")

    def predict(self, X_sample):
        """
        X_sample: (N, look_back, features)
        Returns: Final, LSTM, ARIMA components
        """
        # 1. LSTM Prediction
        lstm_pred = self.lstm.predict(X_sample)  # Shape (N, 1)
        
        # 2. ARIMA Residual Prediction
        arima_res_pred = self.arima.predict_sequence(steps=len(X_sample))
        arima_res_pred = arima_res_pred.reshape(-1, 1)
        
        final_pred = lstm_pred + arima_res_pred
        return final_pred, lstm_pred, arima_res_pred

    def predict_future(self, initial_sequence, steps, close_feature_idx=3):
        """
        Recursive prediction for future steps (multivariate aware).
        
        Args:
            initial_sequence: (1, look_back, n_features) - The last available data window.
            steps: number of steps to predict ahead.
            close_feature_idx: Index of Close in features (default 3 for OHLC ordering)
        
        Returns: Final, LSTM, ARIMA predictions
        """
        print(f"Generating future prediction for {steps} steps...")
        curr_seq = initial_sequence.copy()
        lstm_future = []
        
        # 1. Recursive LSTM
        for _ in range(steps):
            # Predict next step
            pred = self.lstm.model.predict(curr_seq, verbose=0)
            val = pred[0, 0]
            lstm_future.append(val)
            
            # Update sequence: remove first timestep, add new one
            if self.n_features == 1:
                # Univariate case
                new_step = np.array([[[val]]])
            else:
                # Multivariate: create new timestep with predicted Close
                # Copy last known features and update Close
                last_features = curr_seq[0, -1, :].copy()
                last_features[close_feature_idx] = val  # Update Close
                new_step = last_features.reshape(1, 1, self.n_features)
            
            curr_seq = np.concatenate([curr_seq[:, 1:, :], new_step], axis=1)
            
        lstm_future = np.array(lstm_future).reshape(-1, 1)
        
        # 2. ARIMA Forecast
        arima_future = self.arima.predict_sequence(steps=steps).reshape(-1, 1)
        
        final_future = lstm_future + arima_future
        return final_future, lstm_future, arima_future

