import numpy as np
from src.lstm_model import LSTMModel
from src.arima_model import ARIMAModel

class HybridModel:
    def __init__(self, input_shape, arima_order=(5, 1, 0)):
        self.lstm = LSTMModel(input_shape)
        self.arima = ARIMAModel(arima_order)
        self.look_back = input_shape[0]

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        print("Training LSTM part...")
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
        # Note: ARIMA training can be slow on large datasets
        self.arima.train(residuals)
        print("Hybrid Model Training Complete.")

    def predict(self, X_sample):
        """
        X_sample: (N, look_back, features)
        Returns: Final, LSTM, ARIMA components
        """
        # 1. LSTM Prediction
        lstm_pred = self.lstm.predict(X_sample) # Shape (N, 1)
        
        # 2. ARIMA Residual Prediction (forecast N steps for simplicity/batch)
        # Note: Ideally running ARIMA on each rolling window residual would be better but slower.
        # Here we assume X_sample represents a contiguous test block? 
        # If random samples, ARIMA forecast isn't strictly valid this way.
        # Assuming sequential test data for 'predict'.
        arima_res_pred = self.arima.predict_sequence(steps=len(X_sample))
        arima_res_pred = arima_res_pred.reshape(-1, 1)
        
        final_pred = lstm_pred + arima_res_pred
        return final_pred, lstm_pred, arima_res_pred

    def predict_future(self, initial_sequence, steps):
        """
        Recursive prediction for future steps.
        initial_sequence: (1, look_back, 1) - The last available data window.
        steps: number of steps to predict ahead.
        """
        print(f"Generating future prediction for {steps} steps...")
        curr_seq = initial_sequence.copy()
        lstm_future = []
        
        # 1. Recursive LSTM
        for _ in range(steps):
            # Predict next step
            # verbose=0 to reduce clutter
            pred = self.lstm.model.predict(curr_seq, verbose=0) 
            val = pred[0, 0]
            lstm_future.append(val)
            
            # Update sequence: remove first, add new prediction
            new_step = np.array([[[val]]])
            curr_seq = np.concatenate([curr_seq[:, 1:, :], new_step], axis=1)
            
        lstm_future = np.array(lstm_future).reshape(-1, 1)
        
        # 2. ARIMA Forecast
        # Forecast 'steps' ahead from the end of training residuals
        arima_future = self.arima.predict_sequence(steps=steps).reshape(-1, 1)
        
        final_future = lstm_future + arima_future
        return final_future, lstm_future, arima_future

    def predict_future_monte_carlo(self, initial_sequence, steps, n_simulations=100, volatility=None):
        """
        Optimized Monte Carlo simulation for future predictions with confidence bands.
        """
        print(f"Running {n_simulations} Monte Carlo simulations for {steps} steps...")
        
        if volatility is None:
            volatility = 0.002
        
        # Pre-generate ALL noise upfront (faster than generating one at a time)
        all_noise = np.random.normal(0, volatility, (n_simulations, steps))
        
        all_simulations = np.zeros((n_simulations, steps))
        
        for sim in range(n_simulations):
            curr_seq = initial_sequence.copy()
            
            for step in range(steps):
                pred = self.lstm.model.predict(curr_seq, verbose=0)
                val = pred[0, 0] + all_noise[sim, step]
                all_simulations[sim, step] = val
                
                new_step = np.array([[[val]]])
                curr_seq = np.concatenate([curr_seq[:, 1:, :], new_step], axis=1)
            
            # Progress indicator every 10 simulations
            if (sim + 1) % 10 == 0:
                print(f"  Completed {sim + 1}/{n_simulations} simulations...")
        
        # Add ARIMA component
        arima_future = self.arima.predict_sequence(steps=steps)
        all_simulations = all_simulations + arima_future.reshape(1, -1)
        
        # Calculate statistics
        mean_forecast = np.mean(all_simulations, axis=0).reshape(-1, 1)
        lower_band = np.percentile(all_simulations, 5, axis=0).reshape(-1, 1)
        upper_band = np.percentile(all_simulations, 95, axis=0).reshape(-1, 1)
        
        print("Monte Carlo simulation complete.")
        return mean_forecast, lower_band, upper_band, all_simulations

