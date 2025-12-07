from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

class ARIMAModel:
    def __init__(self, order=(5, 1, 0)):
        """
        order: (p, d, q)
        """
        self.order = order
        self.model_fit = None
        self.residuals = None

    def train(self, residuals):
        """
        Train ARIMA on the residuals of LSTM.
        residuals: 1D array or series
        """
        # ARIMA expects a 1D series
        self.residuals = residuals
        model = ARIMA(residuals, order=self.order)
        self.model_fit = model.fit()
        print(self.model_fit.summary())

    def predict_next_step(self):
        """
        Predict the next time step residual.
        """
        if self.model_fit is None:
            raise ValueError("Model not trained yet.")
        
        # Forecast 1 step ahead
        forecast = self.model_fit.forecast(steps=1)
        return forecast[0]
    
    def predict_sequence(self, steps=1):
        if self.model_fit is None:
            raise ValueError("Model not trained yet.")
        return self.model_fit.forecast(steps=steps)
