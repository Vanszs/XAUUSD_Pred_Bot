import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class LSTMModel:
    def __init__(self, input_shape):
        """
        input_shape: (time_steps, features)
        """
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # Paper suggests LSTM learns nonlinear features
        # Common architecture: LSTM layers -> Dense output
        model.add(LSTM(50, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1)) # Output: Predicted Price (Scaled)
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def train(self, X_train, y_train, epochs=20, batch_size=32, validation_data=None):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
