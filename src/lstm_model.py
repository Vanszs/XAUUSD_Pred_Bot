import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class LSTMModel:
    def __init__(self, input_shape, units_1=128, units_2=64, dropout=0.15):
        """
        input_shape: (time_steps, features)
        units_1: First LSTM layer units
        units_2: Second LSTM layer units
        dropout: Dropout rate
        """
        self.input_shape = input_shape
        self.units_1 = units_1
        self.units_2 = units_2
        self.dropout = dropout
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # Enhanced architecture for better pattern capture
        model.add(LSTM(self.units_1, return_sequences=True, input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))
        
        model.add(LSTM(self.units_2, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))  # Output: Predicted Price (Scaled)
        
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None):
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

