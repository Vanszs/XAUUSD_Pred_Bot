import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


class BiGRUModel:
    """
    Bidirectional GRU model for gold price prediction.
    Based on paper: "Implementation of GRU, LSTM and Derivatives for Gold Price Prediction"
    
    Best parameters from paper:
    - Optimizer: Nadam
    - Batch size: 8
    - Time steps: 20
    - Learning rate: 0.0001
    - MAPE achieved: 0.8857%
    """
    
    def __init__(self, input_shape, units=64, dropout_rate=0.2, learning_rate=0.0001):
        """
        Initialize BiGRU model.
        
        Args:
            input_shape: (time_steps, features) - e.g., (20, 1)
            units: Number of GRU units per layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Nadam optimizer
        """
        self.input_shape = input_shape
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the Bidirectional GRU architecture."""
        model = Sequential([
            # First Bi-GRU layer with return sequences
            Bidirectional(
                GRU(self.units, return_sequences=True),
                input_shape=self.input_shape
            ),
            Dropout(self.dropout_rate),
            
            # Second Bi-GRU layer
            Bidirectional(
                GRU(self.units, return_sequences=False)
            ),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dense(1)  # Output: single price prediction
        ])
        
        # Use Nadam optimizer as per paper
        optimizer = Nadam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, X_train, y_train, epochs=1000, batch_size=8, 
              validation_split=0.1, use_callbacks=True):
        """
        Train the BiGRU model.
        
        Args:
            X_train: Training features (N, time_steps, features)
            y_train: Training targets (N, 1)
            epochs: Maximum training epochs (paper uses 1000)
            batch_size: Batch size (paper recommends 8)
            validation_split: Validation data fraction
            use_callbacks: Whether to use early stopping and checkpoints
        """
        callbacks = []
        
        if use_callbacks:
            # Early stopping with patience
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
            
            # Model checkpoint
            os.makedirs('models', exist_ok=True)
            checkpoint = ModelCheckpoint(
                'models/bigru_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
            callbacks.append(checkpoint)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks if callbacks else None,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X, verbose=0)
    
    def summary(self):
        """Print model summary."""
        return self.model.summary()
