import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

from models.base_model import BaseModel
from config import LSTM_SEQUENCE_LENGTH, LSTM_EPOCHS, LSTM_BATCH_SIZE


class NNModel(BaseModel):
    """Нейросетевая модель LSTM."""

    def __init__(self):
        super().__init__("LSTM")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = LSTM_SEQUENCE_LENGTH

    def create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Создает последовательности для обучения LSTM."""
        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])

        return np.array(X), np.array(y)

    def build_model(self, input_shape: tuple) -> tf.keras.Model:
        """Строит архитектуру LSTM модели."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, train_data: pd.DataFrame) -> None:
        """Обучает LSTM модель на исторических данных."""
        # Нормализуем данные
        series = train_data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(series)

        # Создаем последовательности
        X, y = self.create_sequences(scaled_data)

        # Меняем форму для LSTM: [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Строим и обучаем модель
        self.model = self.build_model((self.sequence_length, 1))

        # Обучаем с ранней остановкой
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )

        self.model.fit(
            X, y,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            verbose=0,
            callbacks=[early_stopping]
        )

        self.is_trained = True
        self.last_sequence = scaled_data[-self.sequence_length:]

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Прогнозирует на тестовых данных (упрощенная версия)."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        # Для простоты используем итеративный прогноз
        predictions = []
        current_sequence = self.last_sequence.copy()

        for _ in range(len(test_data)):
            # Подготавливаем последовательность
            X_pred = current_sequence.reshape(1, self.sequence_length, 1)

            # Прогнозируем
            pred_scaled = self.model.predict(X_pred, verbose=0)
            pred = self.scaler.inverse_transform(pred_scaled)[0, 0]
            predictions.append(pred)

            # Обновляем последовательность
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_scaled[0, 0]

        return np.array(predictions)

    def forecast(self, data: pd.DataFrame, steps: int) -> pd.Series:
        """Строит прогноз на steps шагов вперед."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        # Переобучаем на всех данных
        series = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(series)

        X, y = self.create_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        self.model = self.build_model((self.sequence_length, 1))
        self.model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=0)

        # Строим прогноз итеративно
        forecasts = []
        current_sequence = scaled_data[-self.sequence_length:].copy()

        for _ in range(steps):
            X_pred = current_sequence.reshape(1, self.sequence_length, 1)
            pred_scaled = self.model.predict(X_pred, verbose=0)
            pred = self.scaler.inverse_transform(pred_scaled)[0, 0]
            forecasts.append(pred)

            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_scaled[0, 0]

        # Создаем индекс дат для прогноза
        last_date = data['Date'].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='B'
        )

        return pd.Series(forecasts, index=forecast_dates)