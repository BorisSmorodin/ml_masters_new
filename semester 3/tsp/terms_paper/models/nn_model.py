import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow не установлен. LSTM модель будет пропущена.")

from sklearn.preprocessing import MinMaxScaler
import logging

from models.base_model import BaseModel
from config import LSTM_SEQUENCE_LENGTH, LSTM_EPOCHS, LSTM_BATCH_SIZE

logger = logging.getLogger(__name__)


class NNModel(BaseModel):
    """Исправленная нейросетевая модель LSTM."""

    def __init__(self):
        super().__init__("LSTM")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = LSTM_SEQUENCE_LENGTH

        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow не установлен. Установите его для использования LSTM.")

    def create_sequences(self, data: np.ndarray) -> tuple:
        """Создает последовательности для обучения LSTM."""
        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])

        return np.array(X), np.array(y)

    def build_model(self) -> tf.keras.Model:
        """Строит стабильную архитектуру LSTM модели."""
        model = Sequential([
            LSTM(50, activation='tanh', return_sequences=False, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model

    def train(self, train_data: pd.DataFrame) -> None:
        """Обучает LSTM модель на исторических данных."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow не установлен.")

        # Используем только цену закрытия
        series = train_data['Close'].values.reshape(-1, 1)

        # Нормализуем данные
        scaled_data = self.scaler.fit_transform(series)

        # Проверяем, что данные не содержат NaN или бесконечных значений
        if np.any(np.isnan(scaled_data)) or np.any(np.isinf(scaled_data)):
            logger.warning("Обнаружены NaN или бесконечные значения в данных. Заменяю нулями.")
            scaled_data = np.nan_to_num(scaled_data)

        # Создаем последовательности
        X, y = self.create_sequences(scaled_data)

        if len(X) == 0:
            raise ValueError("Недостаточно данных для создания последовательностей")

        # Изменяем форму для LSTM: [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Строим и обучаем модель
        self.model = self.build_model()

        # Обучаем с ранней остановкой
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.0001
        )

        try:
            history = self.model.fit(
                X, y,
                epochs=LSTM_EPOCHS,
                batch_size=LSTM_BATCH_SIZE,
                verbose=0,
                callbacks=[early_stopping],
                validation_split=0.1
            )

            self.is_trained = True
            self.last_sequence = scaled_data[-self.sequence_length:].copy()

            logger.info(f"LSTM обучена. Финальный loss: {history.history['loss'][-1]:.6f}")

        except Exception as e:
            logger.error(f"Ошибка при обучении LSTM: {e}")
            raise

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Прогнозирует на тестовых данных."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        try:
            # Используем итеративный прогноз
            predictions = []
            current_sequence = self.last_sequence.copy()

            for _ in range(len(test_data)):
                # Подготавливаем последовательность
                X_pred = current_sequence.reshape(1, self.sequence_length, 1)

                # Прогнозируем
                pred_scaled = self.model.predict(X_pred, verbose=0)

                # Проверяем, что прогноз не содержит NaN
                if np.any(np.isnan(pred_scaled)):
                    logger.warning("LSTM вернула NaN в прогнозе. Использую последнее значение.")
                    pred_scaled = current_sequence[-1]

                pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                predictions.append(pred)

                # Обновляем последовательность
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred_scaled[0, 0]

            return np.array(predictions)

        except Exception as e:
            logger.error(f"Ошибка при прогнозе LSTM: {e}")
            # Возвращаем последнее известное значение в качестве прогноза
            last_value = self.scaler.inverse_transform(self.last_sequence[-1:])[0, 0]
            return np.full(len(test_data), last_value)

    def forecast(self, data: pd.DataFrame, steps: int) -> pd.Series:
        """Строит прогноз на steps шагов вперед."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        try:
            # Используем все данные для прогноза
            series = data['Close'].values.reshape(-1, 1)

            # Нормализуем данные
            scaled_data = self.scaler.fit_transform(series)

            # Создаем последовательности
            X, y = self.create_sequences(scaled_data)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Обучаем модель на всех данных
            self.model = self.build_model()
            self.model.fit(
                X, y,
                epochs=min(LSTM_EPOCHS, 30),  # Меньше эпох для скорости
                batch_size=LSTM_BATCH_SIZE,
                verbose=0
            )

            # Строим прогноз итеративно
            forecasts = []
            current_sequence = scaled_data[-self.sequence_length:].copy()

            for _ in range(steps):
                X_pred = current_sequence.reshape(1, self.sequence_length, 1)
                pred_scaled = self.model.predict(X_pred, verbose=0)

                # Проверяем, что прогноз не содержит NaN
                if np.any(np.isnan(pred_scaled)):
                    pred_scaled = current_sequence[-1:]

                pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
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

        except Exception as e:
            logger.error(f"Ошибка при построении прогноза LSTM: {e}")
            # Создаем простой прогноз на основе последнего значения
            last_value = data['Close'].iloc[-1]
            last_date = data['Date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq='B'
            )
            return pd.Series([last_value] * steps, index=forecast_dates)