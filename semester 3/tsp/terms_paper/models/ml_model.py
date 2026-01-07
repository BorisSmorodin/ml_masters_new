import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from models.base_model import BaseModel
from data_manage.splitter import create_lag_features
from config import LAG_WINDOW


class MLModel(BaseModel):
    """ML модель на основе Random Forest с лаговыми признаками."""

    def __init__(self):
        super().__init__("Random Forest")
        self.scaler = StandardScaler()
        self.lag_window = LAG_WINDOW

    def prepare_features(self, data: pd.Series, is_training: bool = True) -> tuple[pd.DataFrame, pd.Series]:
        """Подготавливает признаки для обучения/прогноза."""
        # Создаем лаговые признаки
        df_features = create_lag_features(data, self.lag_window)

        # Целевая переменная - текущее значение
        y = df_features['Close']

        # Признаки - лаги
        X = df_features.drop('Close', axis=1)

        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def train(self, train_data: pd.DataFrame) -> None:
        """Обучает модель на исторических данных."""
        # Используем только цену закрытия
        series = train_data['Close']

        # Подготавливаем признаки
        X, y = self.prepare_features(series, is_training=True)

        # Обучаем модель
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Прогнозирует на тестовых данных."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        # Используем всю историю для создания лагов
        full_series = pd.concat([self.last_train_data, test_data['Close']])

        predictions = []
        for i in range(len(test_data)):
            # Для каждого шага создаем лаги из последних данных
            window = full_series.iloc[i:i + self.lag_window].values.reshape(1, -1)
            window_scaled = self.scaler.transform(window)
            pred = self.model.predict(window_scaled)
            predictions.append(pred[0])

        return np.array(predictions)

    def forecast(self, data: pd.DataFrame, steps: int) -> pd.Series:
        """Строит прогноз на steps шагов вперед."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        # Сохраняем последние данные для создания лагов
        last_values = data['Close'].values[-self.lag_window:]

        forecasts = []
        current_window = last_values.copy()

        for _ in range(steps):
            # Масштабируем текущее окно
            window_scaled = self.scaler.transform(current_window.reshape(1, -1))

            # Прогнозируем следующий шаг
            next_pred = self.model.predict(window_scaled)[0]
            forecasts.append(next_pred)

            # Обновляем окно: удаляем самый старый, добавляем прогноз
            current_window = np.roll(current_window, -1)
            current_window[-1] = next_pred

        # Создаем индекс дат для прогноза
        last_date = data['Date'].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='B'
        )

        return pd.Series(forecasts, index=forecast_dates)