import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    # Для совместимости со старыми версиями statsmodels
    from statsmodels.tsa.arima_model import ARIMA as ARIMA_OLD


    class ARIMA(ARIMA_OLD):
        def fit(self, *args, **kwargs):
            return super().fit(*args, **kwargs)

from models.base_model import BaseModel
from config import ARIMA_ORDER


class StatsModel(BaseModel):
    """Статистическая модель ARIMA."""

    def __init__(self):
        super().__init__("ARIMA")
        self.order = ARIMA_ORDER

    def train(self, train_data: pd.DataFrame) -> None:
        """Обучает ARIMA модель на исторических данных."""
        series = train_data['Close']

        # Обучаем ARIMA модель
        self.model = ARIMA(series, order=self.order)
        self.model_fit = self.model.fit()
        self.is_trained = True

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Прогнозирует на тестовых данных."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        # Получаем прогноз на длину тестовых данных
        forecast_result = self.model_fit.forecast(steps=len(test_data))

        return forecast_result

    def forecast(self, data: pd.DataFrame, steps: int) -> pd.Series:
        """Строит прогноз на steps шагов вперед."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        # Переобучаем модель на всех данных для лучшего прогноза
        series = data['Close']
        model = ARIMA(series, order=self.order)
        model_fit = model.fit()

        # Получаем прогноз
        forecast_result = model_fit.forecast(steps=steps)

        # Создаем индекс дат для прогноза
        last_date = data['Date'].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='B'
        )

        return pd.Series(forecast_result, index=forecast_dates)