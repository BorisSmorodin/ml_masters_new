import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from models.base_model import BaseModel

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels не установлен. Установите его для использования ARIMA модели.")


class StatsModel(BaseModel):
    """Статистическая модель ARIMA с автоматическим подбором параметров."""

    def __init__(self):
        super().__init__("ARIMA")
        self.differencing_order = 1  # Порядок дифференцирования по умолчанию

    def check_stationarity(self, series: pd.Series) -> bool:
        """Проверяет стационарность временного ряда."""
        try:
            result = adfuller(series.dropna())
            p_value = result[1]
            return p_value < 0.05  # Ряд стационарен если p-value < 0.05
        except:
            return False

    def find_best_arima_order(self, series: pd.Series) -> tuple:
        """Находит оптимальные параметры ARIMA (p,d,q)."""
        # Проверяем стационарность
        d = 0
        if not self.check_stationarity(series):
            d = 1  # Применяем одно дифференцирование

        # Простые параметры, которые обычно работают для финансовых данных
        # Можно расширить поиск, но для скорости используем фиксированные
        return (1, d, 1)  # AR(1), I(d), MA(1)

    def train(self, train_data: pd.DataFrame) -> None:
        """Обучает ARIMA модель на исторических данных."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels не установлен.")

        series = train_data['Close']

        # Находим оптимальные параметры
        order = self.find_best_arima_order(series)
        self.order = order

        try:
            # Обучаем ARIMA модель
            self.model = ARIMA(series, order=order)
            self.model_fit = self.model.fit()
            self.is_trained = True
            print(f"ARIMA обучена с параметрами: {order}")
        except Exception as e:
            print(f"Ошибка при обучении ARIMA: {e}")
            # Пробуем более простые параметры
            try:
                self.model = ARIMA(series, order=(1, 1, 1))
                self.model_fit = self.model.fit()
                self.is_trained = True
                print("ARIMA обучена с параметрами (1,1,1)")
            except:
                # Последняя попытка с простейшей моделью
                self.model = ARIMA(series, order=(0, 1, 0))
                self.model_fit = self.model.fit()
                self.is_trained = True
                print("ARIMA обучена с параметрами (0,1,0)")

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Прогнозирует на тестовых данных."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        try:
            # Получаем прогноз на длину тестовых данных
            forecast_result = self.model_fit.forecast(steps=len(test_data))
            return forecast_result.values
        except Exception as e:
            print(f"Ошибка при прогнозе ARIMA: {e}")
            # Возвращаем последнее известное значение в качестве прогноза
            last_value = self.model_fit.data.orig_endog[-1]
            return np.full(len(test_data), last_value)

    def forecast(self, data: pd.DataFrame, steps: int) -> pd.Series:
        """Строит прогноз на steps шагов вперед."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        try:
            # Переобучаем модель на всех данных для лучшего прогноза
            series = data['Close']

            # Используем те же параметры, что и при обучении
            model = ARIMA(series, order=self.order)
            model_fit = model.fit()

            # Получаем прогноз
            forecast_result = model_fit.forecast(steps=steps)

            # Проверяем на NaN и заменяем их
            if forecast_result.isna().any():
                print("Обнаружены NaN в прогнозе ARIMA, заменяю на последнее значение")
                last_value = series.iloc[-1]
                forecast_result = forecast_result.fillna(last_value)

            # Создаем индекс дат для прогноза
            last_date = data['Date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq='B'
            )

            # Создаем Series с правильным индексом
            forecast_series = pd.Series(forecast_result.values, index=forecast_dates)

            return forecast_series

        except Exception as e:
            print(f"Ошибка при построении прогноза ARIMA: {e}")
            # Создаем простой прогноз на основе последнего значения
            last_value = data['Close'].iloc[-1]

            # Создаем индекс дат для прогноза
            last_date = data['Date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq='B'
            )

            # Возвращаем постоянный прогноз
            return pd.Series([last_value] * steps, index=forecast_dates)