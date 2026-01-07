import pandas as pd
import numpy as np
from typing import Any
from config import FORECAST_DAYS


def make_forecast(model: Any, data: pd.DataFrame, model_name: str) -> pd.Series:
    """
    Строит прогноз на заданное количество дней вперед.

    Args:
        model: Обученная модель
        data: Исторические данные
        model_name: Название модели (для логирования)

    Returns:
        Прогноз на FORECAST_DAYS дней
    """
    # Строим прогноз
    forecast = model.forecast(data, FORECAST_DAYS)

    return forecast