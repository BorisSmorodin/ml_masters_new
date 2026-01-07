import pandas as pd
import numpy as np
from typing import Any
from config import FORECAST_DAYS


def make_forecast(model: Any, data: pd.DataFrame, model_name: str, steps: int = 30) -> pd.Series:
    """
    Строит прогноз на заданное количество дней вперед.

    Args:
        model: Обученная модель
        data: Исторические данные
        model_name: Название модели (для логирования)
        steps: Количество дней для прогноза

    Returns:
        Прогноз на steps дней
    """
    # Строим прогноз
    forecast = model.forecast(data, steps)

    return forecast