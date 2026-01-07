import pandas as pd
import numpy as np
from typing import Tuple

from config import TEST_SIZE


def train_test_split_time_series(
        data: pd.DataFrame,
        target_col: str = 'Close'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделяет временной ряд на обучающую и тестовую выборки.

    Args:
        data: DataFrame с временным рядом
        target_col: Название целевой колонки

    Returns:
        Кортеж (train_data, test_data)
    """
    # Определяем индекс разделения
    split_idx = int(len(data) * (1 - TEST_SIZE))

    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()

    return train_data, test_data


def create_lag_features(data: pd.Series, window: int) -> pd.DataFrame:
    """
    Создает лаговые признаки для временного ряда.

    Args:
        data: Временной ряд
        window: Количество лагов

    Returns:
        DataFrame с лаговыми признаками
    """
    df = pd.DataFrame(data)

    for lag in range(1, window + 1):
        df[f'lag_{lag}'] = data.shift(lag)

    # Удаляем строки с NaN
    df = df.dropna()

    return df