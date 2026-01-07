import pandas as pd
import numpy as np
from typing import Tuple, Optional

from config import TEST_SIZE


def train_test_split_time_series(data: pd.DataFrame,
                                 target_col: str = 'Close',
                                 test_size: float = TEST_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделяет временной ряд на обучающую и тестовую выборки.

    Args:
        data: DataFrame с временным рядом
        target_col: Название целевой колонки
        test_size: Доля тестовых данных

    Returns:
        Кортеж (train_data, test_data)
    """
    # Определяем индекс разделения
    split_idx = int(len(data) * (1 - test_size))

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

    # Создаем лаги
    for lag in range(1, window + 1):
        df[f'lag_{lag}'] = data.shift(lag)

    # Удаляем строки с NaN (первые window строк)
    df = df.iloc[window:].reset_index(drop=True)

    return df


def prepare_ml_data(data: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подготавливает данные для ML моделей.

    Args:
        data: Временной ряд
        window: Размер окна

    Returns:
        Кортеж (X, y) - признаки и целевая переменная
    """
    X, y = [], []

    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])

    return np.array(X), np.array(y)