from typing import Any

import pandas as pd
import logging

from models.ml_model import MLModel
from models.stats_model import StatsModel
from models.nn_model import NNModel
from data_manage.splitter import train_test_split_time_series

logger = logging.getLogger(__name__)


def train_and_evaluate_models(data: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Обучает и оценивает все модели.

    Returns:
        Список словарей с результатами каждой модели
    """
    # Разделяем данные
    train_data, test_data = train_test_split_time_series(data)

    # Инициализируем модели
    models = [
        MLModel(),
        StatsModel(),
        NNModel()
    ]

    results = []

    for model in models:
        try:
            logger.info(f"Обучаю модель: {model.name}")

            # Обучаем модель
            model.train(train_data)

            # Прогнозируем на тестовых данных
            y_true = test_data['Close'].values
            y_pred = model.predict(test_data)

            # Оцениваем качество
            metrics = model.evaluate(y_true, y_pred)
            metrics['model'] = model
            metrics['predictions'] = y_pred

            results.append(metrics)

            logger.info(f"Модель {model.name}: RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%")

        except Exception as e:
            logger.error(f"Ошибка при обучении модели {model.name}: {e}")
            # Продолжаем с другими моделями

    return results


def select_best_model(results: list[dict[str, Any]]) -> tuple[str, Any, dict[str, float]]:
    """
    Выбирает лучшую модель на основе метрики RMSE.

    Returns:
        Кортеж (имя_модели, объект_модели, метрики)
    """
    if not results:
        raise ValueError("Нет результатов для выбора модели!")

    # Выбираем модель с наименьшим RMSE
    best_result = min(results, key=lambda x: x['rmse'])

    return best_result['model_name'], best_result['model'], best_result