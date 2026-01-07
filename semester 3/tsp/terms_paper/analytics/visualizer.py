import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib

matplotlib.use('Agg')  # Для работы без GUI

from config import TEMP_DIR


def create_forecast_plot(
        historical_data: pd.DataFrame,
        forecast: pd.Series,
        signals: list,
        ticker: str
) -> Path:
    """
    Создает график с историческими данными и прогнозом.

    Args:
        historical_data: Исторические данные
        forecast: Прогноз
        signals: Список торговых сигналов
        ticker: Тикер акции

    Returns:
        Путь к сохраненному изображению
    """
    # Создаем график
    plt.figure(figsize=(14, 8))

    # Исторические данные
    plt.plot(
        historical_data['Date'],
        historical_data['Close'],
        label='Исторические данные',
        color='blue',
        linewidth=2
    )

    # Прогноз
    plt.plot(
        forecast.index,
        forecast.values,
        label='Прогноз на 30 дней',
        color='red',
        linewidth=2,
        linestyle='--'
    )

    # Торговые сигналы
    buy_dates = [s['date'] for s in signals if s['action'] == 'BUY']
    buy_prices = [s['price'] for s in signals if s['action'] == 'BUY']

    sell_dates = [s['date'] for s in signals if s['action'] == 'SELL']
    sell_prices = [s['price'] for s in signals if s['action'] == 'SELL']

    if buy_dates:
        plt.scatter(
            buy_dates, buy_prices,
            color='green', s=200, marker='^',
            label='Сигнал ПОКУПКИ', zorder=5
        )

    if sell_dates:
        plt.scatter(
            sell_dates, sell_prices,
            color='orange', s=200, marker='v',
            label='Сигнал ПРОДАЖИ', zorder=5
        )

    # Настройки графика
    plt.title(f'Прогноз цен акций {ticker}', fontsize=16, fontweight='bold')
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Цена (USD)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Форматирование оси X
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Сохраняем график
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_{timestamp}.png"
    filepath = TEMP_DIR / filename

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filepath