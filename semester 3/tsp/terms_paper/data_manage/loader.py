import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

from config import HISTORY_YEARS

logger = logging.getLogger(__name__)


def load_stock_data(ticker: str) -> pd.DataFrame:
    """
    Загружает исторические данные акций за последние N лет.

    Args:
        ticker: Тикер акции (например, 'AAPL')

    Returns:
        DataFrame с колонками ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    try:
        # Рассчитываем даты
        end_date = datetime.now()
        start_date = end_date - timedelta(days=HISTORY_YEARS * 365)

        logger.info(f"Загрузка данных для {ticker} с {start_date.date()} по {end_date.date()}")

        # Загружаем данные через yfinance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"Нет данных для тикера {ticker}")

        # Сбрасываем индекс чтобы получить Date как колонку
        df = df.reset_index()

        # Оставляем только необходимые колонки
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Удаляем строки с пропущенными значениями
        df = df.dropna(subset=['Close'])

        logger.info(f"Загружено {len(df)} строк для {ticker}")

        return df

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных для {ticker}: {e}")
        raise