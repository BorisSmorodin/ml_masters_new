import pandas as pd
from datetime import datetime

from config import LOG_FILE


def log_request(
        user_id: int,
        timestamp: datetime,
        ticker: str,
        amount: float,
        forecast_days: int,
        best_model: str,
        metric_value: float,
        profit: float
) -> None:
    """
    Записывает информацию о запросе в лог-файл.
    """
    log_entry = {
        'user_id': user_id,
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker,
        'amount': amount,
        'forecast_days': forecast_days,
        'best_model': best_model,
        'metric_value': metric_value,
        'profit': profit
    }

    # Добавляем запись в CSV файл
    df = pd.DataFrame([log_entry])

    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, mode='w', header=True, index=False)

    print(f"Запись добавлена в лог: {log_entry}")