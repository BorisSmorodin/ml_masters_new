import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Пути
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "tmp"

# Создание директорий если их нет
LOGS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Путь к файлу логов
LOG_FILE = LOGS_DIR / "requests_log.csv"

# Параметры данных
START_DATE = '2022-01-01'
END_DATE = '2024-01-01'
FORECAST_DAYS = 30
TEST_SIZE = 0.2  # 20% данных для тестирования

# Параметры моделей
LAG_WINDOW = 30  # Количество лагов для ML модели
ARIMA_ORDER = (5, 1, 0)  # (p, d, q) для ARIMA
LSTM_SEQUENCE_LENGTH = 60  # Длина последовательности для LSTM
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# Telegram Bot
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# Инициализация файла логов если он не существует
if not LOG_FILE.exists():
    with open(LOG_FILE, 'w') as f:
        f.write("user_id,timestamp,ticker,amount,best_model,metric_value,profit\n")