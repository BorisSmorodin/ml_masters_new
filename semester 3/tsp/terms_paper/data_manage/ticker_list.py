import json
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class TickerManager:
    """Менеджер для работы со списком тикеров."""

    def __init__(self, tickers_file: Path = None):
        self.tickers_file = tickers_file or Path(__file__).parent / "tickers.json"
        self.tickers_data = self._load_tickers()

    def _load_tickers(self) -> Dict[str, List[str]]:
        """Загружает тикеры из файла."""
        try:
            with open(self.tickers_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при загрузке тикеров: {e}")
            return {}

    def get_tickers_by_letter(self, letter: str, limit: int = 10) -> List[str]:
        """Возвращает тикеры, начинающиеся с указанной буквы."""
        letter = letter.upper()
        if letter in self.tickers_data:
            return self.tickers_data[letter][:limit]
        return []

    def get_all_tickers(self) -> List[str]:
        """Возвращает все тикеры из списка."""
        all_tickers = []
        for tickers in self.tickers_data.values():
            all_tickers.extend(tickers)
        return sorted(all_tickers)

    def search_tickers(self, query: str, limit: int = 10) -> List[str]:
        """Ищет тикеры по частичному совпадению."""
        query = query.upper()
        results = []

        for letter, tickers in self.tickers_data.items():
            for ticker in tickers:
                if query in ticker:
                    results.append(ticker)

        return sorted(results)[:limit]


# Создаем глобальный экземпляр менеджера
ticker_manager = TickerManager()