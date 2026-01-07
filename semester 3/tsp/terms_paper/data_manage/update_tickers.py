import json
from typing import Dict, List
from pathlib import Path


def get_popular_tickers() -> Dict[str, List[str]]:
    """Получает список популярных тикеров с Yahoo Finance."""

    # Список популярных тикеров по категориям
    popular_tickers = {
        "Tech": ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "ADBE", "NFLX"],
        "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V"],
        "Healthcare": ["JNJ", "UNH", "PFE", "ABT", "MRK", "TMO", "LLY", "AMGN", "GILD", "BMY"],
        "Consumer": ["PG", "KO", "PEP", "WMT", "COST", "MCD", "SBUX", "NKE", "TGT", "HD"],
        "Industrial": ["BA", "CAT", "GE", "HON", "MMM", "UNP", "UPS", "RTX", "LMT", "DE"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "KMI", "WMB"],
        "Communication": ["VZ", "T", "TMUS", "CMCSA", "DIS", "NFLX", "CHTR", "EA", "ATVI", "TTWO"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ED"],
        "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "AVB", "O", "DLR", "WELL"]
    }

    # Группируем по буквам
    tickers_by_letter: Dict[str, List[str]] = {}

    for category, tickers in popular_tickers.items():
        for ticker in tickers:
            first_letter = ticker[0]
            if first_letter not in tickers_by_letter:
                tickers_by_letter[first_letter] = []
            if ticker not in tickers_by_letter[first_letter]:
                tickers_by_letter[first_letter].append(ticker)

    # Сортируем каждый список
    for letter in tickers_by_letter:
        tickers_by_letter[letter].sort()

    return tickers_by_letter


def save_tickers_to_file(tickers_data: Dict[str, List[str]], filepath: Path) -> None:
    """Сохраняет тикеры в JSON файл."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(tickers_data, f, indent=2, ensure_ascii=False)

    print(f"Сохранено {sum(len(v) for v in tickers_data.values())} тикеров в {filepath}")


def main():
    """Основная функция."""
    tickers_data = get_popular_tickers()

    # Путь к файлу
    filepath = Path("data_manage/tickers.json")

    # Сохраняем данные
    save_tickers_to_file(tickers_data, filepath)

    # Выводим статистику
    print("\nСтатистика по тикерам:")
    for letter, tickers in sorted(tickers_data.items()):
        print(f"{letter}: {len(tickers)} тикеров")


if __name__ == "__main__":
    main()