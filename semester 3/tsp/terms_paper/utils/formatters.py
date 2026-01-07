def format_currency(value: float) -> str:
    """Форматирует число как валюту."""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Форматирует число как процент."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"

def format_number(value: float) -> str:
    """Форматирует число с разделителями."""
    return f"{value:,.2f}"