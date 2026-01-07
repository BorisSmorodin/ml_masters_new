import logging
from datetime import datetime
from typing import Dict, Any

from telegram import Update, ReplyKeyboardRemove
from telegram.ext import ContextTypes, ConversationHandler

from core.states import TICKER, AMOUNT, FORECAST_DAYS
from data_manage.loader import load_stock_data
from data_manage.ticker_list import ticker_manager
from models.model_selector import select_best_model, train_and_evaluate_models
from analytics.forecaster import make_forecast
from analytics.visualizer import create_forecast_plot
from analytics.strategy import generate_trading_signals, calculate_profit
from utils.logger import log_request
from utils.formatters import format_currency, format_percentage

logger = logging.getLogger(__name__)

# Глобальный словарь для хранения данных пользователя между состояниями
user_sessions: Dict[int, Dict[str, Any]] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало диалога, запрос тикера."""
    user = update.message.from_user
    logger.info(f"Пользователь {user.first_name} начал диалог.")

    await update.message.reply_text(
        "Привет! Я бот для анализа и прогнозирования акций.\n\n"
        "Я помогу вам:\n"
        "1. Проанализировать исторические данные акций\n"
        "2. Построить прогноз на 30 дней\n"
        "3. Дать торговые рекомендации\n"
        "4. Рассчитать потенциальную прибыль\n\n"
        "Пожалуйста, введите тикер компании (например, AAPL, GOOGL, TSLA):",
        reply_markup=ReplyKeyboardRemove()
    )

    return TICKER


async def get_tickers_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /get_tickers."""
    if not context.args:
        await update.message.reply_text(
            "Пожалуйста, укажите букву для поиска тикеров.\n"
            "Например: /get_tickers A\n"
            "Или: /get_tickers AAPL (для поиска по части тикера)"
        )
        return

    query = context.args[0].upper()

    # Если запрос - одна буква
    if len(query) == 1 and query.isalpha():
        tickers = ticker_manager.get_tickers_by_letter(query)
        if tickers:
            tickers_list = "\n".join([f"• {ticker}" for ticker in tickers])
            await update.message.reply_text(
                f"📊 Тикеры на букву '{query}':\n\n{tickers_list}\n\n"
                f"Всего найдено: {len(tickers)} тикеров\n"
                f"Для анализа выберите тикер и используйте /start"
            )
        else:
            await update.message.reply_text(
                f"Не найдено тикеров на букву '{query}'.\n"
                f"Попробуйте другую букву."
            )
    else:
        # Если запрос - часть тикера
        tickers = ticker_manager.search_tickers(query)
        if tickers:
            tickers_list = "\n".join([f"• {ticker}" for ticker in tickers])
            await update.message.reply_text(
                f"🔍 Результаты поиска для '{query}':\n\n{tickers_list}\n\n"
                f"Всего найдено: {len(tickers)} тикеров\n"
                f"Для анализа выберите тикер и используйте /start"
            )
        else:
            await update.message.reply_text(
                f"Не найдено тикеров по запросу '{query}'.\n"
                f"Попробуйте другой запрос."
            )


async def process_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка введенного тикера, запрос суммы инвестиции."""
    ticker = update.message.text.upper().strip()
    user = update.message.from_user

    # Сохраняем тикер в сессии пользователя
    user_sessions[user.id] = {'ticker': ticker}

    await update.message.reply_text(
        f"Отлично! Тикер: {ticker}\n"
        f"Загружаю исторические данные...",
        reply_markup=ReplyKeyboardRemove()
    )

    # Загружаем данные
    try:
        data = load_stock_data(ticker)
        if data.empty:
            await update.message.reply_text(
                f"Не удалось загрузить данные для тикера {ticker}.\n"
                f"Пожалуйста, проверьте правильность тикера и попробуйте снова."
            )
            return TICKER

        user_sessions[user.id]['data'] = data

        await update.message.reply_text(
            f"Данные успешно загружены! Период: {len(data)} дней.\n\n"
            f"Теперь введите сумму для условной инвестиции в USD (например, 1000):"
        )

        return AMOUNT

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        await update.message.reply_text(
            f"Произошла ошибка при загрузке данных: {str(e)[:100]}\n"
            f"Пожалуйста, попробуйте другой тикер."
        )
        return TICKER


# В функцию process_amount добавляем переход к запросу периода
async def process_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка суммы инвестиции, запрос периода прогноза."""
    user = update.message.from_user

    try:
        amount = float(update.message.text.replace(',', '.'))
        if amount <= 0:
            raise ValueError("Сумма должна быть положительной")
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите корректную сумму (число больше 0)."
        )
        return AMOUNT

    # Сохраняем сумму в сессии
    user_sessions[user.id]['amount'] = amount

    await update.message.reply_text(
        f"Сумма инвестиции: ${amount:,.2f}\n\n"
        f"Теперь укажите период прогнозирования в днях (например, 30):"
    )

    return FORECAST_DAYS


# Добавляем новую функцию для обработки периода
async def process_forecast_days(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка периода прогнозирования, запуск анализа."""
    user = update.message.from_user

    try:
        forecast_days = int(update.message.text.strip())
        if forecast_days < 1 or forecast_days > 365:
            raise ValueError("Период должен быть от 1 до 365 дней")
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите корректное количество дней (от 1 до 365)."
        )
        return FORECAST_DAYS

    # Сохраняем период в сессии
    user_sessions[user.id]['forecast_days'] = forecast_days

    # Отправляем сообщение о начале анализа
    processing_msg = await update.message.reply_text(
        f"🔍 Начинаю анализ...\n"
        f"• Тикер: {user_sessions[user.id]['ticker']}\n"
        f"• Сумма: ${user_sessions[user.id]['amount']:,.2f}\n"
        f"• Период прогноза: {forecast_days} дней\n\n"
        f"Это может занять несколько минут..."
    )

    try:
        # Получаем данные из сессии
        ticker = user_sessions[user.id]['ticker']
        data = user_sessions[user.id]['data']
        amount = user_sessions[user.id]['amount']

        # Шаг 1: Обучаем и сравниваем модели
        await processing_msg.edit_text("🔍 Начинаю анализ...\n1. 📊 Обучаю модели...")
        models_results = train_and_evaluate_models(data)

        # Шаг 2: Выбираем лучшую модель
        await processing_msg.edit_text("🔍 Начинаю анализ...\n2. ⚖️ Сравниваю метрики...")
        best_model_name, best_model, metrics = select_best_model(models_results)

        # Шаг 3: Делаем прогноз на указанный период
        await processing_msg.edit_text(f"🔍 Начинаю анализ...\n3. 🔮 Строю прогноз на {forecast_days} дней...")
        forecast = make_forecast(best_model, data, model_name=best_model_name, steps=forecast_days)

        # Шаг 4: Генерируем торговые сигналы
        await processing_msg.edit_text("🔍 Начинаю анализ...\n4. 📈 Анализирую сигналы...")
        signals = generate_trading_signals(forecast)

        # Шаг 5: Рассчитываем прибыль
        profit = calculate_profit(amount, forecast, signals)

        # Шаг 6: Создаем визуализацию
        await processing_msg.edit_text("🔍 Начинаю анализ...\n5. 🎨 Создаю график...")
        plot_path = create_forecast_plot(data, forecast, signals, ticker, forecast_days)

        # Шаг 7: Формируем финальный отчет
        last_price = data['Close'].iloc[-1]
        forecast_price = forecast.iloc[-1]
        price_change = (forecast_price - last_price) / last_price * 100

        buy_signals = sum(1 for s in signals if s['action'] == 'BUY')
        sell_signals = sum(1 for s in signals if s['action'] == 'SELL')

        report = f"""
📊 **ОТЧЕТ ПО АКЦИЯМ {ticker}**

📈 **Прогноз на {forecast_days} дней:**
• Текущая цена: {format_currency(last_price)}
• Прогноз через {forecast_days} дней: {format_currency(forecast_price)}
• Изменение: {format_percentage(price_change)}

🏆 **Лучшая модель:** {best_model_name}
• Метрика RMSE: {metrics['rmse']:.4f}
• Метрика MAPE: {metrics['mape']:.2f}%

🎯 **Торговые рекомендации:**
• Сигналов на ПОКУПКУ: {buy_signals}
• Сигналов на ПРОДАЖУ: {sell_signals}

💰 **Симуляция стратегии:**
• Начальный капитал: {format_currency(amount)}
• Конечный капитал: {format_currency(profit['final_amount'])}
• Прибыль: {format_currency(profit['profit_abs'])} ({format_percentage(profit['profit_pct'])})
"""

        # Отправляем график
        with open(plot_path, 'rb') as photo:
            await update.message.reply_photo(
                photo=photo,
                caption=report,
                parse_mode='Markdown'
            )

        # Логируем запрос
        log_request(
            user_id=user.id,
            timestamp=datetime.now(),
            ticker=ticker,
            amount=amount,
            forecast_days=forecast_days,
            best_model=best_model_name,
            metric_value=metrics['rmse'],
            profit=profit['profit_pct']
        )

        await processing_msg.delete()

        # Предлагаем новый анализ
        await update.message.reply_text(
            "Хотите проанализировать другую акцию? Напишите /start"
        )

        # Очищаем сессию пользователя
        if user.id in user_sessions:
            del user_sessions[user.id]

        return ConversationHandler.END

    except Exception as e:
        logger.error(f"Ошибка при анализе: {e}")
        await processing_msg.delete()
        await update.message.reply_text(
            f"Произошла ошибка при анализе: {str(e)[:200]}\n"
            f"Пожалуйста, попробуйте снова с другим тикером."
        )
        return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправка справки по команде /help."""
    await update.message.reply_text(
        "📈 **Бот для анализа акций**\n\n"
        "**Доступные команды:**\n"
        "/start - начать анализ акций\n"
        "/help - показать эту справку\n"
        "/cancel - отменить текущий анализ\n\n"
        "**Как использовать:**\n"
        "1. Нажмите /start\n"
        "2. Введите тикер акции (например, AAPL, TSLA, GOOGL)\n"
        "3. Введите сумму для инвестиции\n"
        "4. Получите прогноз и рекомендации\n\n"
        "**Примеры тикеров:**\n"
        "• AAPL - Apple\n"
        "• MSFT - Microsoft\n"
        "• GOOGL - Google\n"
        "• TSLA - Tesla\n"
        "• AMZN - Amazon",
        parse_mode='Markdown'
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена диалога."""
    user = update.message.from_user
    logger.info(f"Пользователь {user.first_name} отменил диалог.")

    # Очищаем сессию пользователя
    if user.id in user_sessions:
        del user_sessions[user.id]

    await update.message.reply_text(
        "Анализ отменен. Чтобы начать заново, нажмите /start",
        reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END