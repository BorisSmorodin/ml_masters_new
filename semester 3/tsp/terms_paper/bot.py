import logging

from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler

from core import bot_handlers
from core.bot_handlers import start, help_command, cancel, process_ticker, process_amount,TICKER, AMOUNT
from config import BOT_TOKEN

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Запуск бота."""
    # Загружаем переменные окружения

    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN не найден в переменных окружения!")

    # Создаем Application
    application = Application.builder().token(BOT_TOKEN).build()

    # Создаем ConversationHandler для диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', bot_handlers.start)],
        states={
            bot_handlers.TICKER: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot_handlers.process_ticker)
            ],
            bot_handlers.AMOUNT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot_handlers.process_amount)
            ],
            bot_handlers.FORECAST_DAYS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot_handlers.process_forecast_days)
            ],
        },
        fallbacks=[CommandHandler('cancel', bot_handlers.cancel)],
    )

    # Регистрируем обработчики
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(CommandHandler("get_tickers", bot_handlers.get_tickers_command))

    # Запускаем бота
    print("Бот запущен...")
    application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == '__main__':
    main()