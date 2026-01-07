import logging

from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler
from dotenv import load_dotenv, find_dotenv

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
        entry_points=[CommandHandler('start', start)],
        states={
            TICKER: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_ticker)
            ],
            AMOUNT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_amount)
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    # Регистрируем обработчики
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cancel", cancel))

    # Запускаем бота
    print("Бот запущен...")
    application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == '__main__':
    main()