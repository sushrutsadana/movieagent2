from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from chatbot import chat_with_user
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    try:
        # Call the chatbot logic
        response = await chat_with_user(user_input)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text("Sorry, something went wrong. Please try again later.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the user starts the bot."""
    await update.message.reply_text("Hello! I am your Movie Recommendation Chatbot. Ask me about movies, genres, or cinemas!")

def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    print("Telegram bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()