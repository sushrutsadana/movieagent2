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
import logging

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Add dictionary to store chat history per user
user_histories = {}
MAX_HISTORY = 20  # Same as in chatbot.py

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text

    # Initialize history for new users
    if user_id not in user_histories:
        user_histories[user_id] = []

    # Add user message to history
    user_histories[user_id].append(f"User: {user_input}")
    if len(user_histories[user_id]) > MAX_HISTORY:
        user_histories[user_id] = user_histories[user_id][-MAX_HISTORY:]

    try:
        # Call the chatbot logic with history
        response = await chat_with_user(user_input, user_histories[user_id])
        
        # Add bot response to history
        user_histories[user_id].append(f"Bot: {response}")
        if len(user_histories[user_id]) > MAX_HISTORY:
            user_histories[user_id] = user_histories[user_id][-MAX_HISTORY:]
            
        await update.message.reply_text(response)
    except Exception as e:
        logging.error(f"Error in handle_message: {str(e)}")
        await update.message.reply_text("Sorry, something went wrong. Please try again later.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the user starts the bot."""
    welcome_message = """
🍿 Welcome to PopcornAI! 🎬

I'm your friendly movie assistant, here to help you discover and book movies in Bangalore. You can:
• Find movie showtimes by date, time, or location
• Search movies by language, genre, or theater
• Get movie reviews and details
• Book tickets for your favorite shows

Some example queries:
- "Show me English movies playing today"
- "What are the evening shows for Oppenheimer?"
- "Tell me about theaters in Indiranagar"
- "Book 2 tickets for Dune at PVR"

Currently serving Bangalore, with more cities coming soon!

How can I help you today?
    """
    # Initialize history for new users
    user_id = update.effective_user.id
    user_histories[user_id] = []
    
    await update.message.reply_text(welcome_message)

def main():
    try:
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

        # Add command and message handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        # Start the bot with polling timeout
        print("Telegram bot is running...")
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            poll_interval=1.0,
            timeout=30
        )
    except Exception as e:
        logging.error(f"Bot error: {str(e)}")
    finally:
        if 'application' in locals():
            application.stop()

if __name__ == "__main__":
    main()