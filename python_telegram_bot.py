from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from chatbot import chat_with_user, WELCOME_MESSAGE
import asyncio
import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, Request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PORT = int(os.environ.get("PORT", 8080))

# Get your render.com URL from environment or use localhost for local testing
WEBHOOK_URL = os.getenv("WEBHOOK_URL", f"https://your-app-name.onrender.com")

# Store chat histories
chat_histories = {}
MAX_HISTORY = 20

# Initialize bot application globally
application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

@app.get("/")
async def root():
    return {"status": "Bot is running"}

@app.post(f"/{TELEGRAM_BOT_TOKEN}")
async def webhook_handler(request: Request):
    """Handle incoming Telegram updates"""
    try:
        data = await request.json()
        async with application:
            update = Update.de_json(data, application.bot)
            await application.initialize()
            await application.process_update(update)
        return {"ok": True}
    except Exception as e:
        logger.error(f"Error processing update: {str(e)}")
        return {"ok": False, "error": str(e)}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text
    
    # Initialize chat history for new users
    if user_id not in chat_histories:
        chat_histories[user_id] = []
    
    # Add user message to history
    chat_histories[user_id].append(f"User: {user_input}")
    
    try:
        # Call the chatbot logic with history
        response = await chat_with_user(user_input, chat_histories[user_id])
        
        # Add bot response to history
        chat_histories[user_id].append(f"Bot: {response}")
        
        # Trim history if too long
        if len(chat_histories[user_id]) > MAX_HISTORY:
            chat_histories[user_id] = chat_histories[user_id][-MAX_HISTORY:]
        
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        await update.message.reply_text("Sorry, something went wrong. Please try again later.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the user starts the bot."""
    user_id = update.effective_user.id
    chat_histories[user_id] = []
    await update.message.reply_text(WELCOME_MESSAGE)

async def setup_webhook():
    """Setup webhook for the bot"""
    webhook_url = f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}"
    
    async with application:
        await application.initialize()
        webhook_info = await application.bot.get_webhook_info()
        
        # Only set webhook if it's not already set correctly
        if webhook_info.url != webhook_url:
            await application.bot.set_webhook(url=webhook_url)
            logger.info(f"Webhook set to {webhook_url}")

def main():
    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Setup webhook
    asyncio.run(setup_webhook())

    # Start FastAPI server
    logger.info(f"Starting bot on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()