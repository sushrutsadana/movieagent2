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
from fastapi import FastAPI

# Create FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PORT = int(os.environ.get("PORT", 8080))

# Store chat histories
chat_histories = {}
MAX_HISTORY = 20

@app.get("/")
async def root():
    return {"status": "Bot is running"}

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
        print(f"Error in handle_message: {str(e)}")
        await update.message.reply_text("Sorry, something went wrong. Please try again later.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the user starts the bot."""
    user_id = update.effective_user.id
    
    # Initialize or reset chat history for this user
    chat_histories[user_id] = []
    
    await update.message.reply_text(WELCOME_MESSAGE)

def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot with webhook
    print(f"Telegram bot is running on port {PORT}...")
    
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()