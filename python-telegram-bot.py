from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from chatbot import ChatbotWorkflow
import asyncio
import os
from dotenv import load_dotenv
import logging
import csv
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Store chat histories
chat_histories = {}
MAX_HISTORY = 10

# Load movie data from CSV
available_movies = []
try:
    with open('./Data/Showtimessampledata.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            movie = row.get('movie_name', '').strip()
            if movie and movie not in available_movies:
                available_movies.append(movie)
except FileNotFoundError:
    logger.error("CSV file not found at ./Data/Showtimessampledata.csv")
    available_movies = ['Moana 2', 'Animal', 'KGF 3']

# Get three random movies for examples
example_movies = random.sample(available_movies, min(3, len(available_movies)))

WELCOME_MESSAGE = f"""
🍿 Welcome to PopcornAI! 🎬

I'm your friendly movie assistant, here to help you discover and book movies in Bangalore. You can:
• Find movie showtimes by date, time, or location
• Search movies by language, genre, or theater
• Get movie reviews and details
• Book tickets for your favorite shows

Some example queries:
- "Show me movies playing today"
- "What are the evening shows for {example_movies[0]}?"
- "Tell me about theaters in Indiranagar"
- "Book 2 tickets for {example_movies[1]} at PVR"
- "Is {example_movies[2]} playing this weekend?"

This chatbot is still in development. To give you the most accurate results, please provide as much context as possible:
• Movie name (e.g., "{example_movies[0]}")
• Cinema name (e.g., "PVR", "INOX")
• Number of tickets
• Preferred showtime
• Location/area in Bangalore

Currently serving Bangalore, with more cities coming soon!

How can I help you today?
"""


async def chat_with_user(question: str, history: list = None):
    if history is None:
        history = []
    
    logger.info(f"Processing query: {question}")
    logger.info(f"Current history: {history}")
    
    # Handle generic inputs and greetings
    generic_inputs = ['hello', 'hi', 'hey', 'help', 'start']
    if question.lower().strip() in generic_inputs or len(question.split()) < 2:
        return WELCOME_MESSAGE
    
    workflow = ChatbotWorkflow()
    
    # For booking requests, preserve the exact showtime context
    formatted_history = []
    if "book" in question.lower():
        # Find the last showtime interaction (both user query and bot response)
        showtime_query = None
        showtime_response = None
        
        for i in range(len(history)-1, -1, -1):
            msg = history[i]
            if "🎬" in msg and msg.startswith("Bot:"):
                showtime_response = msg.replace("Bot:", "").strip()
                if i > 0:  # Get the corresponding user query
                    showtime_query = history[i-1]
                break
        
        if showtime_query and showtime_response:
            formatted_history.extend([
                showtime_query,
                f"Bot: {showtime_response}"
            ])
    
    # Add the current query
    combined_input = "\n".join(formatted_history + [f"User: {question}", "Bot:"])
    
    logger.info(f"Formatted input to workflow: {combined_input}")
    
    try:
        result = await workflow.run(input=combined_input)
        response = result.output if hasattr(result, 'output') else str(result)
        
        logger.info(f"Raw response from workflow: {response}")
        
        if not response or response.isspace():
            return WELCOME_MESSAGE
            
        return response
        
    except Exception as e:
        logger.error(f"Error in chat_with_user: {str(e)}", exc_info=True)
        return WELCOME_MESSAGE

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_input = update.message.text.strip()
    
    logger.info(f"Received message from user {chat_id}: {user_input}")
    
    # Initialize or get chat history
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    
    try:
        # Handle generic inputs directly
        generic_inputs = ['hello', 'hi', 'hey', 'help', 'start']
        if user_input.lower().strip() in generic_inputs or len(user_input.split()) < 2:
            await update.message.reply_text(WELCOME_MESSAGE)
            return
            
        # Get response using the same format as terminal
        response = await chat_with_user(user_input, chat_histories[chat_id])
        
        # Log the response before sending
        logger.info(f"Response to be sent: {response}")
        
        # Update history with both user input and bot response
        chat_histories[chat_id].extend([
            f"User: {user_input}",
            f"Bot: {response}"
        ])
        
        # Keep only relevant context
        if len(chat_histories[chat_id]) > MAX_HISTORY:
            chat_histories[chat_id] = chat_histories[chat_id][-MAX_HISTORY:]
        
        await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        await update.message.reply_text(WELCOME_MESSAGE)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_MESSAGE)
    chat_id = update.effective_chat.id
    chat_histories[chat_id] = []

def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Telegram bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()