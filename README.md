# Movie Agent

## Setup Instructions

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your API keys
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Generate the movie index:
   ```bash
   python llama_index_builder.py
   ```
5. Launch the terminal chatbot:
   ```bash
   python terminal_chatbot.py
   ```
6. Launch the Telegram bot:
   ```bash
   python telegram_bot.py
   ```

## Important Notes
- Never commit the `.env` file
- Never commit `movie_index.pkl` or `movie_index.json`
- Each developer should generate their own index locally 
