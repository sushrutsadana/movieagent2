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

## Important Notes
- Never commit the `.env` file
- Never commit `movie_index.pkl` or `movie_index.json`
- Each developer should generate their own index locally 