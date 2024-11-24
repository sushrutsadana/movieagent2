from dotenv import load_dotenv
import os
import pickle
from fetch_data import fetch_movies, fetch_cinemas, fetch_showtimes_for_movie
from llama_index.core import Document
from llama_index.core.indices.vector_store import VectorStoreIndex

load_dotenv()  # Load environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Get the API key

def create_index():
    """Create index without storing any sensitive data."""
    # Fetch data from the database
    movies = fetch_movies()
    cinemas = fetch_cinemas()
    showtimes = fetch_showtimes_for_movie()

    documents = []

    # Only store public information
    for movie in movies:
        text = (
            f"Movie: {movie['film_name']}, "
            f"Age Rating: {movie['age_rating']}"
        )
        documents.append(Document(text=text))

    for cinema in cinemas:
        text = (
            f"Cinema: {cinema['cinema_name']} (City: {cinema['city']}), "
            f"Address: {cinema['address']}"
        )
        documents.append(Document(text=text))

    # Build the index
    index = VectorStoreIndex.from_documents(documents)
    
    # Save the index
    with open("movie_index.pkl", "wb") as f:
        pickle.dump(index, f)
    print("Index saved successfully.")


def load_index_from_disk(filename="movie_index.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    create_index()