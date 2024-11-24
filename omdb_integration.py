import os
import requests

def fetch_movie_details(title):
    """Fetch detailed information about a movie from OMDb API."""
    api_key = os.getenv("OMDB_API_KEY")
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    response = requests.get(url)
    return response.json() if response.ok else {"Error": "Failed to connect to OMDb API"}

def suggest_movies_by_genre(genre):
    """Fetch a list of movies in a specific genre using OMDb API."""
    api_key = os.getenv("OMDB_API_KEY")
    url = f"http://www.omdbapi.com/?s=&type=movie&genre={genre}&apikey={api_key}"
    response = requests.get(url)
    return [movie["Title"] for movie in response.json().get("Search", [])] if response.ok else []
