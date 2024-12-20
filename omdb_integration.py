import os
import requests
import logging

# Configure logging
logger = logging.getLogger(__name__)

def fetch_movie_details(title):
    """Fetch detailed information about a movie from OMDb API."""
    api_key = os.getenv("OMDB_API_KEY")
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    
    logger.info(f"Fetching movie details for: {title}")
    logger.debug(f"OMDB API URL: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logger.info(f"OMDB API Response for {title}: {data}")
        
        # Check if movie was found
        if data.get("Response") == "False":
            logger.warning(f"Movie not found: {title}. Error: {data.get('Error')}")
            return {"Error": data.get("Error", "Movie not found")}
            
        # Process ratings
        imdb_rating = None
        rotten_tomatoes = None
        metacritic = None
        
        # Get IMDB rating directly
        if data.get("imdbRating") and data["imdbRating"] != "N/A":
            imdb_rating = data["imdbRating"]
            logger.debug(f"IMDB Rating found: {imdb_rating}")
        else:
            logger.debug(f"No IMDB Rating available. Value: {data.get('imdbRating')}")
            
        # Look through Ratings array for other sources
        logger.debug(f"Processing Ratings array: {data.get('Ratings', [])}")
        for rating in data.get("Ratings", []):
            if rating["Source"] == "Rotten Tomatoes":
                rotten_tomatoes = rating["Value"]
            elif rating["Source"] == "Metacritic":
                metacritic = rating["Value"]
        
        # Add processed ratings to the response
        data["ProcessedRatings"] = {
            "IMDB": imdb_rating,
            "RottenTomatoes": rotten_tomatoes,
            "Metacritic": metacritic
        }
        logger.debug(f"Processed ratings: {data['ProcessedRatings']}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching movie details: {e}")
        return {"Error": "Failed to connect to OMDb API"}