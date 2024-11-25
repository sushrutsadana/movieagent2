import os
import openai
from dotenv import load_dotenv
from fetch_data import fetch_cinemas, fetch_movies, fetch_showtimes_for_movie
from omdb_integration import fetch_movie_details
from llama_index_builder import load_index_from_disk
import asyncio
import logging
import random
import json

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the llama index and create a query engine
index = load_index_from_disk("movie_index.pkl")
query_engine = index.as_query_engine()


async def query_llama_index(question):
    """Query LlamaIndex for contextual information."""
    try:
        results = query_engine.query(question)
        return str(results)
    except Exception as e:
        return f"Error querying LlamaIndex: {str(e)}"


async def preprocess_query(question):
    """Use GPT to extract structured information from user query."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": """Extract structured information from movie-related queries.
                    Return a JSON object with these possible fields (include only if mentioned):
                    - intent: [movie_info, showtimes, cinema_location, movie_review, genre_search]
                    - movie_name: exact movie name if mentioned
                    - city: city name if mentioned
                    - locality: specific area/locality/neighborhood if mentioned (e.g., Koramangala, JP Nagar, Church Street, etc.)
                    - genre: movie genre if mentioned
                    - time_context: [currently_playing, evening, tomorrow, this_week]
                    
                    Examples:
                    "movies playing in koramangala" ->
                    {"intent": "showtimes", "locality": "koramangala", "time_context": "currently_playing"}
                    
                    "theatres in JP Nagar bangalore" ->
                    {"intent": "cinema_location", "city": "bangalore", "locality": "JP Nagar"}
                    
                    "showtimes for singham again in indiranagar" ->
                    {"intent": "showtimes", "movie_name": "singham again", "locality": "indiranagar"}
                    
                    "what movies are showing in church street" ->
                    {"intent": "showtimes", "locality": "church street", "time_context": "currently_playing"}
                    """
                },
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"}
        )
        
        parsed_query = json.loads(response.choices[0].message.content)
        logging.debug(f"Parsed query: {parsed_query}")
        return parsed_query

    except Exception as e:
        logging.error(f"Error in query preprocessing: {e}")
        return None


async def handle_movie_reviews_or_details(parsed_query):
    """Handle queries related to movie reviews, ratings, or details using OMDB."""
    movie_name = parsed_query.get("movie_name")
    
    if not movie_name:
        return "Please specify a movie title for reviews or details (e.g., 'review Inception')."
    
    # Fetch movie details from OMDB
    movie_details = fetch_movie_details(movie_name)
    
    # Debug logging
    logging.debug(f"OMDB Response for {movie_name}: {movie_details}")
    
    if "Error" in movie_details or not movie_details.get("Response", "False") == "True":
        return (f"Sorry, I couldn't find ratings for '{movie_name}'. "
               "This might be because the movie is too recent or not yet released.")

    # Extract ratings
    imdb_rating = movie_details.get("imdbRating", "N/A")
    rotten_tomatoes = next((rating["Value"] for rating in movie_details.get("Ratings", []) 
                          if rating["Source"] == "Rotten Tomatoes"), "N/A")
    
    # Build the response with available information
    response_parts = [f"Here are the details for '{movie_details.get('Title')}' ({movie_details.get('Year')}):"]
    
    if imdb_rating != "N/A":
        response_parts.append(f"IMDB Rating: {imdb_rating}/10")
    if rotten_tomatoes != "N/A":
        response_parts.append(f"Rotten Tomatoes: {rotten_tomatoes}")
    
    response_parts.extend([
        f"Genre: {movie_details.get('Genre', 'Not available')}",
        f"Director: {movie_details.get('Director', 'Not available')}",
        f"Actors: {movie_details.get('Actors', 'Not available')}",
        f"Plot: {movie_details.get('Plot', 'Not available')}"
    ])

    return "\n".join(response_parts)


async def handle_cinemas(parsed_query):
    """Handle cinema location queries using parsed information."""
    city = parsed_query.get("city")
    locality = parsed_query.get("locality")
    
    if not city and not locality:
        return "Please specify a city or locality."
    
    cinemas = fetch_cinemas()
    filtered_cinemas = cinemas
    
    # Filter by city if specified
    if city:
        filtered_cinemas = [
            cinema for cinema in filtered_cinemas 
            if city.lower() in cinema["city"].lower()
        ]
    
    # Filter by locality if specified
    if locality:
        filtered_cinemas = filter_by_locality(filtered_cinemas, locality)
    
    if not filtered_cinemas:
        location = f"{locality}, {city}" if city and locality else (locality or city)
        return f"No cinemas found in {location}."
    
    if len(filtered_cinemas) > 5:
        filtered_cinemas = random.sample(filtered_cinemas, 5)
    
    # Format response
    location_str = []
    if locality:
        location_str.append(locality.title())
    if city:
        location_str.append(city.title())
    
    response_parts = [f"Cinemas in {', '.join(location_str)}:"]
    for cinema in filtered_cinemas:
        response_parts.append(f"\n- {cinema['cinema_name']}")
        if cinema.get('address'):
            response_parts.append(f"  Address: {cinema['address']}")
    
    return "\n".join(response_parts)


async def handle_general_queries(question):
    """Handle general queries using LlamaIndex."""
    return await query_llama_index(question)


def filter_by_locality(items, locality, key_to_check=None):
    """Helper function to filter items by locality match in name or address."""
    if not locality:
        return items
    
    filtered_items = []
    locality_lower = locality.lower()
    
    for item in items:
        # Get the searchable text fields
        cinema_name = item.get("cinema_name", "").lower()
        address = item.get("address", "").lower()
        
        # Check if locality exists as a substring in either name or address
        if (locality_lower in cinema_name or 
            locality_lower in address):
            filtered_items.append(item)
    
    return filtered_items


async def handle_current_movies(parsed_query):
    """Handle queries about currently playing movies."""
    city = parsed_query.get("city")
    locality = parsed_query.get("locality")
    
    # Fetch data from the database
    cinemas = fetch_cinemas()
    movies = fetch_movies()
    showtimes = fetch_showtimes_for_movie()
    
    # Filter cinemas by city and/or locality
    filtered_cinemas = cinemas
    if city:
        filtered_cinemas = [
            cinema for cinema in filtered_cinemas 
            if city.lower() in cinema["city"].lower()
        ]
    
    # Further filter by locality if specified
    if locality:
        filtered_cinemas = filter_by_locality(filtered_cinemas, locality)
    
    if not filtered_cinemas:
        location = locality if locality else city
        return f"No cinemas found in {location}."
    
    # Create a mapping of movie IDs to their names and details
    movie_map = {movie["id"]: {
        "name": movie["film_name"],
        "version": movie.get("version_type", ""),
        "rating": movie.get("age_rating", "")
    } for movie in movies}
    
    # Create a dictionary to store movies and their showtimes
    movie_showtimes = {}
    
    # Filter showtimes for cinemas in the city and organize by movie
    for showtime in showtimes:
        if showtime["cinema_id"] in [c["id"] for c in filtered_cinemas]:
            movie_id = showtime["film_id"]
            if movie_id in movie_map:
                movie_name = movie_map[movie_id]["name"]
                cinema_name = next((c["cinema_name"] for c in cinemas if c["id"] == showtime["cinema_id"]), "Unknown Cinema")
                
                if movie_name not in movie_showtimes:
                    movie_showtimes[movie_name] = {
                        "version": movie_map[movie_id]["version"],
                        "rating": movie_map[movie_id]["rating"],
                        "shows": []
                    }
                
                movie_showtimes[movie_name]["shows"].append({
                    "cinema": cinema_name,
                    "time": showtime["start_time"]
                })
    
    if not movie_showtimes:
        return f"No movies currently playing in {city}."
    
    # Format the response
    response_parts = [f"Currently playing movies in {city}:"]
    
    for movie_name, details in movie_showtimes.items():
        # Add movie header with version and rating if available
        movie_header = [f"\n{movie_name}"]
        if details["version"]:
            movie_header.append(f" ({details['version']})")
        if details["rating"]:
            movie_header.append(f" - {details['rating']}")
        response_parts.append("".join(movie_header))
        
        # Group showtimes by cinema
        cinema_times = {}
        for show in details["shows"]:
            if show["cinema"] not in cinema_times:
                cinema_times[show["cinema"]] = []
            cinema_times[show["cinema"]].append(show["time"])
        
        # Add showtimes for each cinema
        for cinema, times in cinema_times.items():
            sorted_times = sorted(times)
            response_parts.append(f"- {cinema}:")
            response_parts.append(f"  {', '.join(sorted_times)}")
    
    return "\n".join(response_parts)


async def handle_movies_by_genre(question, genre):
    """Handle queries for movies of a specific genre currently playing."""
    # Fetch data from the database
    movies = fetch_movies()
    showtimes = fetch_showtimes_for_movie()
    
    # Count showtimes for each movie
    movie_showcount = {}
    for showtime in showtimes:
        movie_id = showtime['film_id']
        movie_showcount[movie_id] = movie_showcount.get(movie_id, 0) + 1
    
    # Sort movies by number of showtimes
    popular_movies = sorted(
        [(movie_id, count) for movie_id, count in movie_showcount.items()],
        key=lambda x: x[1],
        reverse=True
    )[:6]  # Get top 6 movies with most shows
    
    # Get movie details for these popular movies
    genre_matched_movies = []
    for movie_id, _ in popular_movies:
        movie = next((m for m in movies if m['id'] == movie_id), None)
        if movie:
            # Get OMDB details to check genre
            omdb_details = fetch_movie_details(movie['film_name'])
            if ("Error" not in omdb_details and 
                genre.lower() in omdb_details.get("Genre", "").lower()):
                genre_matched_movies.append({
                    "Title": movie['film_name'],
                    "Genre": omdb_details.get("Genre", "N/A"),
                    "Director": omdb_details.get("Director", "N/A"),
                    "Actors": omdb_details.get("Actors", "N/A"),
                    "Plot": omdb_details.get("Plot", "N/A"),
                    "imdbRating": omdb_details.get("imdbRating", "N/A"),
                    "show_count": movie_showcount[movie_id]
                })
    
    if not genre_matched_movies:
        return f"No {genre} movies currently playing."
    
    # Format the response
    response_parts = [f"Currently playing {genre} movies:"]
    for movie in genre_matched_movies:
        response_parts.extend([
            f"\n{movie['Title']}",
            f"- Genre: {movie['Genre']}",
            f"- Director: {movie['Director']}",
            f"- Stars: {movie['Actors']}",
            f"- IMDB Rating: {movie['imdbRating']}/10" if movie['imdbRating'] != "N/A" else "",
            f"- Number of shows today: {movie['show_count']}"
        ])
    
    return "\n".join(filter(None, response_parts))


async def handle_movie_showtimes(parsed_query):
    """Handle queries for showtimes of a specific movie."""
    movie_name = parsed_query.get("movie_name")
    locality = parsed_query.get("locality")
    
    if not movie_name:
        return "Please specify a movie title."
    
    # Fetch data
    movies = fetch_movies()
    showtimes = fetch_showtimes_for_movie()
    cinemas = fetch_cinemas()
    
    # Find the movie
    movie = next(
        (m for m in movies if movie_name.lower() in m["film_name"].lower() 
         or m["film_name"].lower() in movie_name.lower()),
        None
    )
    
    if not movie:
        return f"No showtimes found for '{movie_name}'."
    
    # Filter cinemas by locality if specified
    filtered_cinemas = cinemas
    if locality:
        filtered_cinemas = filter_by_locality(cinemas, locality)
        if not filtered_cinemas:
            return f"No cinemas found in {locality} showing {movie_name}."
    
    # Create cinema mapping using filtered cinemas
    cinema_map = {cinema["id"]: cinema for cinema in filtered_cinemas}
    
    # Get showtimes for this movie in filtered cinemas
    movie_showtimes = [
        showtime for showtime in showtimes 
        if showtime["film_id"] == movie["id"] and
        showtime["cinema_id"] in cinema_map
    ]
    
    if not movie_showtimes:
        return f"No showtimes available for '{movie['film_name']}'."
    
    # Group showtimes by cinema
    cinema_showtimes = {}
    for showtime in movie_showtimes:
        cinema_id = showtime["cinema_id"]
        if cinema_id in cinema_map:
            cinema_name = cinema_map[cinema_id]["cinema_name"]
            if cinema_name not in cinema_showtimes:
                cinema_showtimes[cinema_name] = []
            cinema_showtimes[cinema_name].append(showtime["start_time"])
    
    # Format the response
    response_parts = [f"Showtimes for '{movie['film_name']}':"]
    for cinema, times in cinema_showtimes.items():
        response_parts.append(f"\n{cinema}:")
        # Sort times and format them
        sorted_times = sorted(times)
        response_parts.append("- " + ", ".join(sorted_times))
    
    return "\n".join(response_parts)


async def chat_with_user(question):
    """Process user queries using the preprocessed structured information."""
    # First, preprocess the query to extract structured information
    parsed_query = await preprocess_query(question)
    if not parsed_query:
        return "I'm having trouble understanding your query. Could you rephrase it?"
    
    # Route to appropriate handler based on intent
    intent = parsed_query.get("intent")
    
    if intent == "movie_review":
        return await handle_movie_reviews_or_details(parsed_query)
    
    elif intent == "showtimes":
        if parsed_query.get("movie_name"):
            return await handle_movie_showtimes(parsed_query)
        else:
            return await handle_current_movies(parsed_query)
    
    elif intent == "cinema_location":
        return await handle_cinemas(parsed_query)
    
    elif intent == "genre_search":
        genre = parsed_query.get("genre")
        if genre:
            return await handle_movies_by_genre(question, genre)
    
    return await handle_general_queries(question)


async def chatbot():
    """Interactive console chatbot for testing."""
    print("Welcome to the Movie Chatbot!")
    while True:
        question = input("Ask me about movies, genres, or cinemas: ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = await chat_with_user(question)
        print(response)


if __name__ == "__main__":
    asyncio.run(chatbot())
