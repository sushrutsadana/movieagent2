import os
import openai
from dotenv import load_dotenv
from fetch_data import fetch_cinemas, fetch_movies, fetch_showtimes_for_movie
from omdb_integration import fetch_movie_details, suggest_movies_by_genre
from llama_index_builder import load_index_from_disk
import asyncio

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the llama index and create a query engine
index = load_index_from_disk("movie_index.pkl")
query_engine = index.as_query_engine()


async def format_with_gpt(raw_response):
    """Format raw data into human-readable text using GPT."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful movie assistant."},
                {"role": "user", "content": f"Format this data: {raw_response}"},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error while fetching GPT response: {str(e)}"


async def query_llama_index(question):
    """Query LlamaIndex for contextual information."""
    try:
        results = query_engine.query(question)
        return str(results)
    except Exception as e:
        return f"Error querying LlamaIndex: {str(e)}"


async def handle_movie_reviews_or_details(question):
    """Handle queries related to movie reviews or details using OMDB."""
    if "reviews" in question.lower() or "details" in question.lower():
        title = question.split("details:")[1].strip() if "details:" in question else None
        if title:
            movie_details = fetch_movie_details(title)
            if "Error" in movie_details:
                return f"Could not fetch details for the movie '{title}'."
            formatted_details = await format_with_gpt(movie_details)
            return formatted_details
        else:
            return "Please specify a movie title for reviews or details (e.g., 'details: Inception')."


async def handle_showtimes_and_cinemas(question):
    """Handle queries related to showtimes or cinemas using Supabase."""
    if "showtimes" in question.lower() or "cinemas" in question.lower():
        city = question.split("in")[1].strip() if "in" in question.lower() else None
        preference = "evening" if "evening" in question.lower() else None
        if city:
            # Fetch data from the database
            cinemas = fetch_cinemas()
            movies = fetch_movies()
            showtimes = fetch_showtimes_for_movie()

            # Filter cinemas in the specified city
            cinemas_in_city = [cinema for cinema in cinemas if cinema["city"].lower() == city.lower()]
            if not cinemas_in_city:
                return f"No cinemas found in {city}."

            # Create mappings for easy ID-to-name lookup
            cinema_map = {cinema["id"]: cinema["cinema_name"] for cinema in cinemas}
            movie_map = {movie["id"]: movie["film_name"] for movie in movies}

            # Filter showtimes for relevant cinemas and add movie/cinema names
            filtered_showtimes = [
                f"{movie_map.get(showtime['film_id'], 'Unknown Movie')} "
                f"at {cinema_map.get(showtime['cinema_id'], 'Unknown Cinema')} "
                f"from {showtime['start_time']} to {showtime['end_time']}"
                for cinema in cinemas_in_city
                for showtime in showtimes
                if showtime["cinema_id"] == cinema["id"]
                and (not preference or preference.lower() in showtime["start_time"].lower())
            ]

            # Format the response
            if filtered_showtimes:
                raw_response = f"Showtimes in {city}: {', '.join(filtered_showtimes)}"
            else:
                raw_response = f"No showtimes available in {city}."

            return await format_with_gpt(raw_response)
        else:
            return "Please specify a city for showtimes or cinemas (e.g., 'showtimes in Bangalore')."


async def handle_general_queries(question):
    """Handle general queries using LlamaIndex."""
    return await query_llama_index(question)


async def chat_with_user(question):
    """Process user queries and dynamically decide the data source."""
    if "recommend a movie" in question.lower() or "details" in question.lower() or "reviews" in question.lower():
        return await handle_movie_reviews_or_details(question)
    elif "showtimes" in question.lower() or "cinemas" in question.lower():
        return await handle_showtimes_and_cinemas(question)
    else:
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
