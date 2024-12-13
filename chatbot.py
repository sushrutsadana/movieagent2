import os
import openai
from dotenv import load_dotenv
from omdb_integration import fetch_movie_details
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
from pydantic import BaseModel
from llama_index_builder import load_index_from_disk
import logging
import json
from typing import Union, Optional
import csv
from booking_integration import book_tickets
from datetime import datetime, timedelta
import itertools

# Define welcome message
WELCOME_MESSAGE = """
🍿 Welcome to PopcornAI! 🎬

I'm your friendly movie assistant, here to help you discover and book movies in Bangalore. You can:
• Find movie showtimes by date, time, or location
• Search movies by language, genre, or theater
• Get movie reviews and details
• Book tickets for your favorite shows

Some example queries:
- "Show me movies playing today"
- "What are the evening shows for The Sabarmati Report?"
- "Tell me about theaters in Indiranagar"
- "Is Moana 2 playing this weekend?"

This chatbot is still in development. To give you the most accurate results, please provide as much context as possible

Currently serving Bangalore, with more cities coming soon!

How can I help you today?
"""

# Define your Pydantic model for structured output
class MovieIntent(BaseModel):
    intent: str
    movie_name: Optional[str] = None
    city: Optional[str] = None
    locality: Optional[str] = None
    cinema_name: Optional[str] = None
    showtime_str: Optional[str] = None
    genre: Optional[str] = None
    time_context: Optional[str] = None
    num_tickets: Optional[int] = None
    language: Optional[str] = None

# Define event types
class MovieReviewEvent(Event):
    pass

class ShowtimesEvent(Event):
    pass

class CinemaLocationEvent(Event):
    pass

class BookTicketsEvent(Event):
    pass

class GeneralEvent(Event):
    pass

# Load environment variables
logging.basicConfig(level=logging.INFO)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI LLM and index
index = load_index_from_disk("./movie_index")
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
query_engine = RetrieverQueryEngine(retriever=retriever)

# Define the prompt template for structured prediction
template_str = """
You are an AI assistant that helps with movie-related queries.

Current date and time: {current_datetime}

Known genres: action, thriller, comedy, drama, animation, adventure, horror, romance

Conversation History:
{history}

Your task:
- Analyze the latest user query: "{query}"
- Determine the user's intent from these options: movie_review, showtimes, cinema_location, book_tickets, general
- For general queries like greetings, help requests, or casual conversation, use intent "general"
- For movie reviews, extract the movie name
- For showtimes:
  * If a word matches a known genre, treat it as genre, not movie name
  * Extract movie name (if specific movie mentioned), locality, and language
  * Extract genre if mentioned (e.g., "action movies", "thriller films", etc.)
  * Understand temporal context from the current query (today, tomorrow, this weekend, next week, etc.)
  * Consider that weekend means Saturday and Sunday
  * If no time is explicitly mentioned, assume "today"

IMPORTANT: 
- If a word matches a known genre (action, thriller, comedy, etc.), set it as genre, not movie_name
- For queries like "show me thriller movies", set genre="thriller" and movie_name=null

Return a JSON that fits this structure:
{
    "intent": "movie_review|showtimes|cinema_location|book_tickets",
    "movie_name": "movie title if mentioned or from context",
    "city": "city if mentioned",
    "locality": "locality if mentioned",
    "cinema_name": "cinema if mentioned or from context",
    "showtime_str": "date, time if mentioned",
    "genre": "genre if mentioned (action, thriller, comedy, etc.)",
    "time_context": "temporal context from CURRENT QUERY ONLY",
    "num_tickets": "number if mentioned",
    "language": "movie language if mentioned or from context"
}
"""

# Create the PromptTemplate
prompt_template = PromptTemplate(template=template_str)

class ChatbotWorkflow(Workflow):
    @step
    async def start(self, event: StartEvent) -> Union[StopEvent, MovieReviewEvent, ShowtimesEvent, CinemaLocationEvent, BookTicketsEvent, GeneralEvent]:
        try:
            combined_input = event.input
            user_query = combined_input.split("\n")[-2].replace("User: ", "").strip()
            
            # Add logging for chat history
            logging.info("Current Chat History:")
            logging.info(combined_input)
            
            llm = OpenAI(model="gpt-4o-mini")
            Settings.llm = llm
            
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add logging for structured prediction
            parsed_query = llm.structured_predict(
                MovieIntent,
                prompt_template,
                query=user_query,
                history=combined_input,
                current_datetime=current_datetime
            )
            
            # Log the parsed query
            logging.info(f"Parsed Query with Context: {parsed_query.model_dump_json(indent=2)}")
            
            # Process the response based on intent
            intent = parsed_query.intent
            if intent == "general":
                return GeneralEvent(input=parsed_query.model_dump_json())
            elif intent == "movie_review":
                return MovieReviewEvent(input=parsed_query.model_dump_json())
            elif intent == "showtimes":
                return ShowtimesEvent(input=parsed_query.model_dump_json())
            elif intent == "cinema_location":
                return CinemaLocationEvent(input=parsed_query.model_dump_json())
            elif intent == "book_tickets":
                return BookTicketsEvent(input=parsed_query.model_dump_json())
            else:
                results = query_engine.query(user_query)
                return StopEvent(str(results))
                
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return StopEvent("Sorry, something went wrong on our end. Please try again.")

    @step
    async def handle_movie_review(self, event: MovieReviewEvent) -> StopEvent:
        parsed_query = json.loads(event.input)
        movie_name = parsed_query.get("movie_name")
        
        if not movie_name:
            return StopEvent("Please specify a movie title for reviews or details.")
        
        movie_details = fetch_movie_details(movie_name)
        
        if movie_details.get("Response", "False") == "True":
            # Format core movie information
            ratings = movie_details.get('Ratings', [])
            response_parts = [
                f"{movie_name.upper()} ({movie_details.get('Year', 'N/A')})"
            ]
            
            # Add ratings
            for rating in ratings:
                if rating['Source'] == 'Internet Movie Database':
                    response_parts.append(f"IMDB: {rating['Value']}")
                elif rating['Source'] == 'Rotten Tomatoes':
                    response_parts.append(f"Rotten Tomatoes: {rating['Value']}")
            
            # Add core movie details
            response_parts.extend([
                "",  # Empty line for spacing
                f"Genre: {movie_details.get('Genre', 'N/A')}",
                f"Plot: {movie_details.get('Plot', 'N/A')}",
                f"Cast: {movie_details.get('Actors', 'N/A')}",
                f"Director: {movie_details.get('Director', 'N/A')}"
            ])
            
            return StopEvent("\n".join(response_parts))
        else:
            return StopEvent(f"Sorry, I couldn't find information for '{movie_name}'.")

    @step
    async def handle_showtimes(self, event: ShowtimesEvent) -> StopEvent:
        try:
            parsed_query = json.loads(event.input)
            movie_name = parsed_query.get("movie_name")
            locality = parsed_query.get("locality")
            cinema_name = parsed_query.get("cinema_name")
            time_context = parsed_query.get("time_context", "today")
            genre = parsed_query.get("genre")
            language = parsed_query.get("language")
            
            # Read showtimes data
            with open("Data/Showtimessampledata.csv", "r") as file:
                csv_reader = csv.DictReader(file)
                all_showtimes = list(csv_reader)
            
            # Convert time_context to actual dates
            current_date = datetime.now().date()
            target_dates = []
            
            if time_context.lower() == "today":
                target_dates = [current_date]
            elif time_context.lower() == "tomorrow":
                target_dates = [current_date + timedelta(days=1)]
            elif time_context.lower() in ["weekend", "this weekend"]:
                days_until_saturday = (5 - current_date.weekday()) % 7
                saturday = current_date + timedelta(days=days_until_saturday)
                sunday = saturday + timedelta(days=1)
                target_dates = [saturday, sunday]
            elif time_context.lower() == "next week":
                target_dates = [current_date + timedelta(days=i) for i in range(1, 8)]
            else:
                target_dates = [current_date]  # Default to today
            
            # Filter showtimes
            filtered_showtimes = []
            for showtime in all_showtimes:
                try:
                    showtime_date = datetime.strptime(showtime['date'], '%Y-%m-%d').date()
                    
                    # Apply filters
                    matches = True
                    if movie_name and not any(movie_name.lower() in s.lower() for s in [showtime['movie_name'], showtime.get('alternate_titles', '')]):
                        matches = False
                    if cinema_name and showtime['theater_location'].lower() != cinema_name.lower():
                        matches = False
                    if language and showtime['language'].lower() != language.lower():
                        matches = False
                    if locality and locality.lower() not in showtime['address'].lower():
                        matches = False
                    if genre and genre.lower() not in showtime['genre'].lower():
                        matches = False
                    if showtime_date not in target_dates:
                        matches = False
                    
                    if matches:
                        filtered_showtimes.append(showtime)
                except Exception as e:
                    logging.error(f"Error processing showtime: {showtime}, Error: {str(e)}")
                    continue
            
            # Sort by theater and time
            filtered_showtimes.sort(key=lambda x: (
                x['theater_location'],
                x['date'],
                x['time']
            ))
            
            # Format results
            formatted_results = []
            
            # Add header
            header_parts = []
            if movie_name:
                header_parts.append(movie_name)
            if language:
                header_parts.append(f"in {language}")
            if genre:
                header_parts.append(f"({genre})")
            if cinema_name:
                header_parts.append(f"at {cinema_name}")
            
            header = "🎬 Showtimes for " + " ".join(header_parts) if header_parts else "Available Showtimes"
            formatted_results.append(header)
            formatted_results.append("-" * len(header))
            
            # Group by theater
            current_theater = None
            for showtime in filtered_showtimes:
                theater = showtime['theater_location']
                if theater != current_theater:
                    if current_theater is not None:
                        formatted_results.append("")
                    formatted_results.append(f"📍 {theater} - {showtime['address']}")
                    current_theater = theater
                
                formatted_results.append(
                    f" 🕒 {showtime['date']} at {showtime['time']} - "
                    f"{showtime['movie_name']} ({showtime['language']})"
                )
            
            if len(formatted_results) <= 2:  # Only header and separator
                return StopEvent("No showtimes found matching your criteria.")
            
            return StopEvent("\n".join(formatted_results))
            
        except Exception as e:
            logging.error(f"Error in handle_showtimes: {str(e)}")
            return StopEvent("Sorry, there was an error processing your request. Please try again.")

    @step
    async def handle_cinema_location(self, event: CinemaLocationEvent) -> StopEvent:
        try:
            parsed_query = json.loads(event.input)
            locality = parsed_query.get("locality")
            city = parsed_query.get("city")
            
            if not locality and not city:
                return StopEvent("Please specify a location (city or locality) to search for cinemas.")
            
            # Build a more specific query
            query = f"""Find cinemas where:
            {f"locality contains '{locality}'" if locality else ''}
            {f"AND city contains '{city}'" if city else ''}
            
            Format the response as a list with:
            - Cinema name
            - Full address
            - Available facilities
            - Contact information"""
            
            results = query_engine.query(query)
            
            # Format the response
            if not str(results).strip():
                return StopEvent(f"No cinemas found in {locality or city}.")
            
            formatted_results = ["🎬 Available Cinemas:", ""]
            
            for line in str(results).strip().split('\n'):
                if line.strip():
                    formatted_results.append(f"📍 {line.strip()}")
            
            return StopEvent("\n".join(formatted_results))
            
        except Exception as e:
            logging.error(f"Error in handle_cinema_location: {str(e)}")
            return StopEvent("Sorry, there was an error processing your request. Please try again.")

    @step
    async def handle_book_tickets(self, event: BookTicketsEvent) -> StopEvent:
        try:
            parsed_query = json.loads(event.input)
            movie_name = parsed_query.get("movie_name")
            cinema_name = parsed_query.get("cinema_name")
            showtime_str = parsed_query.get("showtime_str")
            num_tickets = parsed_query.get("num_tickets", 1)
            language = parsed_query.get("language")
            
            logging.info(f"Processing booking for: {movie_name} at {cinema_name}, {showtime_str}")
            
            if not all([movie_name, cinema_name, showtime_str]):
                missing = []
                if not movie_name: missing.append("movie name")
                if not cinema_name: missing.append("cinema name")
                if not showtime_str: missing.append("showtime")
                return StopEvent(f"Missing required information: {', '.join(missing)}")
            
            # Clean up cinema name (remove everything after "-")
            cinema_name = cinema_name.split(" - ")[0].strip()
            
            # Clean up the showtime string
            date, time = showtime_str.replace(',', '').split()
            
            # Fix language spelling and handle None
            if language:
                language_map = {
                    'telegu': 'telugu',
                    'kanada': 'kannada',
                    # Add more mappings if needed
                }
                language = language_map.get(language.lower(), language)
            
            # Read directly from CSV
            with open("Data/Showtimessampledata.csv", "r") as file:
                reader = csv.DictReader(file)
                showtimes = list(reader)
            
            # Find the exact showtime
            matching_show = None
            for show in showtimes:
                # Log the comparison for debugging
                logging.info(f"Comparing with show: {show}")
                
                # Handle language comparison with None
                language_matches = (
                    language is None or  # If no language specified, consider it a match
                    show['language'].lower() == language.lower()
                )
                
                if (show['movie_name'] == movie_name and
                    show['theater_location'] == cinema_name and
                    show['date'] == date and
                    show['time'] == time and
                    language_matches):
                    matching_show = show
                    logging.info(f"Found matching show: {show}")
                    break
            
            if matching_show:
                # Use the show's language if none was specified
                show_language = language or matching_show['language']
                
                success, message = book_tickets(
                    query_result=f"{date} {time} - {show_language} - {matching_show['available_seats']} seats",
                    num_tickets=num_tickets,
                    movie_name=movie_name,
                    theater_name=cinema_name
                )
                
                if success:
                    import random
                    import string
                    confirmation = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                    
                    return StopEvent(
                        f"Successfully booked {num_tickets} ticket(s) for {movie_name} "
                        f"at {cinema_name} for {showtime_str}\n\n"
                        f"🎟️ Confirmation Number: {confirmation}\n"
                        f"Please show this confirmation number at the theatre.\n\n"
                        f"🍿 Enjoy your show!"
                    )
                else:
                    return StopEvent(f"Booking failed: {message}")
            else:
                logging.warning(f"No matching showtime found for {movie_name} at {cinema_name} on {date} at {time}")
                return StopEvent("Could not find the specified showtime. Please verify the show details and try again.")
            
        except Exception as e:
            logging.error(f"Error in handle_book_tickets: {str(e)}")
            return StopEvent(f"An error occurred while processing your booking: {str(e)}")

    @step
    async def handle_general(self, event: GeneralEvent) -> StopEvent:
        return StopEvent(WELCOME_MESSAGE)

async def chat_with_user(question: str, history: list = None):
    workflow = ChatbotWorkflow()
    
    # Initialize history if None
    if history is None:
        history = []
    
    # Combine history and new question
    combined_input = "\n".join(history + [f"User: {question}", "Bot:"])
    
    try:
        result = await workflow.run(input=combined_input)
        return result.output if isinstance(result, StopEvent) else str(result)
    except Exception as e:
        logging.error(f"Error in chat_with_user: {str(e)}")
        return "Sorry, something went wrong. Please try again."

if __name__ == "__main__":
    import asyncio
    print(WELCOME_MESSAGE)
    print("\nType 'quit' to exit.")
    history = []
    MAX_HISTORY = 20  # Adjust as needed
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        history.append(f"User: {user_input}")
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
        response = asyncio.run(chat_with_user(user_input, history))
        print(f"Bot: {response}")
        history.append(f"Bot: {response}")
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]