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
Chat History:
{history}

Known genres: action, thriller, comedy, drama, animation, adventure, horror, romance
Known languages: English, Hindi, Telugu, Tamil, Kannada, Malayalam
Known localities: Indiranagar, Koramangala, JP Nagar, Whitefield, Rajajinagar, Magrath Road

STRICT WEEKEND CALCULATION:
1. "this weekend" ALWAYS means the upcoming Saturday and Sunday
2. If today is:
   - Monday to Friday: use the upcoming Saturday/Sunday
   - Saturday: use today and tomorrow
   - Sunday: use yesterday and today
3. Example (if today is {current_datetime}):
   - "this weekend" -> "2024-12-14,2024-12-15" (next Saturday and Sunday)
   - NEVER use Monday as part of weekend

Your task:
- Analyze the latest user query: "{query}"
- Consider the previous messages in chat history for context
- If "weekend" is mentioned, ALWAYS return Saturday,Sunday dates
- Determine the user's intent from these options: movie_review, showtimes, cinema_location, book_tickets, general
- For general asks like (tell me about movie, what's the plot), intent will be movie_review

STRICT DATE HANDLING:
1. For Single Day Requests:
   - "this Sunday" -> use just that date: "YYYY-MM-DD"
   - Do NOT return two dates for single day requests
   
2. For Weekend Requests:
   - "this weekend" -> "YYYY-MM-DD,YYYY-MM-DD" (Saturday,Sunday)YYYY-MM-DD,YYYY-MM-DD"

LANGUAGE VS GENRE:
1. Language Detection:
   - Words like "Telugu", "Hindi", "English" are languages, NOT genres
   - Example: "Telugu movies" -> language: "Telugu", genre: null
   - Example: "Hindi films" -> language: "Hindi", genre: null

2. Genre Detection:
   - Only use known genres list: action, thriller, comedy, etc.
   - Example: "action movies" -> genre: "action", language: null

CONTEXT HANDLING RULES:
1. For Booking After Showtimes:
   - If user tries to book a show that was just displayed
   - Use ALL details from the previous message:
     * theater name
     * movie name
     * language
   - Example:
     Bot: "📍 INOX Lido - 17:30 - Bhool Bhulaiyaa 3 (Hindi)"
     User: "book me 2 tickets for this show" ->
       cinema_name: "INOX Lido"
       movie_name: "Bhool Bhulaiyaa 3"
       language: "Hindi"
       showtime_str: "2024-12-15 at 17:30"

2. Show Context Keywords:
   - "this show" -> use ALL details from the last displayed show
   - "for this" -> use ALL details from the last displayed show
   - "book me" -> use ALL details from the last displayed show

3. Location Context:
   - Keep theater and locality from previous message if relevant
   - Example:
     Bot: shows theaters in Indiranagar
     User: "book the 5 PM show" -> locality: "Indiranagar"

Your task:
- Analyze the latest user query: "{query}"
- Consider the previous messages in chat history for context
- If booking a specific showtime that was just displayed, use the theater from the previous context
- Determine the user's intent from these options: movie_review, showtimes, cinema_location, book_tickets, general

Examples (assuming today is {current_datetime}):
1. "Telugu movies this Sunday" ->
   intent: "showtimes"
   language: "Telugu"
   showtime_str: "2024-12-15"
   genre: null

2. "action movies in Hindi" ->
   intent: "showtimes"
   language: "Hindi"
   genre: "action"

3. "book the 5:15 PM show for Moana 2 at PVR Koramangala" ->
   intent: "book_tickets"
   movie_name: "Moana 2"
   cinema_name: "PVR Koramangala"
   locality: "Koramangala"
   showtime_str: "2024-12-15 at 17:15"

Return a JSON that fits this structure:
{
    "intent": "movie_review|showtimes|cinema_location|book_tickets|general",
    "movie_name": "movie title if mentioned",
    "city": "city if mentioned (default: Bangalore)",
    "locality": "area name if mentioned",
    "cinema_name": "cinema if mentioned",
    "showtime_str": "YYYY-MM-DD for single day, YYYY-MM-DD,YYYY-MM-DD for weekends",
    "genre": "genre if mentioned (action, thriller, etc.)",
    "time_context": "original temporal reference",
    "num_tickets": "number if mentioned",
    "language": "language if mentioned (Telugu, Hindi, etc.)"
}

CRITICAL CHECKS:
1. Single day requests return single date
2. Weekend requests return Saturday,Sunday dates
3. Telugu/Hindi/Tamil/etc. are languages, NOT genres
4. Only words from known genres list can be genres
5. Check language before checking genre
6. ALWAYS extract locality when mentioned with "in" or "at"
7. NEVER return null for locality if area is mentioned
8. For booking requests, if cinema_name is not in current query, CHECK PREVIOUS MESSAGE
9. Use theater name from previous bot message if booking a show that was just displayed
10. Booking requests MUST include "at HH:MM"
11. When user says "this show", COPY ALL DETAILS from last shown showtime
12. For booking context, movie_name should NEVER be null if show was just displayed
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
            showtime_str = parsed_query.get("showtime_str", "")
            genre = parsed_query.get("genre")
            language = parsed_query.get("language")
            
            logging.info(f"Processing showtimes with dates: {showtime_str}")
            
            # Read showtimes data
            with open("Data/Showtimessampledata.csv", "r") as file:
                csv_reader = csv.DictReader(file)
                filtered_showtimes = list(csv_reader)
            
            # Apply filters
            if movie_name:
                filtered_showtimes = [s for s in filtered_showtimes 
                                    if movie_name.lower() in s['movie_name'].lower()]
                logging.info(f"After movie filter: {len(filtered_showtimes)} shows")
            
            if locality:
                filtered_showtimes = [s for s in filtered_showtimes 
                                    if locality.lower() in s['address'].lower()]
                logging.info(f"After locality filter: {len(filtered_showtimes)} shows")
            
            if cinema_name:
                filtered_showtimes = [s for s in filtered_showtimes 
                                    if cinema_name.lower() in s['theater_location'].lower()]
                logging.info(f"After cinema filter: {len(filtered_showtimes)} shows")
            
            # Add genre filtering
            if genre:
                filtered_showtimes = [s for s in filtered_showtimes 
                                    if genre.lower() in s['genre'].lower()]
                logging.info(f"After genre filter: {len(filtered_showtimes)} shows")
            
            # Handle date filtering for both single dates and weekend ranges
            if showtime_str:
                target_dates = [d.strip() for d in showtime_str.split(',')]
                logging.info(f"Filtering for dates: {target_dates}")
                filtered_showtimes = [s for s in filtered_showtimes 
                                    if any(s['date'] == date for date in target_dates)]
                logging.info(f"After date filter: {len(filtered_showtimes)} shows")
            
            if language:
                filtered_showtimes = [s for s in filtered_showtimes 
                                    if language.lower() in s['language'].lower()]
                logging.info(f"After language filter: {len(filtered_showtimes)} shows")
            
            # Sort by theater and time
            filtered_showtimes.sort(key=lambda x: (
                x['theater_location'],
                x['date'],
                x['time']
            ))
            
            # Format results
            formatted_results = []
            
            # Add header with all available information
            header_parts = []
            if movie_name:
                header_parts.append(movie_name)
            if genre:
                header_parts.append(f"({genre})")
            if language:
                header_parts.append(f"in {language}")
            if cinema_name:
                header_parts.append(f"at {cinema_name}")
            if locality:
                header_parts.append(f"in {locality}")
            if showtime_str:
                dates = showtime_str.split(',')
                if len(dates) == 2:
                    header_parts.append(f"for this weekend ({dates[0]} - {dates[1]})")
                else:
                    header_parts.append(f"for {dates[0]}")
            
            header = "🎬 Showtimes" + (" for " + " ".join(header_parts) if header_parts else "")
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
            
            # Handle showtime string parsing with different formats
            showtime_str = showtime_str.replace(',', '').strip()
            
            if ' at ' in showtime_str:
                date, time = showtime_str.split(' at ')
            else:
                # If no 'at' separator, try to split on space and take first and last parts
                parts = showtime_str.split()
                if len(parts) >= 2:
                    date = parts[0]
                    time = parts[-1]
                else:
                    return StopEvent("Invalid showtime format. Expected format: 'YYYY-MM-DD at HH:MM' or 'YYYY-MM-DD HH:MM'")
            
            date = date.strip()
            time = time.strip()
            
            logging.info(f"Parsed date: {date}, time: {time}")
            
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

async def chat_with_user(question: str, history: list):
    workflow = ChatbotWorkflow()
    # Combine history and new question
    combined_input = "\n".join(history + [f"User: {question}", "Bot:"])
    result = await workflow.run(input=combined_input)
    return result.output if isinstance(result, StopEvent) else str(result)

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