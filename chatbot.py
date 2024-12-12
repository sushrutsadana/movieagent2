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
import random

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

Conversation History:
{history}

Your task:
- Analyze the latest user query: "{query}"
- Determine the user's intent from these options: movie_review, showtimes, cinema_location, book_tickets
- For movie reviews, extract the movie name
- For showtimes:
  * Extract movie name, locality, and language (if mentioned)
  * Understand temporal context ONLY from the current query (today, tomorrow, this weekend, next week, etc.)
  * Consider that weekend means Saturday and Sunday
  * If no time is explicitly mentioned in current query, assume "today"
- For cinema locations, extract city and locality
- For bookings, extract movie name, cinema name, showtime, language, and number of tickets

If any data is not available in the current query (EXCEPT time_context), try extracting it from the previous chat history.
Time context should ONLY be derived from the current query.

Return a JSON that fits this structure:
{
    "intent": "movie_review|showtimes|cinema_location|book_tickets",
    "movie_name": "movie title if mentioned",
    "city": "city if mentioned",
    "locality": "locality if mentioned",
    "cinema_name": "cinema if mentioned",
    "showtime_str": "date, time if mentioned",
    "time_context": "temporal context from CURRENT QUERY ONLY",
    "num_tickets": "number if mentioned",
    "language": "movie language if mentioned"
}
"""

# Create the PromptTemplate
prompt_template = PromptTemplate(template=template_str)

class ChatbotWorkflow(Workflow):
    @step
    async def start(self, event: StartEvent) -> Union[StopEvent, MovieReviewEvent, ShowtimesEvent, CinemaLocationEvent, BookTicketsEvent]:
        try:
            combined_input = event.input
            user_query = combined_input.split("\n")[-2].replace("User: ", "").strip()
            
            llm = OpenAI(model="gpt-4o-mini")
            Settings.llm = llm
          
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # For booking-related queries, keep the history
            is_booking_related = any(word in user_query.lower() for word in ['book', 'ticket', 'seats', 'booking'])
            history_context = combined_input if is_booking_related else ""
            
            # Add logging for structured prediction
            parsed_query = llm.structured_predict(
                MovieIntent,
                prompt_template,
                query=user_query,
                history=history_context,  # Only include history for booking queries
                current_datetime=current_datetime
            )
            
            # Log the parsed query
            logging.info(f"Parsed Query: {parsed_query.model_dump_json(indent=2)}")
            
            # Process the response based on intent
            intent = parsed_query.intent
            if intent == "movie_review":
                return MovieReviewEvent(input=parsed_query.model_dump_json())
            elif intent == "showtimes":
                # Clear locality and cinema_name if not explicitly mentioned in current query
                if not any(word in user_query.lower() for word in ['in', 'at', 'near']):
                    parsed_query.locality = None
                    parsed_query.cinema_name = None
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
            time_context = parsed_query.get("time_context", "today")
            genre = parsed_query.get("genre")
            language = parsed_query.get("language")
            
            # Let the LLM construct the appropriate date-based query
            date_prompt = PromptTemplate("""
            Given:
            - Time context: '{time_context}'
            - Movie: '{movie_name}'
            - Genre: '{genre}'
            - Current date: {current_date}
            
            Construct a natural query to find showtimes that match these criteria.
            
            Examples:
            - For "today" → show only {current_date} showtimes
            - For "tomorrow" → show only {tomorrow_date} showtimes
            - For "this weekend" → show Saturday and Sunday showtimes
            - For "next week" → show next week's showtimes
            
            Return a natural language query that our database would understand.
            """)
            
            current_date = datetime.now()
            tomorrow_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
            current_date = current_date.strftime("%Y-%m-%d")
            
            # Fix the date query completion and add error handling
            try:
                date_query_response = Settings.llm.complete(
                    date_prompt.format(
                        time_context=time_context,
                        movie_name=movie_name or "any movie",
                        genre=genre or "any genre",
                        current_date=current_date,
                        tomorrow_date=tomorrow_date
                    )
                )
                date_query = str(date_query_response).strip()
                logging.info(f"Date query response: {date_query}")
                
                if not date_query:  # Fallback if date query is empty
                    date_query = f"date is {current_date}"
            except Exception as e:
                logging.error(f"Error getting date query: {str(e)}")
                date_query = f"date is {current_date}"  # Fallback to today
            
            # Log for debugging
            logging.info(f"Time Context: {time_context}")
            logging.info(f"Movie Name: {movie_name}")
            logging.info(f"Genre: {genre}")
            logging.info(f"Language: {language}")
            logging.info(f"Final Date Query: {date_query}")
            
            # Construct the search query
            query_parts = []
            if movie_name:
                query_parts.append(f"movie_name contains '{movie_name}'")
            if language:
                query_parts.append(f"language is '{language}'")
            if locality:
                # Make the locality search stricter
                query_parts.append(f"(theater_location ILIKE '%{locality}%' OR address ILIKE '%{locality}%')")
            
            # Add the date query
            query_parts.append(date_query)
            
            # Build the final query with explicit instructions
            query = f"""Find up to 6 showtimes where ALL of these conditions are true:
{' AND '.join(query_parts)}

Important:
- Only return showtimes where theater_location or address contains '{locality}'
- Return multiple showtimes if available (up to 6)
- Return results in CSV format with these exact columns:
  theater_location, address, city, state, movie_name, language, genre, date, time

Sort results by date, and time."""
            
            logging.info(f"Constructed query: {query}")
            
            # Increase the number of results to search through
            retriever.similarity_top_k = 10
            results = str(query_engine.query(query))
            logging.info(f"Raw results: {results}")
            
            if not str(results).strip():
                return StopEvent(f"No showtimes found for {movie_name or 'movies'} in {locality} {time_context}.")
            
            # Post-process results to ensure locality match
            formatted_results = []
            raw_results = str(results).strip().split('\n')
            current_theater = None
            
            # Add a header with search criteria
            header_parts = []
            if movie_name:
                header_parts.append(movie_name)
            if language:
                header_parts.append(f"in {language}")
            if locality:
                header_parts.append(f"in {locality}")
            
            header = "Showtimes for " + " ".join(header_parts) if header_parts else "Available Showtimes"
            formatted_results.append(header)
            formatted_results.append("-" * len(header))
            
            # Track unique showtimes to avoid duplicates
            seen_showtimes = set()
            
            for line in raw_results:
                if line.strip():
                    try:
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) >= 9:
                            theater = parts[0]
                            address = parts[1]
                            
                            # Skip if theater doesn't match locality
                            if locality and locality.lower() not in theater.lower() and locality.lower() not in address.lower():
                                logging.info(f"Skipping non-matching theater: {theater}")
                                continue
                            
                            movie = parts[4]
                            lang = parts[5]
                            date = parts[7]
                            time = parts[8]
                            
                            # Create unique key for showtime
                            showtime_key = f"{theater}-{movie}-{date}-{time}"
                            if showtime_key in seen_showtimes:
                                continue
                            seen_showtimes.add(showtime_key)
                            
                            if theater != current_theater:
                                if current_theater is not None:
                                    formatted_results.append("")
                                formatted_results.append(f"📍 - {theater} ({address})")
                                current_theater = theater
                            
                            formatted_results.append(
                                f"   🎬 {movie} ({lang}) - {date} at {time}"
                            )
                    except Exception as e:
                        logging.error(f"Error parsing line: {line}")
                        logging.error(f"Error details: {str(e)}")
                        continue
            
            response = "\n".join(formatted_results)
            logging.info(f"Formatted response: {response}")
            return StopEvent(response)
            
        except Exception as e:
            logging.error(f"Error in handle_showtimes: {str(e)}")
            return StopEvent("Sorry, I couldn't find any showtimes matching your criteria.")

    @step
    async def handle_cinema_location(self, event: CinemaLocationEvent) -> StopEvent:
        parsed_query = json.loads(event.input)
        locality = parsed_query.get("locality")
        city = parsed_query.get("city")
        
        # Simplify the query to reduce processing time
        query = f"""Find cinemas where:
        {f"locality contains '{locality}'" if locality else ''}
        {f"AND city is '{city}'" if city else ''}
        
        Return only:
        cinema_name, address
        Format as: cinema_name, full_address
        """
        
        # Use direct index lookup if possible
        if locality:
            retriever.similarity_top_k = 3
        
        results = query_engine.query(query)
        
        if not str(results).strip():
            return StopEvent(f"Sorry, I couldn't find any cinemas in {locality or city or 'this area'}.")
        
        try:
            formatted_results = ["🎬 Available Cinemas"]
            formatted_results.append("=" * 20)
            
            seen_cinemas = set()
            for cinema in str(results).split('\n'):
                if not cinema.strip():
                    continue
                
                # First split by comma, then clean each part
                parts = [part.strip() for part in cinema.split(',', 1)]  # Split into max 2 parts
                
                # Only process if we have all required parts
                if len(parts) == 2:  # Must have exactly 2 parts
                    cinema_name = parts[0]
                    address = parts[1]
                    
                    # Skip duplicates
                    if cinema_name in seen_cinemas:
                        continue
                    seen_cinemas.add(cinema_name)
                    
                    formatted_results.extend([
                        f"\n📍 {cinema_name}",
                        f"   Address: {address}"
                    ])
            
            # If no valid cinemas were found after filtering
            if len(formatted_results) <= 2:  # Only has header and separator
                return StopEvent(f"No valid cinema information found in {locality or city or 'this area'}.")
            
            return StopEvent("\n".join(formatted_results))
            
        except Exception as e:
            logging.error(f"Error formatting cinema locations: {str(e)}")
            return StopEvent("Sorry, there was an error formatting the cinema information. Please try again.")

    @step
    async def handle_book_tickets(self, event: BookTicketsEvent) -> StopEvent:
        try:
            parsed_query = json.loads(event.input)
            movie_name = parsed_query.get("movie_name")
            cinema_name = parsed_query.get("cinema_name")
            showtime_str = parsed_query.get("showtime_str")
            num_tickets = parsed_query.get("num_tickets", 1)
            language = parsed_query.get("language")
            
            logging.info(f"Parsed query: {parsed_query}")
            
            if not all([movie_name, cinema_name]):
                missing = []
                if not movie_name: missing.append("movie name")
                if not cinema_name: missing.append("cinema location")
                return StopEvent(f"Missing required information: {', '.join(missing)}")
            
            # First, query all available showtimes
            query = f"""Find all showtimes where:
            theater_location is exactly '{cinema_name}' AND
            movie_name is exactly '{movie_name}'
            {f"AND language is '{language}'" if language else ''}
            
            Return all matching showtimes."""
            
            results = query_engine.query(query)
            
            # If multiple showtimes exist and specific time not provided
            if str(results).count('\n') > 1 and not showtime_str:
                return StopEvent(
                    f"Multiple showtimes available for {movie_name} at {cinema_name}. "
                    f"Please specify which showtime you'd like to book:\n{results}"
                )
            
            # If specific showtime provided, proceed with booking
            if showtime_str:
                # Clean up showtime string and extract date and time
                showtime_str = showtime_str.replace(',', '')
                if 'at' in showtime_str:
                    date, time = showtime_str.split('at')
                else:
                    date, time = showtime_str.split(' ', 1)
                
                date = date.strip()
                time = time.strip()
                
                logging.info(f"Searching for showtime: date={date}, time={time}")
                
                specific_query = f"""Find the exact showtime where:
                theater_location is exactly '{cinema_name}' AND
                movie_name is exactly '{movie_name}' AND
                date is exactly '{date}' AND
                time is exactly '{time}'
                
                Return only the matching CSV row."""
                
                logging.info(f"Booking query: {specific_query}")
                specific_results = str(query_engine.query(specific_query))
                logging.info(f"Booking query results: {specific_results}")
                
                if specific_results.strip():
                    try:
                        logging.info(f"Attempting to book tickets with result: {specific_results}")
                        # Call book_tickets function to handle seat availability
                        booking_result = book_tickets(
                            query_result=specific_results,
                            num_tickets=num_tickets
                        )
                        
                        logging.info(f"Booking result: {booking_result}")
                        
                        # Handle the booking result
                        if isinstance(booking_result, tuple):
                            success, message = booking_result
                        else:
                            success = True
                            message = "Booking successful"
                        
                        if success:
                            # Generate a random confirmation number
                            import random
                            import string
                            confirmation = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                            
                            return StopEvent(
                                f"✅ Successfully booked {num_tickets} ticket(s) for {movie_name} "
                                f"at {cinema_name} for {date} at {time}\n\n"
                                f"🎟️ Confirmation Number: {confirmation}\n"
                                f"Please show this confirmation number at the theatre.\n\n"
                                f"🍿 Enjoy your show!"
                            )
                        else:
                            return StopEvent(f"Booking failed: {message}")
                    except Exception as booking_error:
                        logging.error(f"Error during booking: {booking_error}")
                        return StopEvent("Error processing booking. Please try again.")
                else:
                    return StopEvent("Could not find that exact showtime. Please check the available showtimes and try again.")
            
            return StopEvent("Please specify which showtime you'd like to book.")
                
        except Exception as e:
            logging.error(f"Error in handle_book_tickets: {str(e)}")
            return StopEvent("Please check the available showtimes first and then book using the exact time shown.")

async def chat_with_user(question: str, history: list):
    workflow = ChatbotWorkflow()
    # Combine history and new question
    combined_input = "\n".join(history + [f"User: {question}", "Bot:"])
    result = await workflow.run(input=combined_input)
    return result.output if isinstance(result, StopEvent) else str(result)

if __name__ == "__main__":
    import asyncio
    import csv
    import random

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
        print("CSV file not found at ./Data/Showtimessampledata.csv")
        available_movies = ['Moana 2', 'Pushpa 2', 'The Sabarmati Report']

    # Get three random movies for examples
    example_movies = random.sample(available_movies, min(3, len(available_movies)))

    welcome_message = f"""
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
    print(welcome_message)
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
