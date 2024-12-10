import os
import openai
from dotenv import load_dotenv
from omdb_integration import fetch_movie_details
from booking_integration import book_tickets, get_showtime_record
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
You are an AI assistant that extracts movie-related intents and details from user queries.

Your task:
- Read the user query.
- Determine the 'intent' (one of: movie_review, showtimes, cinema_location, book_tickets, or unknown).
- If the intent is movie_review, try to identify 'movie_name'.
- If the intent is showtimes or cinema_location or book_tickets, try to identify 'movie_name', 'city', 'locality', 'cinema_name', 'showtime_str', 'genre', 'time_context', and 'num_tickets' as applicable.
- If a field is not applicable or not provided, leave it as null.
- Return the result as a JSON object strictly matching the MovieIntent schema. Do not include extra fields.

User query: {user_query}

Return a JSON that fits the MovieIntent model.
"""

# Create the PromptTemplate
prompt_template = PromptTemplate(template=template_str)

class ChatbotWorkflow(Workflow):
    @step
    async def start(self, event: StartEvent) -> Union[StopEvent, MovieReviewEvent, ShowtimesEvent, CinemaLocationEvent, BookTicketsEvent]:
        try:
            user_query = event.input
            
            # Initialize the OpenAI model and set it in settings
            llm = OpenAI(model="gpt-4")
            Settings.llm = llm

            # Use structured prediction with the prompt template and pass user_query as a kwarg
            parsed_query = llm.structured_predict(MovieIntent, prompt_template, user_query=user_query)
            
            # Process the response based on intent
            intent = parsed_query.intent
            if intent == "movie_review":
                return MovieReviewEvent(input=parsed_query.json())
            elif intent == "showtimes":
                return ShowtimesEvent(input=parsed_query.json())
            elif intent == "cinema_location":
                return CinemaLocationEvent(input=parsed_query.json())
            elif intent == "book_tickets":
                return BookTicketsEvent(input=parsed_query.json())
            else:
                # Default fallback: Query LlamaIndex
                results = query_engine.query(user_query)
                return StopEvent(str(results))
                
        except Exception as e:
            logging.error(f"Error processing query: {e}")
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
        parsed_query = json.loads(event.input)
        movie_name = parsed_query.get("movie_name")
        locality = parsed_query.get("locality")
        
        # Query LlamaIndex for showtimes
        query = f"Show me showtimes for {movie_name or 'movies'} in {locality or 'your area'}"
        results = query_engine.query(query)
        return StopEvent(str(results))

    @step
    async def handle_cinema_location(self, event: CinemaLocationEvent) -> StopEvent:
        parsed_query = json.loads(event.input)
        locality = parsed_query.get("locality")
        city = parsed_query.get("city")
        
        # Query LlamaIndex for cinemas
        query = f"Tell me about cinemas in {locality or city or 'this area'}"
        results = query_engine.query(query)
        return StopEvent(str(results))

    @step
    async def handle_book_tickets(self, event: BookTicketsEvent) -> StopEvent:
        parsed_query = json.loads(event.input)
        movie_name = parsed_query.get("movie_name")
        cinema_name = parsed_query.get("cinema_name")
        showtime_str = parsed_query.get("showtime_str")
        num_tickets = parsed_query.get("num_tickets", 1)
        
        if not (movie_name and cinema_name and showtime_str):
            return StopEvent("Please specify the movie, cinema, and showtime.")
        
        # Validate and book tickets
        showtime_record = get_showtime_record(movie_name, cinema_name, showtime_str)
        if not showtime_record:
            return StopEvent("Sorry, that showtime is not available.")
        
        booking_id, error_msg = book_tickets(movie_name, cinema_name, showtime_str, num_tickets)
        if error_msg:
            return StopEvent(f"Error booking tickets: {error_msg}")
        
        return StopEvent(f"Booking confirmed! Booking ID: {booking_id}")

async def chat_with_user(question: str):
    workflow = ChatbotWorkflow()
    result = await workflow.run(input=question)
    return result.output if isinstance(result, StopEvent) else str(result)

if __name__ == "__main__":
    import asyncio
    print("Chatbot running. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        response = asyncio.run(chat_with_user(user_input))
        print(f"Bot: {response}")
