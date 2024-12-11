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
- Read the user query carefully.
- For booking requests, extract ALL of these fields:
  * intent: "book_tickets"
  * movie_name: The exact movie title
  * cinema_name: The exact theater name
  * showtime_str: The date and time in format "YYYY-MM-DD,HH:MM"
  * num_tickets: Number of tickets requested (default to 1 if not specified)

Example booking query:
"Book me 3 tickets for The Sabarmati Report at PVR Global Mall on 2024-12-15,15:30"
Should extract:
{
  "intent": "book_tickets",
  "movie_name": "The Sabarmati Report",
  "cinema_name": "PVR Global Mall",
  "showtime_str": "2024-12-15,15:30",
  "num_tickets": 3
}

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
        try:
            parsed_query = json.loads(event.input)
            movie_name = parsed_query.get("movie_name")
            cinema_name = parsed_query.get("cinema_name")
            showtime_str = parsed_query.get("showtime_str")
            num_tickets = parsed_query.get("num_tickets", 1)
            
            # Add debug logging
            print(f"Parsed query: {parsed_query}")
            
            if not all([movie_name, cinema_name, showtime_str]):
                missing = []
                if not movie_name: missing.append("movie name")
                if not cinema_name: missing.append("cinema location")
                if not showtime_str: missing.append("showtime")
                return StopEvent(f"Missing required information: {', '.join(missing)}")
            
            # Split date and time
            date, time = showtime_str.split(',')
            time = time.strip()
            
            # Make the query more specific
            query = f"""Find the exact showtime where:
            theater_location is exactly '{cinema_name}' AND
            movie_name is exactly '{movie_name}' AND
            date is exactly '{date}' AND
            time is exactly '{time}'
            
            Return only the matching CSV row."""
            
            results = query_engine.query(query)
            
            # Add debug logging
            print(f"Query result: {str(results)}")
            
            # Process the booking
            success, message = book_tickets(
                query_result=str(results),
                num_tickets=num_tickets
            )
            
            if success:
                return StopEvent(success)
            else:
                return StopEvent(f"Booking failed: {message}")
            
        except Exception as e:
            print(f"Error in handle_book_tickets: {str(e)}")  # Add error logging
            return StopEvent(f"An error occurred while processing your booking: {str(e)}")

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
