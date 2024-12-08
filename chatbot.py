import os
import openai
from dotenv import load_dotenv
from omdb_integration import fetch_movie_details
from booking_integration import book_tickets, get_showtime_record
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
from llama_index_builder import load_index_from_disk
import logging
import json
from typing import Union

# Define event types
class MovieReviewEvent(Event):
    pass

class ShowtimesEvent(Event):
    pass

class CinemaLocationEvent(Event):
    pass

class BookTicketsEvent(Event):
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI LLM
Settings.llm = OpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
llm = Settings.llm

# Load the LlamaIndex
index = load_index_from_disk("./movie_index")
if index is None:
    raise RuntimeError(
        "Failed to load index. Please ensure you've generated the index first by running: "
        "python llama_index_builder.py"
    )

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

class ChatbotWorkflow(Workflow):
    @step
    async def start(self, event: StartEvent) -> Union[StopEvent, MovieReviewEvent, ShowtimesEvent, CinemaLocationEvent, BookTicketsEvent]:
        user_query = event.input
        try:
            # Parse user query using OpenAI
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                         "role": "system",
                        "content": """
                        Extract structured information from movie-related queries.
                        Possible intents:
                          - movie_info
                          - showtimes
                          - cinema_location
                          - movie_review
                          - genre_search
                          - book_tickets

                        Return a JSON object with these possible fields (only if mentioned):
                        - intent
                        - movie_name
                        - city
                        - locality
                        - cinema_name
                        - showtime_str
                        - genre
                        - time_context
                        - num_tickets
                        Examples:
                    "movies playing in koramangala" ->
                    {"intent": "showtimes_str", "locality": "koramangala", "time_context": "currently_playing"}
                    
                    "theatres in JP Nagar bangalore" ->
                    {"intent": "cinema_location", "city": "bangalore", "locality": "JP Nagar"}
                    
                    "showtimes for singham again in indiranagar" ->
                    {"intent": "showtimes_str", "movie_name": "singham again", "locality": "indiranagar"}
                    
                    "what movies are showing in church street" ->
                    {"intent": "showtimes_str", "locality": "church street", "time_context": "currently_playing"}
                    """
                        
                    },
                    {"role": "user", "content": user_query}
                ]
            )
            
            parsed_query = json.loads(response.choices[0].message.content)
            logging.info(f"Parsed query: {parsed_query}")
            
            intent = parsed_query.get("intent")
            if intent == "movie_review":
                return MovieReviewEvent(input=json.dumps(parsed_query))
            elif intent == "showtimes":
                return ShowtimesEvent(input=json.dumps(parsed_query))
            elif intent == "cinema_location":
                return CinemaLocationEvent(input=json.dumps(parsed_query))
            elif intent == "book_tickets":
                return BookTicketsEvent(input=json.dumps(parsed_query))
            else:
                # Default fallback: Query LlamaIndex
                results = query_engine.query(user_query)
                return StopEvent(str(results))
                
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return StopEvent("I encountered an error understanding your query. Could you rephrase it?")

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
        query = f"Show me showtimes for {movie_name or 'movies'}"
        if locality:
            query += f" in {locality}"
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
