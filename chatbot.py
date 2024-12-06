import os
import openai
from dotenv import load_dotenv
from omdb_integration import fetch_movie_details
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,)
from llama_index_builder import load_index_from_disk
import logging
import json
from typing import Union

# Custom Event types
class MovieReviewEvent(Event):
    pass

class ShowtimesEvent(Event):
    pass

class CinemaLocationEvent(Event):
    pass

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI LLM
Settings.llm = OpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
llm = Settings.llm

# Load the llama index and create a query engine
index = load_index_from_disk("movie_index.pkl")
query_engine = index.as_query_engine()

class ChatbotWorkflow(Workflow):
    @step
    async def start(self, event: StartEvent) -> Union[StopEvent, MovieReviewEvent, ShowtimesEvent, CinemaLocationEvent]:
        user_query = event.input
        
        # Use GPT to extract structured information
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
                        """
                    },
                    {"role": "user", "content": user_query}
                ],
                response_format={"type": "json_object"}
            )
            
            parsed_query = json.loads(response.choices[0].message.content)
            logging.info(f"Parsed query: {parsed_query}")
            
            # Route to appropriate handler based on intent
            intent = parsed_query.get("intent")
            if intent == "movie_review":
                return MovieReviewEvent(input=json.dumps(parsed_query))
            elif intent == "showtimes":
                return ShowtimesEvent(input=json.dumps(parsed_query))
            elif intent == "cinema_location":
                return CinemaLocationEvent(input=json.dumps(parsed_query))
            else:
                # For general queries, use LlamaIndex
                results = query_engine.query(user_query)
                return StopEvent(str(results))
                
        except Exception as e:
            logging.error(f"Error in query preprocessing: {e}")
            return StopEvent("I'm having trouble understanding your query. Could you rephrase it?")

    @step
    async def handle_movie_review(self, event: MovieReviewEvent) -> StopEvent:
        parsed_query = json.loads(event.input)
        movie_name = parsed_query.get("movie_name")
        
        if not movie_name:
            return StopEvent("Please specify a movie title for reviews or details.")
        
        # Get movie information from llama-index
        local_info = query_engine.query(f"Tell me about the movie {movie_name}")
        
        # Get additional details from OMDB
        movie_details = fetch_movie_details(movie_name)
        
        # Start with local information
        response_parts = [str(local_info)]
        
        # Add OMDB information if available
        if "Error" not in movie_details and movie_details.get("Response", "False") == "True":
            response_parts.extend([
                f"\nAdditional Details from OMDB:",
                f"Release Year: {movie_details.get('Year', 'N/A')}",
                f"IMDB Rating: {movie_details.get('imdbRating', 'N/A')}/10",
                f"Plot: {movie_details.get('Plot', 'N/A')}"
            ])
        
        return StopEvent("\n".join(response_parts))

    @step
    async def handle_showtimes(self, event: ShowtimesEvent) -> StopEvent:
        parsed_query = json.loads(event.input)
        movie_name = parsed_query.get("movie_name")
        locality = parsed_query.get("locality")
        
        # Build a natural language query that will match our index content
        if movie_name:
            query = f"Show me all showtimes and locations for the movie {movie_name}"
            if locality:
                query += f" in {locality}"
        else:
            query = "Show me all current movie showtimes"
            if locality:
                query += f" in {locality}"
            
        results = query_engine.query(query)
        response = str(results)
        
        # If no results found, try a broader search
        if "no showtimes" in response.lower() or "couldn't find" in response.lower():
            if movie_name:
                broader_query = f"Tell me about the movie {movie_name} and where it's playing"
                results = query_engine.query(broader_query)
                response = str(results)
        
        return StopEvent(response)

    @step
    async def handle_cinema_location(self, event: CinemaLocationEvent) -> StopEvent:
        parsed_query = json.loads(event.input)
        
        # Build a natural language query that will match our index content
        if parsed_query.get("locality"):
            query = f"Tell me about cinemas and theatres in {parsed_query['locality']}"
            if parsed_query.get("city"):
                query += f", {parsed_query['city']}"
        elif parsed_query.get("city"):
            query = f"Tell me about cinemas and theatres in {parsed_query['city']}"
        else:
            query = "Tell me about all cinemas and theatres"
            
        results = query_engine.query(query)
        response = str(results)
        
        # If no results found, try a broader search
        if "no cinemas" in response.lower() or "couldn't find" in response.lower():
            broader_query = "List all cinema locations and what's playing there"
            results = query_engine.query(broader_query)
            response = str(results)
        
        return StopEvent(response)

async def chat_with_user(question: str) -> str:
    """Process user queries using the workflow."""
    workflow = ChatbotWorkflow()
    try:
        result = await workflow.run(input=question)
        if isinstance(result, StopEvent):
            return result.output
        return str(result)
    except Exception as e:
        logging.error(f"Error in workflow: {e}")
        return "I encountered an error processing your request. Please try again."

async def chatbot():
    """Interactive console chatbot for testing."""
    print("Movie Agent Chatbot (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ['quit', 'exit']:
            break
            
        response = await chat_with_user(question)
        print(f"\nBot: {response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(chatbot())
