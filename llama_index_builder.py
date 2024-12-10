from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_local_index():
    """Create index from local files in the Data directory."""
    # Initialize OpenAI LLM
    Settings.llm = OpenAI(model="gpt-4-turbo", api_key=OPENAI_API_KEY)
    
    # Get the path to the Data directory
    data_dir = Path("Data")
    
    if not data_dir.exists():
        raise FileNotFoundError("Data directory not found")
    
    # Load documents from the Data directory
    documents = SimpleDirectoryReader(
        input_dir=str(data_dir)
    ).load_data()
    
    # Build the index
    local_index = VectorStoreIndex.from_documents(documents)
    
    # Save the index directly to movie_index directory
    local_index.storage_context.persist(persist_dir="movie_index")
    print("Index saved successfully to ./movie_index")
    return local_index

def load_index_from_disk(storage_dir="movie_index"):
    """Load the index from disk."""
    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context=storage_context)
        return index
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

if __name__ == "__main__":
    create_local_index()