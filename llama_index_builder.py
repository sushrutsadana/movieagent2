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
    """Create index from showtimes sample data CSV."""
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    
    # Get the path to the CSV file
    data_file = Path("Data/showtimessampledata.csv")
    
    if not data_file.exists():
        raise FileNotFoundError("Showtime sample data CSV file not found")
    
    # Load only the CSV file
    documents = SimpleDirectoryReader(
        input_files=[str(data_file)]
    ).load_data()
    
    # Build the index
    local_index = VectorStoreIndex.from_documents(documents)
    
    # Save the index
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