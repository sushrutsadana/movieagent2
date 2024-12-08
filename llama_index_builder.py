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
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader
from llama_parse import LlamaParse
from typing import List

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")

# URLs to scrape
URLS = [
    "https://timesofindia.indiatimes.com/entertainment/bengaluru/movie-showtimes/5",
    "https://timesofindia.indiatimes.com/entertainment/mumbai/movie-showtimes/2",
    "https://timesofindia.indiatimes.com/entertainment/d/movie-showtimes/1",
    "https://timesofindia.indiatimes.com/entertainment/d/movie-showtimes/3"
    # Add more URLs as needed
]

def create_web_index(urls: List[str]):
    """Create index from web pages."""
    Settings.llm = OpenAI(model="gpt-4-turbo", api_key=OPENAI_API_KEY)
    
    # Initialize LlamaParse
    parser = LlamaParse(api_key=LLAMA_PARSE_API_KEY, result_type="markdown")
    
    try:
        # Load data from web pages
        documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
        
        # Create index from web documents
        web_index = VectorStoreIndex.from_documents(documents)
        
        # Save the web index
        web_index.storage_context.persist(persist_dir="./web_storage")
        print("Web index saved successfully.")
        return web_index
    except Exception as e:
        print(f"Error creating web index: {e}")
        return None

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
    
    # Save the index
    local_index.storage_context.persist(persist_dir="./local_storage")
    print("Local index saved successfully.")
    return local_index

def merge_indices(local_index, web_index):
    """Merge local and web indices."""
    if local_index and web_index:
        # Get documents as strings
        all_docs = [doc.text for doc in local_index.docstore.docs.values()] + \
                  [doc.text for doc in web_index.docstore.docs.values()]
        
        # Convert strings to Document objects
        document_objects = [Document(text=doc) for doc in all_docs]
        
        # Create a new combined index
        combined_index = VectorStoreIndex.from_documents(document_objects)
        
        # Save the combined index
        combined_index.storage_context.persist(persist_dir="./storage")
        print("Combined index saved successfully.")
        return combined_index
    return None

def load_index_from_disk(storage_dir="./storage"):
    """Load the index from disk."""
    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context=storage_context)
        return index
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

def create_index():
    """Create and merge indices from both local files and web sources."""
    local_index = create_local_index()
    web_index = create_web_index(URLS)
    combined_index = merge_indices(local_index, web_index)
    
    if combined_index:
        # Save to a consistent location that chatbot.py will use
        combined_index.storage_context.persist(persist_dir="./movie_index")
        print("Index saved to ./movie_index")
    return combined_index

if __name__ == "__main__":
    create_index()