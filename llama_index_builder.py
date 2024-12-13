from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.schema import MetadataMode
from llama_index.llms.openai import OpenAI
import pandas as pd
from typing import List

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_structured_documents(csv_path: str) -> List[Document]:
    """Create structured documents from CSV with field validation."""
    # Read CSV with pandas for better data validation
    df = pd.read_csv(csv_path)
    
    # Define expected columns and their types
    expected_columns = {
        'theater_location': str,
        'address': str,
        'city': str,
        'state': str,
        'movie_name': str,
        'language': str,
        'genre': str,
        'date': str,
        'time': str,
        'available_seats': int
    }
    
    # Validate columns
    for col, dtype in expected_columns.items():
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].astype(dtype)
    
    # Validate date and time formats
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['time'] = pd.to_datetime(df['time'], format='%H:%M').dt.strftime('%H:%M')
    
    documents = []
    
    # Create a document for each unique movie-theater combination
    for theater in df['theater_location'].unique():
        theater_data = df[df['theater_location'] == theater]
        
        for movie in theater_data['movie_name'].unique():
            movie_shows = theater_data[theater_data['movie_name'] == movie]
            
            # Create structured text representation
            text = f"""
            Theater: {theater}
            Address: {movie_shows['address'].iloc[0]}
            Movie: {movie}
            Languages: {', '.join(movie_shows['language'].unique())}
            Genres: {', '.join(movie_shows['genre'].unique())}
            
            Available Showtimes:
            """
            
            # Add each showtime with validation
            for _, show in movie_shows.iterrows():
                text += f"\n{show['date']} {show['time']} - {show['language']} - {show['available_seats']} seats"
            
            # Add metadata for better querying
            metadata = {
                'theater_location': theater,
                'address': movie_shows['address'].iloc[0],
                'city': movie_shows['city'].iloc[0],
                'movie_name': movie,
                'languages': list(movie_shows['language'].unique()),
                'genres': list(movie_shows['genre'].unique()),
                'dates': list(movie_shows['date'].unique()),
                'document_type': 'movie_showtime'
            }
            
            doc = Document(
                text=text,
                metadata=metadata,
                metadata_mode=MetadataMode.ALL
            )
            documents.append(doc)
    
    return documents

def create_local_index():
    """Create index from structured showtime documents."""
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    
    # Get the path to the CSV file
    data_file = Path("Data/Showtimessampledata.csv")
    
    if not data_file.exists():
        raise FileNotFoundError("Showtime sample data CSV file not found")
    
    # Create structured documents
    documents = create_structured_documents(str(data_file))
    
    # Build the index with metadata
    local_index = VectorStoreIndex.from_documents(
        documents,
        metadata_mode=MetadataMode.ALL,
    )
    
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