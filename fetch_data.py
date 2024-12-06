import os
from supabase import create_client
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def initialize_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

def fetch_movies():
    supabase = initialize_supabase()
    return supabase.table("movies").select("""
        id,
        film_name,
        imdb_id,
        imdb_title_id,
        version_type,
        age_rating
    """).execute().data

def fetch_cinemas():
    supabase = initialize_supabase()
    return supabase.table("cinemas").select("""
        id,
        cinema_name,
        address,
        city,
        state,
        postcode,
        lat,
        lng,
        distance
    """).execute().data

def fetch_movie_by_name(movie_name):
    supabase = initialize_supabase()
    result = supabase.table("movies").select("*").ilike("film_name", f"%{movie_name}%").execute()
    if result.data:
        return result.data[0]
    return None

def fetch_showtimes_for_movie(movie_name=None, date=None, time_filter=None):
    supabase = initialize_supabase()
    
    # First find the movie ID
    if movie_name:
        movie = fetch_movie_by_name(movie_name)
        if not movie:
            return []
        film_id = movie["id"]
    else:
        return []

    # Then query showtimes
    query = supabase.table("Showtimes").select("""
        id,
        film_id,
        cinema_id,
        start_time,
        end_time
    """)
    
    query = query.eq("film_id", film_id)
    
    if date:
        query = query.eq("date", date.strftime("%Y-%m-%d"))
    
    return query.execute().data
