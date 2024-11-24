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

def fetch_showtimes_for_movie(film_id=None, date=None, time_filter=None):
    supabase = initialize_supabase()
    query = supabase.table("showtimes").select("""
        id,
        film_id,
        cinema_id,
        start_time,
        end_time
    """)
    if film_id:
        query = query.eq("film_id", film_id)
    if date:
        query = query.eq("date", date.strftime("%Y-%m-%d"))  # Assumes date is a datetime object
    if time_filter:
        query = query.gte("start_time", time_filter.strftime("%H:%M:%S"))  # Assuming time_filter is a datetime object specifying the time
    return query.execute().data
