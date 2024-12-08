import os
from supabase import create_client
from datetime import datetime
from dotenv import load_dotenv
import logging

load_dotenv()

def initialize_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

def get_showtime_record(movie_name: str, cinema_name: str, showtime_str: str):
    supabase = initialize_supabase()
    try:
        # Log the exact values we're searching for
        logging.info(f"Searching for exact values:")
        logging.info(f"Movie name: '{movie_name}'")
        logging.info(f"Cinema name: '{cinema_name}'")
        logging.info(f"Showtime: '{showtime_str}'")
        
        # Let's first get ALL records to see what's in the database
        all_records = supabase.table('showtimes').select('*').execute()
        logging.info(f"All records in database: {all_records.data}")
        
        # Then try just the movie search
        movie_records = supabase.table('showtimes').select('*').ilike(
            'movie_name', f'%{movie_name}%'
        ).execute()
        logging.info(f"Records with similar movie name: {movie_records.data}")
        
        # Original exact search
        response = supabase.table('showtimes').select('*').eq(
            'movie_name', movie_name
        ).eq(
            'cinema_name', cinema_name
        ).eq(
            'showtime', showtime_str
        ).execute()
        
        records = response.data
        if not records:
            logging.info("No records found matching all criteria")
        return records[0] if records else None
        
    except Exception as e:
        logging.error(f"Error fetching showtime record: {e}")
        return None

def decrement_seats(movie_name: str, cinema_name: str, showtime_str: str, num_tickets: int):
    supabase = initialize_supabase()
    showtime = get_showtime_record(movie_name, cinema_name, showtime_str)
    if not showtime:
        return False, "Showtime not found."

    available = showtime["available_seats"]
    if available < num_tickets:
        return False, "Not enough seats available."
    
    updated_res = supabase.table("showtimes")\
        .update({"available_seats": available - num_tickets})\
        .eq("id", showtime["id"])\
        .execute()

    if updated_res.error:
        return False, "Could not update seat availability."
    return True, None

def create_booking(movie_name: str, cinema_name: str, showtime_str: str, num_tickets: int):
    supabase = initialize_supabase()
    res = supabase.table("bookings").insert({
        "movie_name": movie_name,
        "cinema_name": cinema_name,
        "showtime": showtime_str,
        "num_tickets": num_tickets,
        "created_at": datetime.now().isoformat()
    }).execute()
    if res.data and len(res.data) > 0:
        return res.data[0]["id"]
    return None

def book_tickets(movie_name: str, cinema_name: str, showtime_str: str, num_tickets: int = 2):
    showtime = get_showtime_record(movie_name, cinema_name, showtime_str)
    if not showtime:
        return None, "Sorry, that showtime is not available."

    success, error_msg = decrement_seats(movie_name, cinema_name, showtime_str, num_tickets)
    if not success:
        return None, error_msg

    booking_id = create_booking(movie_name, cinema_name, showtime_str, num_tickets)
    if booking_id:
        return booking_id, None
    else:
        return None, "Failed to create booking record."
