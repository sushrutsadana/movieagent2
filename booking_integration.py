import csv
import re
from datetime import datetime, time, timedelta
import logging

def get_time_period(time_str: str) -> str:
    """Categorize time into morning, afternoon, evening, or night"""
    hour = int(time_str.split(':')[0])
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"

def filter_showtimes(showtimes: list, date_filter: str = None, time_filter: str = None) -> list:
    """Filter showtimes based on date and time preferences"""
    filtered_shows = showtimes
    
    # Date filtering
    if date_filter:
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        if "today" in date_filter.lower():
            filtered_shows = [show for show in filtered_shows if show["date"] == today]
        elif "tomorrow" in date_filter.lower():
            filtered_shows = [show for show in filtered_shows if show["date"] == tomorrow]
    
    # Time filtering
    if time_filter:
        time_filter = time_filter.lower()
        if any(period in time_filter for period in ["morning", "afternoon", "evening", "night"]):
            filtered_shows = [
                show for show in filtered_shows 
                if get_time_period(show["time"]) in time_filter
            ]
    
    return filtered_shows

def extract_showtime_details(query_result: str) -> dict:
    """Extract showtime details from LlamaIndex query result"""
    try:
        # Clean up the result string and split by commas
        result = query_result.strip()
        parts = [part.strip() for part in result.split(',')]
        
        if len(parts) >= 10:
            return {
                'theater_location': parts[0],
                'movie_name': parts[4],
                'date': parts[7],
                'time': parts[8]
            }
        
        print(f"Could not parse result: {result}")
        return None
        
    except Exception as e:
        print(f"Error extracting showtime details: {e}")
        return None

def book_tickets(query_result: str, num_tickets: int, movie_name: str = None, theater_name: str = None) -> tuple[bool, str]:
    try:
        # Read directly from CSV
        with open("Data/Showtimessampledata.csv", "r") as file:
            reader = csv.DictReader(file)
            showtimes = list(reader)
        
        # Parse the query parameters from the result string
        # Example: "2024-12-15 22:15 - hindi - 150 seats"
        parts = query_result.strip().split(' - ')
        if len(parts) >= 2:
            date_time = parts[0].split()
            if len(date_time) == 2:
                date, time = date_time
                language = parts[1]
                
                # Find matching showtime in CSV
                matching_show = None
                for show in showtimes:
                    logging.info(f"Checking show: {show}")
                    if (show['date'] == date and 
                        show['time'] == time and 
                        show['language'].lower() == language.lower() and 
                        show['movie_name'] == movie_name and  
                        show['theater_location'] == theater_name):  # Exact match
                        matching_show = show
                        logging.info(f"Found matching show: {show}")
                        break
                
                if matching_show:
                    available = int(matching_show["available_seats"])
                    if available >= num_tickets:
                        # Update seats
                        matching_show["available_seats"] = str(available - num_tickets)
                        
                        # Write back to CSV
                        fieldnames = reader.fieldnames
                        with open("Data/Showtimessampledata.csv", "w", newline="") as wfile:
                            writer = csv.DictWriter(wfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(showtimes)
                        
                        return True, f"Successfully booked {num_tickets} ticket(s)"
                    else:
                        return False, f"Not enough seats available. Only {available} seats left."
                
                return False, "Could not find matching showtime in the database."
        
        return False, "Invalid booking data format"
        
    except Exception as e:
        logging.error(f"Booking error: {str(e)}")
        return False, f"Booking error: {str(e)}"