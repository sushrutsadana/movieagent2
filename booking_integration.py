import csv
import re
from datetime import datetime, time, timedelta

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

def book_tickets(query_result: str, num_tickets: int, date_filter: str = None, time_filter: str = None) -> tuple[str, str]:
    try:
        showtime_details = extract_showtime_details(query_result)
        if not showtime_details:
            return None, "Could not find matching showtime in the database."

        with open("Data/Showtimessampledata.csv", "r") as file:
            reader = csv.DictReader(file)
            showtimes = list(reader)

        # Apply filters
        filtered_shows = filter_showtimes(showtimes, date_filter, time_filter)
        if not filtered_shows:
            return None, f"No shows found for the specified time/date preferences."

        # Find the exact matching show from filtered results
        matching_show = None
        for show in filtered_shows:
            if (show["movie_name"].lower() == showtime_details["movie_name"].lower() and
                show["theater_location"].lower() == showtime_details["theater_location"].lower() and
                show["date"] == showtime_details["date"] and
                show["time"] == showtime_details["time"]):
                matching_show = show
                break

        if matching_show:
            available = int(matching_show["available_seats"])
            if available >= num_tickets:
                # Book it
                matching_show["available_seats"] = str(available - num_tickets)

                # Write back
                fieldnames = reader.fieldnames
                with open("Data/Showtimessampledata.csv", "w", newline="") as wfile:
                    writer = csv.DictWriter(wfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(showtimes)

                return f"Successfully booked {num_tickets} ticket(s) for {matching_show['movie_name']} at {matching_show['theater_location']} for {matching_show['date']} {matching_show['time']}", None
            else:
                return None, f"Not enough seats available. Only {available} seats left."
        
        return None, "Could not find the exact showtime in the database."

    except Exception as e:
        return None, f"Booking error: {str(e)}"
