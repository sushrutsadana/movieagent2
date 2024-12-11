import csv
import re
from datetime import datetime

def extract_showtime_details(query_result: str) -> dict:
    """Extract showtime details from LlamaIndex query result"""
    try:
        # Clean up the result string and split by commas
        result = query_result.strip()
        parts = [part.strip() for part in result.split(',')]
        
        # If we have at least 10 parts (as per CSV format), extract the details
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

def book_tickets(query_result: str, num_tickets: int) -> tuple[str, str]:
    try:
        # Extract showtime details from LlamaIndex query result
        showtime_details = extract_showtime_details(query_result)
        if not showtime_details:
            return None, "Could not find matching showtime in the database."

        # Read the CSV file
        with open("Data/Showtimessampledata.csv", "r") as file:
            reader = csv.DictReader(file)
            showtimes = list(reader)

        # Find the exact matching show
        matching_show = None
        for show in showtimes:
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
