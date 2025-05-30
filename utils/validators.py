import re
import phonenumbers
from email_validator import validate_email, EmailNotValidError
from datetime import datetime, timedelta
from dateutil import parser
import calendar


class InputValidator:
    @staticmethod
    def validate_email(email: str) -> tuple[bool, str]:
        """Validate email format"""
        try:
            validated_email = validate_email(email)
            return True, validated_email.email
        except EmailNotValidError as e:
            return False, str(e)
    
    @staticmethod
    def validate_phone(phone: str, region: str = "US") -> tuple[bool, str]:
        """Validate phone number format"""
        try:
            parsed_number = phonenumbers.parse(phone, region)
            if phonenumbers.is_valid_number(parsed_number):
                formatted = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
                return True, formatted
            else:
                return False, "Invalid phone number format"
        except phonenumbers.NumberParseException as e:
            return False, f"Phone validation error: {e}"
    
    @staticmethod
    def validate_name(name: str) -> tuple[bool, str]:
        """Validate name format"""
        name = name.strip()
        if len(name) < 2:
            return False, "Name must be at least 2 characters long"
        if not re.match(r"^[a-zA-Z\s\-\.\']+$", name):
            return False, "Name can only contain letters, spaces, hyphens, dots, and apostrophes"
        return True, name.title()


class DateParser:
    @staticmethod
    def parse_date_from_text(text: str) -> tuple[bool, str, str]:
        """
        Parse date from natural language text
        Returns: (success, formatted_date_YYYY-MM-DD, explanation)
        """
        text = text.lower().strip()
        today = datetime.now().date()
        
        # Handle relative dates
        if "today" in text:
            return True, today.strftime("%Y-%m-%d"), "Today"
        elif "tomorrow" in text:
            date = today + timedelta(days=1)
            return True, date.strftime("%Y-%m-%d"), "Tomorrow"
        elif "yesterday" in text:
            date = today - timedelta(days=1)
            return True, date.strftime("%Y-%m-%d"), "Yesterday"
        
        # Handle "next [day]" or "this [day]"
        days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, day in enumerate(days_of_week):
            if day in text:
                current_weekday = today.weekday()  # Monday is 0
                target_weekday = i
                
                if "next" in text:
                    days_ahead = target_weekday - current_weekday + 7
                else:  # "this" or just the day name
                    days_ahead = target_weekday - current_weekday
                    if days_ahead <= 0:
                        days_ahead += 7
                
                target_date = today + timedelta(days=days_ahead)
                return True, target_date.strftime("%Y-%m-%d"), f"Next {day.title()}"
        
        # Handle "in X days"
        days_match = re.search(r'in (\d+) days?', text)
        if days_match:
            days = int(days_match.group(1))
            target_date = today + timedelta(days=days)
            return True, target_date.strftime("%Y-%m-%d"), f"In {days} days"
        
        # Try to parse absolute dates
        try:
            parsed_date = parser.parse(text, fuzzy=True)
            if parsed_date.date() < today:
                # If parsed date is in the past, assume next year
                parsed_date = parsed_date.replace(year=today.year + 1)
            return True, parsed_date.strftime("%Y-%m-%d"), f"Parsed: {parsed_date.strftime('%B %d, %Y')}"
        except:
            pass
        
        return False, "", "Could not parse date. Please try formats like 'next Monday', 'tomorrow', '2024-03-15', or 'March 15'"
    


class TimeParser:
    @staticmethod
    def parse_time_from_text(text: str) -> tuple[bool, str, str]:
        """
        Parse time from natural language text
        Returns: (success, formatted_time_HH:MM, explanation)
        """
        text = text.lower().strip()
        
        # Remove common words
        text = text.replace("at", "").replace("around", "").replace("about", "").strip()
        
        # Handle common time formats
        import re
        
        # Pattern for 12-hour format (9am, 2:30pm, 10:15 AM)
        twelve_hour_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)'
        match = re.search(twelve_hour_pattern, text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            period = match.group(3).lower().replace('.', '')
            
            # Convert to 24-hour format
            if period in ['pm', 'pm'] and hour != 12:
                hour += 12
            elif period in ['am', 'am'] and hour == 12:
                hour = 0
                
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                time_str = f"{hour:02d}:{minute:02d}"
                period_display = "AM" if hour < 12 else "PM"
                display_hour = hour if hour <= 12 else hour - 12
                if display_hour == 0:
                    display_hour = 12
                return True, time_str, f"{display_hour}:{minute:02d} {period_display}"
        
        # Pattern for 24-hour format (14:30, 09:00, 23:45)
        twenty_four_hour_pattern = r'(\d{1,2}):(\d{2})'
        match = re.search(twenty_four_hour_pattern, text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                time_str = f"{hour:02d}:{minute:02d}"
                period = "AM" if hour < 12 else "PM"
                display_hour = hour if hour <= 12 else hour - 12
                if display_hour == 0:
                    display_hour = 12
                return True, time_str, f"{display_hour}:{minute:02d} {period}"
        
        # Pattern for simple hour (9, 14, "nine")
        simple_hour_pattern = r'\b(\d{1,2})\b'
        match = re.search(simple_hour_pattern, text)
        if match:
            hour = int(match.group(1))
            if 1 <= hour <= 12:
                # Assume PM for business hours (9-5), AM for early hours
                if 9 <= hour <= 12:
                    actual_hour = hour + 12 if hour != 12 else 12
                    return True, f"{actual_hour:02d}:00", f"{hour}:00 PM"
                else:
                    return True, f"{hour:02d}:00", f"{hour}:00 AM"
            elif 13 <= hour <= 23:
                display_hour = hour - 12
                return True, f"{hour:02d}:00", f"{display_hour}:00 PM"
        
        # Handle text-based times
        text_times = {
            "morning": ("09:00", "9:00 AM"),
            "afternoon": ("14:00", "2:00 PM"), 
            "evening": ("18:00", "6:00 PM"),
            "night": ("20:00", "8:00 PM"),
            "noon": ("12:00", "12:00 PM"),
            "midnight": ("00:00", "12:00 AM"),
            "lunch": ("12:30", "12:30 PM"),
            "dinner": ("19:00", "7:00 PM")
        }
        
        for key, (time_24, time_display) in text_times.items():
            if key in text:
                return True, time_24, time_display
        
        return False, "", "Could not parse time. Please use formats like '2:30 PM', '14:30', '9am', or 'morning'"
