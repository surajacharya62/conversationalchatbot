from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
import json
from datetime import datetime
from utils.validators import InputValidator, DateParser,TimeParser
from utils.document_processor import DocumentProcessor


##------------------ Creating the conversational form to call and book the appointment
class ConversationalForm:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.data = {
            "name": None,
            "email": None,
            "phone": None,
            "appointment_date": None,
            "appointment_time":None,
            "purpose": None
        }
        self.current_step = None
        self.validation_errors = {}
    
    def is_complete(self) -> bool:
        required_fields = ["name", "email", "phone", "appointment_date","appointment_time"]
        return all(self.data[field] is not None for field in required_fields)
    
    def get_next_missing_field(self) -> Optional[str]:
        required_fields = ["name", "email", "phone", "appointment_date","appointment_time", "purpose"]
        for field in required_fields:
            if self.data[field] is None:
                return field
        return None
    
    def validate_and_set_field(self, field: str, value: str) -> tuple[bool, str]:
        """Validate and set field value"""
        if field == "name":
            is_valid, result = InputValidator.validate_name(value)
            if is_valid:
                self.data["name"] = result
                return True, f"✅ Name set to: {result}"
            else:
                return False, f"❌ {result}"
        
        elif field == "email":
            is_valid, result = InputValidator.validate_email(value)
            if is_valid:
                self.data["email"] = result
                return True, f"✅ Email set to: {result}"
            else:
                return False, f"❌ {result}"
        
        elif field == "phone":
            is_valid, result = InputValidator.validate_phone(value)
            if is_valid:
                self.data["phone"] = result
                return True, f"✅ Phone set to: {result}"
            else:
                return False, f"❌ {result}"
        
        elif field == "appointment_date":
            is_valid, date_str, explanation = DateParser.parse_date_from_text(value)
            if is_valid:
                self.data["appointment_date"] = date_str
                return True, f"✅ Appointment date set to: {date_str} ({explanation})"
            else:
                return False, f"❌ {explanation}"
        
        elif field == "appointment_time":
            is_valid, time_str, explanation = TimeParser.parse_time_from_text(value)
            if is_valid:
                self.data["appointment_time"] = time_str
                return True, f"✅ Appointment time set to: {time_str} ({explanation})"
            else:
                return False, f"❌ {explanation}"
        
        
        elif field == "purpose":
            if len(value.strip()) > 0:
                self.data["purpose"] = value.strip()
                return True, f"✅ Purpose noted: {value.strip()}"
            else:
                # Purpose is optional
                return True, "✅ No specific purpose noted (optional field skipped)"
        
        return False, "❌ Unknown field"


class SimpleChatbot:
    def __init__(self, google_api_key: str, documents_path: List[str] = None):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.1
        )
        
        self.document_processor = DocumentProcessor(google_api_key)
        self.conversational_form = ConversationalForm()

        self.vectorstore_cleared = False

        # Initialize document store if paths provided
        if documents_path is None:
            print("empty")
            
        elif documents_path:
            print("document_path")
            self.setup_documents(documents_path)
        else:
            try:
                print("load_existing")
                if not self.vectorstore_cleared:
                    self.document_processor.load_existing_vectorstore()           

                    print("Loaded existing document store")
            except FileNotFoundError:
                print("No existing document store found")
    

    def setup_documents(self, file_paths: List[str]):
        """Setup document processing"""
        try:
            documents = self.document_processor.load_documents(file_paths)
            self.document_processor.create_vectorstore(documents)
            print(f"Document store created with {len(file_paths)} files")
        except Exception as e:
            print(f"Error setting up documents: {e}")
    

    def clear_vectorstoredb(self):
        print("clear")
        self.document_processor.vectorstore = None
        self.vectorstore_cleared = True 
        self.document_processor.clear_vectorstore()
        print(self.document_processor.vectorstore)


    def chat(self, user_input: str) -> str:
        """
        user_input: take user chat input from text_input box
        return: processed string  
        """
        
        user_lower = user_input.lower().strip()
        
        # Handle booking flow first (highest priority when active)
        if self.conversational_form.current_step == "collecting":
            return self._handle_booking_flow(user_input)
        
        # Handling call or booking requests
        booking_keywords = ['book', 'appointment', 'call me', 'contact me', 'schedule', 'meeting']
        if any(keyword in user_lower for keyword in booking_keywords):
            self.conversational_form.current_step = "collecting"
            return "I'd be happy to help you book an appointment! 📅\n\nLet's start with your full name:"
        
        # Handling chat reset requests
        if any(word in user_lower for word in ["reset", "start over", "clear", "restart"]):
            self.conversational_form.reset()
            return ("🔄 I've reset everything. How can I help you today?\n\n"
                   "• Ask questions about uploaded documents\n"
                   "• Say 'I want to book an appointment' to schedule a meeting")
        
       
        if self.document_processor.vectorstore:  

            doc_keywords = ["what", "how", "tell me", "explain", "about", "service", 
                           "company", "information", "describe", "details", "help"]
            
            if any(keyword in user_lower for keyword in doc_keywords):
                return self._search_documents(user_input)
            
            return self._general_response(user_input)
        
        else:
           return "Please upload the documents. Thank you."
    

    def _handle_booking_flow(self, user_input: str) -> str:
        """Handling the booking conversation flow"""
        next_field = self.conversational_form.get_next_missing_field()
               
        if next_field is None:
            
            summary = self._format_booking_summary()
            self.conversational_form.current_step = "complete"
            return f"🎉 Perfect! Here's your booking information:\n\n{summary}\n\n📞 We'll contact you soon to confirm your appointment!"
        
        if next_field == "purpose" and any(word in user_input.lower() for word in ["skip", "no", "none", "nothing"]):
            self.conversational_form.data["purpose"] = "Not specified"
            summary = self._format_booking_summary()
            self.conversational_form.current_step = "complete"
            return f"✅ No problem! Here's your booking information:\n\n{summary}\n\n📞 We'll contact you soon to confirm your appointment!"
        
        # Validate and set the current field
        success, message = self.conversational_form.validate_and_set_field(next_field, user_input)
        
        if success:
            # Successfully set field, check what's next
            next_missing = self.conversational_form.get_next_missing_field()
            
            if next_missing:
                prompt = self._get_field_prompt(next_missing)
                return f"{message}\n\n--{prompt}"
            else:
                # when all fields complete
                summary = self._format_booking_summary()
                self.conversational_form.current_step = "complete"
                return f"{message}\n\n🎉 Perfect! Here's your booking information:\n\n{summary}\n\n📞 We'll contact you soon to confirm your appointment!"
        else:
            # if validation check is failed, asking again for the same field
            prompt = self._get_field_prompt(next_field)
            return f"{message}\n\n--{prompt}"
    

    def _search_documents(self, query: str) -> str:
        """Search through documents"""
        try:
            results = self.document_processor.similarity_search(query, k=3)
            if results:
                context = "\n".join([doc.page_content[:1000] + "..." for doc in results])
                return f"Based on the uploaded documents:\n\n{context}\n\n❓ Do you have any other questions about the documents?"
            else:
                return "I don't have specific information about that in the uploaded documents. Could you try a different question or upload relevant documents?"
        except Exception as e:
            return "I encountered an issue searching the documents. Please try rephrasing your question or check if documents are properly uploaded."
    

    def _general_response(self, user_input: str) -> str:
        """Generate general response"""
        return f"""👋 Hello! I'm here to help you. I can:

                **Answer questions** about uploaded documents
                **Book appointments** (just say 'I want to book an appointment')
                **Have conversations** about various topics

                You asked: "{user_input}"

                What would you like to know more about?"""
    

    def _get_field_prompt(self, field: str) -> str:
        """Get appropriate prompt for each field"""
        prompts = {
            "name": "What's your full name? 👤",
            "email": "What's your email address? 📧",
            "phone": "What's your phone number? 📱Start with +977",
            "appointment_date": "When would you like to schedule your appointment? 📅\n(e.g., 'next Monday', 'tomorrow', '2024-03-15')",
            "appointment_time": "When would you like to have your appointment_time? 📅\n(e.g., 'morning time', 'afternoon time', '9am')",
            "purpose": "What's the purpose of your appointment? 🎯\n(optional - you can say 'skip' if you prefer not to specify)"
        }
        return prompts.get(field, "Please provide the requested information:")
    

    def _format_booking_summary(self) -> str:
        """Format booking summary"""
        data = self.conversational_form.data
        summary = f"""📋 **Appointment Request Summary**        
                    👤 **Name:** {data['name']}
                    📧 **Email:** {data['email']}
                    📞 **Phone:** {data['phone']}
                    📅 **Requested Date:** {data['appointment_date']}
                    📅 **Requested Time:** {data['appointment_time']}
                    """
        
        if data['purpose'] and data['purpose'] != "Not specified":
            summary += f"\n🎯 **Purpose:** {data['purpose']}"
        
        return summary
    
    
    def get_booking_status(self) -> Dict[str, Any]:
        """Get current booking form status"""
        return {
            "is_complete": self.conversational_form.is_complete(),
            "current_step": self.conversational_form.current_step,
            "data": self.conversational_form.data,
            "next_field": self.conversational_form.get_next_missing_field()
        }