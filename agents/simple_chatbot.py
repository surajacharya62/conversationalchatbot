from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
import json
import os
from datetime import datetime
from utils.validators import InputValidator, DateParser, TimeParser
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
            "appointment_time": None,
            "purpose": None
        }
        self.current_step = None
        self.validation_errors = {}
    
    def is_complete(self) -> bool:
        required_fields = ["name", "email", "phone", "appointment_date", "appointment_time"]
        return all(self.data[field] is not None for field in required_fields)
    
    def get_next_missing_field(self) -> Optional[str]:
        required_fields = ["name", "email", "phone", "appointment_date", "appointment_time", "purpose"]
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

        # initializing document store if paths provided
        if documents_path:
            self.setup_documents(documents_path, force_refresh=True)
        else:
            try:
                self.document_processor.load_existing_vectorstore()
                print("Loaded existing document store")
            except FileNotFoundError:
                print("No existing document store found")
    
    def setup_documents(self, file_paths: List[str], force_refresh: bool = True):
        """Setup document processing"""
        print(f"🔧 Processing {len(file_paths)} files...")
        
        # clearing first to prevent old content
        if force_refresh:
            print("🗑️ Clearing old documents first...")
            self.clear_documents()
        
        try:
           
            documents = self.document_processor.load_documents(file_paths)
            if not documents:
                raise ValueError("No documents could be loaded")
            
            print(f"📚 Loaded {len(documents)} document(s)")
            
            # showing what we're actually loading (first 100 chars of first document)
            if documents:
                preview = documents[0].page_content[:100]
                print(f"📄 Document preview: {preview}...")
            
            # creating vector store
            self.document_processor.create_vectorstore(documents, force_refresh=True)
            
            # verifying it works with actual content
            if self.document_processor.vectorstore:
                print("✅ Document processing completed successfully")                
                
                test_results = self.document_processor.similarity_search("", k=1)
                if test_results:
                    test_preview = test_results[0].page_content[:100]
                    print(f"🧪 Test search preview: {test_preview}...")
                else:
                    print("⚠️ Test search returned no results")
            else:
                raise ValueError("Vector store creation failed")
                
        except Exception as e:
            print(f"❌ Document setup failed: {e}")
            raise e
    
    def clear_documents(self):
        """Clear all documents from vector store"""
        import shutil
        try:
            # deleting the current vectorstore
            if hasattr(self.document_processor, 'vectorstore') and self.document_processor.vectorstore:
                try:
                    del self.document_processor.vectorstore
                except:
                    pass
            
            self.document_processor.vectorstore = None
            
            # removing the database directory completely
            db_paths = ["./chroma_db", "./.chroma", "./chromadb"]
            for db_path in db_paths:
                if os.path.exists(db_path):
                    try:
                        shutil.rmtree(db_path)
                        print(f"🗑️ Removed {db_path}")
                    except:
                        pass
            
            print("✅ All document data cleared")
        except Exception as e:
            print(f"Error clearing documents: {e}")

    
    # def clear_vectorstoredb(self):
    #     """Legacy method - use clear_documents() instead"""
    #     self.clear_documents()

    def chat(self, user_input: str) -> str:
        """
        user_input: take user chat input from text_input box
        return: processed string  
        """
        
        user_lower = user_input.lower().strip()        
      
        # handling booking flow first (highest priority when active)
        if self.conversational_form.current_step == "collecting":
            return self._handle_booking_flow(user_input)
        
        # handling call or booking requests
        booking_keywords = ['book', 'appointment', 'call me', 'contact me', 'schedule', 'meeting']
        if any(keyword in user_lower for keyword in booking_keywords):
            self.conversational_form.current_step = "collecting"
            return "I'd be happy to help you book an appointment! 📅\n\nLet's start with your full name:"
        
        # handling chat reset requests
        if any(word in user_lower for word in ["reset", "start over", "clear", "restart"]):
            self.conversational_form.reset()
            return ("🔄 I've reset everything. How can I help you today?\n\n"
                   "• Ask questions about uploaded documents\n"
                   "• Say 'I want to book an appointment' to schedule a meeting")
        
        
        if self.document_processor.vectorstore:
            print("📄 Attempting document search...")
            try:
                return self._search_documents(user_input)
            except Exception as e:
                print(f"❌ Document search failed: {e}")
                return f"📄 Document search failed: {str(e)}. The documents may not be properly loaded."
        
        # messaging when no documents are loaded
        print("⚠️ No vectorstore available")
        return ("📄 No documents are currently uploaded. Please upload documents using the sidebar to get started!\n\n"
               "I can also help you:\n"
               "• Book appointments (say 'I want to book an appointment')\n"
               "• Have general conversations")

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
        
        # validating and setting the current field
        success, message = self.conversational_form.validate_and_set_field(next_field, user_input)
        
        if success:
            # successfully setting field and check what's next field
            next_missing = self.conversational_form.get_next_missing_field()
            
            if next_missing:
                prompt = self._get_field_prompt(next_missing)
                return f"{message}\n\n{prompt}"
            else:
                # when all fields complete
                summary = self._format_booking_summary()
                self.conversational_form.current_step = "complete"
                return f"{message}\n\n🎉 Perfect! Here's your booking information:\n\n{summary}\n\n📞 We'll contact you soon to confirm your appointment!"
        else:
            # if validation check is failed, asking again for the same field
            prompt = self._get_field_prompt(next_field)
            return f"{message}\n\n{prompt}"

    def _search_documents(self, query: str) -> str:
        """Search through documents"""
        try:
            if not self.document_processor.vectorstore:
                return "📄 No documents are currently loaded. Please upload documents first."           
                    
           
            results = []
            
            # Strategy 1: Direct similarity search
            try:
                results = self.document_processor.similarity_search(query, k=5)
                print(f"📊 Direct search found {len(results)} results")
            except Exception as e:
                print(f"Direct search failed: {e}")
            
            # Strategy 2: If no results, try broader search
            if not results:
                try:
                    # Search for individual keywords
                    keywords = query.lower().split()
                    for keyword in keywords[:3]:  # Try first 3 keywords
                        if len(keyword) > 3:  # Skip short words
                            results = self.document_processor.similarity_search(keyword, k=3)
                            if results:
                                print(f"📊 Keyword '{keyword}' found {len(results)} results")
                                break
                except Exception as e:
                    print(f"Keyword search failed: {e}")
            
            # Strategy 3: If still no results, get any content
            if not results:
                try:
                    results = self.document_processor.similarity_search("", k=3)
                    print(f"📊 Fallback search found {len(results)} results")
                except Exception as e:
                    print(f"Fallback search failed: {e}")
            
            if results and len(results) > 0:
                response = "📄 **Here's what I found in your documents:**\n\n"
                
                for i, doc in enumerate(results[:3]):                    
                    content = doc.page_content.strip()
                    if len(content) > 300:
                        content = content[:300] + "..."                    
                   
                    source = doc.metadata.get('source', 'Document')
                    filename = os.path.basename(source) if source != 'Document' else f'Section {i+1}'
                    
                    response += f"**📁 {filename}:**\n{content}\n\n"
                
                response += "❓ Would you like me to search for something more specific?"
                return response
            else:
                return ("📄 I processed your documents but couldn't find relevant content for your query. "
                       "This might be because:\n"
                       "• The document content doesn't match your question\n"
                       "• Try asking about specific topics mentioned in your documents\n"
                       "• Or ask 'What is in the document?' to see the general content")
                
        except Exception as e:
            print(f"❌ Error in document search: {e}")
            return f"📄 Search error: {str(e)}. Please try re-uploading your documents."

    def _general_response(self, user_input: str) -> str:
        """Generate general response"""
        return f"""👋 Hello! I'm here to help you. I can:

            📄 **Answer questions** about uploaded documents
            📅 **Book appointments** (just say 'I want to book an appointment')
            💬 **Have conversations** about various topics

            You asked: "{user_input}"

            What would you like to know more about?"""

    def _get_field_prompt(self, field: str) -> str:
        """Get appropriate prompt for each field"""
        prompts = {
            "name": "What's your full name? 👤",
            "email": "What's your email address? 📧",
            "phone": "What's your phone number? 📱 (Start with +977)",
            "appointment_date": "When would you like to schedule your appointment? 📅\n(e.g., 'next Monday', 'tomorrow', '2024-03-15')",
            "appointment_time": "What time would you prefer for your appointment? 🕐\n(e.g., '2:30 PM', '14:30', '9am', 'morning')",
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
        📅 **Date:** {data['appointment_date']}
        🕐 **Time:** {data['appointment_time']}"""
        
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