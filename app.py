import streamlit as st
import os
from dotenv import load_dotenv
from agents.simple_chatbot import SimpleChatbot
import tempfile
import json
from utils.document_processor import DocumentProcessor
# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False

def initialize_chatbot(api_key, uploaded_files=None):
    """Initialize the chatbot with optional documents"""
    try:
        file_paths = []
        
        if uploaded_files:
            # Save uploaded files temporarily
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_paths.append(tmp_file.name)
        
        # Initialize chatbot
        chatbot = SimpleChatbot(
            google_api_key=api_key,
            documents_path=file_paths if file_paths else None
        )
        
        return chatbot, file_paths
        
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return None, []


def main():
    st.title("🤖 AI Document Chatbot")
    st.markdown("Upload documents and chat with AI that can answer questions and book appointments!")
    
    # Sidebar panel
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("👈 Upload documents here in the sidebar!")
        
        # Assigning Google API key
        api_key = st.text_input(
            "Google API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Get your API key from Google"
        )
        
        if not api_key:
            st.warning("Please enter your Google API Key to continue")
            st.stop()
        
                       
        st.divider()

               
        # Document uploading here
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files for the chatbot to reference"
        )
        
        if uploaded_files and not st.session_state.documents_uploaded:
            # print(uploaded_files)
            with st.spinner("Processing documents..."):
                chatbot, file_paths = initialize_chatbot(api_key, uploaded_files)
                if chatbot:
                    st.session_state.chatbot = chatbot
                    st.session_state.documents_uploaded = True
                    st.success(f"✅ Processed {len(uploaded_files)} documents!")
                    
                    # Cleaning the temporary files
                    for file_path in file_paths:
                        try:
                            os.unlink(file_path)
                        except:
                            pass

        delete_files = st.button("Clear Files")
        st.info("💡 **Note:** Please first manually clear the file uploader above by clicking the 'x' if file already exists, "
                "or refresh the page to upload new files.")
        if delete_files:           
             
            if st.session_state.chatbot:
                # Clearing the chatbot vectorstore
                st.session_state.chatbot.clear_vectorstoredb()
                
                # Clearing the chatbot instance
                st.session_state.chatbot = None
                st.session_state.documents_uploaded = False    
                        
                st.success("✅ Documents and files cleared!")
            else:
                st.warning("No documents to clear")
            st.rerun()

            
        # Initialize chatbot without documents if not already done
        if st.session_state.chatbot is None:
            with st.spinner("Initializing chatbot..."):
                chatbot = SimpleChatbot(api_key)
                if chatbot:
                    st.session_state.chatbot = chatbot
        
        st.divider()
        
        # Booking status
        if st.session_state.chatbot:
            booking_status = st.session_state.chatbot.get_booking_status()
            
            st.header("📅 Booking Status")
            if booking_status["current_step"] == "collecting":
                st.info("🔄 Currently collecting appointment information...")
                next_field = booking_status["next_field"]
                if next_field:
                    st.warning(f"⏳ Waiting for: {next_field.replace('_', ' ').title()}")
                
                # Show collected data
                data = booking_status["data"]
                for field, value in data.items():
                    if value:
                        st.success(f"✅ {field.replace('_', ' ').title()}: {value}")
            elif booking_status["current_step"] == "complete":
                st.success("✅ Booking completed!")
                data = booking_status["data"]
                for field, value in data.items():
                    if value:
                        st.text(f"✅ {field.replace('_', ' ').title()}: {value}")
            else:
                st.text("No active booking - say 'I want to book an appointment' to start")
            
            # Reset button
            if st.button("🔄Reset Booking Form"):
                st.session_state.chatbot.conversational_form.reset()
                st.success("Booking form reset!")
                st.rerun()
    
    # Main chat interface
    if st.session_state.chatbot is None:
        st.error("Chatbot not initialized. Please check your API key.")
        return
    
    # Displaying chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # user provides the input here from streamlit interface
    user_input = st.chat_input("Ask me anything about the documents or say 'book appointment' to schedule a meeting!")
    if user_input:

        # It addes user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):             
            st.markdown(user_input)
        
        # Getting response from the gemini-1.5-flash AI model" 
        with st.chat_message("🤖"):           
            with st.spinner("Thinking..."):             
                response = st.session_state.chatbot.chat(user_input)
                st.markdown(response)
        
        # This adds google AI model response to chat history
        st.session_state.messages.append({"role": "🤖", "content": response})
        
        # Auto-refresh sidebar to show updated booking status
        st.rerun()
    
    
    st.markdown("---")    
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Ask about documents"):
            example_query = "What are the main topics covered in the uploaded documents?"
            st.session_state.messages.append({"role": "user", "content": example_query})
            with st.spinner("Processing..."):
                print('spinning------------')
                response = st.session_state.chatbot.chat(example_query)     
                print(response)           
                st.session_state.messages.append({"role": "🤖", "content": response})
            st.rerun()
    
    with col2:
        if st.button("📞 Book appointment"):
            example_query = "I'd like to book an appointment"
            st.session_state.messages.append({"role": "user", "content": example_query})
            with st.spinner("Processing..."):
                response = st.session_state.chatbot.chat(example_query)
                st.session_state.messages.append({"role": "🤖", "content": response})
            st.rerun()
    
    with col3:
        if st.button("🔄 Clear chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()