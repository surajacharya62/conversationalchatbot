import streamlit as st
import os
from dotenv import load_dotenv
from agents.simple_chatbot import SimpleChatbot

# loading environment variables
load_dotenv()

# page configuration
st.set_page_config(
    page_title="Welcome to AI Conversational Chatbot",
    page_icon="🤖",
    layout="wide"
)

# initializing session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

def main():
    st.title("🤖 Welcome to AI Conversational Chatbot")
    st.markdown("Upload documents and chat with AI that can answer questions and book appointments!")

    # sidebar panel
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "Google API Key",
            type="password",
            value="AIzaSyBh6_bbvUCDTOdQPlW-lRfdQUXblVXPOeI",
            help="Get your API key from Google AI Studio"
        )
        
        if not api_key:
            st.warning("Please enter your Google API Key to continue")
            st.stop()
        
        st.divider()        
        
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files for the chatbot to reference"
        )        
        
        # Process uploaded files directly without temp files
        if uploaded_files:
            if st.button("🔄 Process Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        # Initialize chatbot if needed
                        if not st.session_state.chatbot:
                            st.session_state.chatbot = SimpleChatbot(api_key)
                        
                        # Clear old documents
                        st.session_state.chatbot.clear_documents()
                        
                        # Process files directly
                        success = st.session_state.chatbot.setup_documents_direct(uploaded_files)
                        
                        if success:
                            st.session_state.documents_loaded = True
                            st.success(f"✅ Processed {len(uploaded_files)} documents!")
                            
                            for file in uploaded_files:
                                st.text(f"📁 {file.name} - {file.size} bytes")
                        else:
                            st.error("❌ Document processing failed!")
                            st.session_state.documents_loaded = False
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.documents_loaded = False
        
        # Show status
        if st.session_state.documents_loaded:
            st.success("✅ Documents ready for questions!")
        elif uploaded_files:
            st.info("📄 Click 'Process Documents' to load files")
        else:
            st.info("📄 No documents uploaded")
        
        if st.button("🗑️ Clear Documents"):
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_documents()
                st.session_state.documents_loaded = False
                st.success("✅ Documents cleared!")
                st.rerun()
        
        # Initialize chatbot
        if st.session_state.chatbot is None:
            with st.spinner("Initializing chatbot..."):
                st.session_state.chatbot = SimpleChatbot(api_key)
        
        st.divider()
        
        # Booking status
        if st.session_state.chatbot:
            booking_status = st.session_state.chatbot.get_booking_status()
            
            st.header("📅 Booking Status")
            if booking_status["current_step"] == "collecting":
                st.info("🔄 Collecting appointment info...")
                next_field = booking_status["next_field"]
                if next_field:
                    st.warning(f"⏳ Need: {next_field.replace('_', ' ').title()}")
                
                for field, value in booking_status["data"].items():
                    if value:
                        st.success(f"✅ {field.replace('_', ' ').title()}: {value}")
            elif booking_status["current_step"] == "complete":
                st.success("✅ Booking completed!")
            else:
                st.text("Say 'book appointment' to start")
            
            if st.button("🔄 Reset Booking"):
                st.session_state.chatbot.conversational_form.reset()
                st.success("Booking reset!")
                st.rerun()
    
    # Main chat interface
    if st.session_state.chatbot is None:
        st.error("Chatbot not initialized.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask about documents or say 'book appointment'!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("🤖"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.chat(prompt)
                    st.markdown(response)
                except Exception as e:
                    response = f"Error: {str(e)}"
                    st.error(response)
        
        st.session_state.messages.append({"role": "🤖", "content": response})
        st.rerun()
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Ask about documents"):
            if st.session_state.documents_loaded:
                example_query = "What are the main topics in the documents?"
                st.session_state.messages.append({"role": "user", "content": example_query})
                with st.spinner("Processing..."):
                    response = st.session_state.chatbot.chat(example_query)
                    st.session_state.messages.append({"role": "🤖", "content": response})
                st.rerun()
            else:
                st.warning("⚠️ Please upload and process documents first!")
    
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