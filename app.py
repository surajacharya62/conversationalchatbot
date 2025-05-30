import streamlit as st
import os
from dotenv import load_dotenv
from agents.simple_chatbot import SimpleChatbot
import tempfile
import json

# Loading environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initializing session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

def get_file_signature(uploaded_files):
    """Create a signature for uploaded files to detect changes"""
    if not uploaded_files:
        return None
    return [(f.name, f.size) for f in uploaded_files]

def main():
    st.title("🤖 AI Document Chatbot")
    st.markdown("Upload documents and chat with AI that can answer questions and book appointments!")
    
    # Sidebar panel
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("👈 Upload documents here in the sidebar!")
        
        # FIXED: Secure API key handling
        api_key = st.text_input(
            "Google API Key",
            type="password",
            # value=os.getenv("GOOGLE_API_KEY", ""),  
            value="AIzaSyBh6_bbvUCDTOdQPlW-lRfdQUXblVXPOeI",
            help="Get your API key from Google AI Studio"
        )
        
        if not api_key:
            st.warning("Please enter your Google API Key to continue")
            st.stop()
        
        st.divider()
        
        # Document uploading - FIXED: Proper file change detection
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files for the chatbot to reference"
        )
        
        # FIXED: Proper file change detection and processing
        current_file_signature = get_file_signature(uploaded_files)
        files_changed = current_file_signature != st.session_state.last_uploaded_files
        
        if uploaded_files and files_changed:
            st.session_state.last_uploaded_files = current_file_signature
            st.session_state.documents_loaded = False
            st.session_state.vectorstore_ready = False
            
            with st.spinner("🔄 Processing documents (refreshing database)..."):
                # Initialize chatbot first if not exists
                if not st.session_state.chatbot:
                    st.session_state.chatbot = SimpleChatbot(api_key)
                else:
                    # Force clear everything first
                    st.session_state.chatbot.clear_documents()
                
                # Save uploaded files temporarily and process them
                file_paths = []
                try:
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            file_paths.append(tmp_file.name)
                    
                    # Show file info
                    st.info(f"📁 Processing files: {[f.name for f in uploaded_files]}")
                    
                    # Process documents with force refresh
                    st.session_state.chatbot.setup_documents(file_paths, force_refresh=True)
                    
                    # Verify documents are loaded
                    if st.session_state.chatbot.document_processor.vectorstore:
                        st.session_state.documents_loaded = True
                        st.session_state.vectorstore_ready = True
                        st.success(f"✅ Processed {len(uploaded_files)} documents!")
                        st.success("🔄 Database refreshed with new content!")
                    else:
                        st.error("❌ Document processing failed!")
                        st.session_state.documents_loaded = False
                        st.session_state.vectorstore_ready = False
                    
                    # Show which files were processed
                    for file in uploaded_files:
                        st.text(f"📁 {file.name} ({file.type}) - {file.size} bytes")
                        
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                    st.session_state.documents_loaded = False
                    st.session_state.vectorstore_ready = False

                finally:
                    # Clean up temporary files
                    for file_path in file_paths:
                        try:
                            os.unlink(file_path)
                        except:
                            pass
        
        elif uploaded_files and not files_changed:
            st.info(f"📄 {len(uploaded_files)} documents already loaded")
            for file in uploaded_files:
                st.text(f"📁 {file.name} ({file.type}) - {file.size} bytes")
        
        # FIXED: Accurate document status display
        if st.session_state.chatbot and hasattr(st.session_state.chatbot, 'document_processor'):
            vectorstore = st.session_state.chatbot.document_processor.vectorstore
            
            if st.session_state.vectorstore_ready and vectorstore and uploaded_files:
                st.success("✅ Documents are loaded and ready for questions!")
            elif uploaded_files and not st.session_state.documents_loaded:
                st.warning("⚠️ Documents uploaded but processing failed")
            elif uploaded_files:
                st.warning("⚠️ Documents uploaded but not yet processed")
            else:
                st.info("ℹ️ No documents uploaded yet")
        
        # FIXED: Improved clear button functionality
        if st.button("🗑️ Clear All Documents"):
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_documents()
                st.session_state.last_uploaded_files = None
                st.session_state.documents_loaded = False
                st.session_state.vectorstore_ready = False
                st.success("✅ All documents cleared! Upload new documents to refresh.")
                st.rerun()
            else:
                st.warning("No documents to clear")
        
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
            if st.button("🔄 Reset Booking Form"):
                st.session_state.chatbot.conversational_form.reset()
                st.success("Booking form reset!")
                st.rerun()
    
    # Main chat interface
    if st.session_state.chatbot is None:
        st.error("Chatbot not initialized. Please check your API key.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input - FIXED: Better error handling
    if prompt := st.chat_input("Ask me anything about the documents or say 'book appointment' to schedule a meeting!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from AI model gemini-flash
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.chat(prompt)
                    st.markdown(response)
                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}"
                    st.error(error_msg)
                    response = "I'm sorry, I encountered an error. Please try rephrasing your question or check if documents are properly uploaded."
                    st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Auto-refresh sidebar to show updated booking status
        st.rerun()
    
    # Example queries
    st.markdown("---")
    st.markdown("### 💡 Try these examples:")
    
    col1, col2, col3 = st.columns(3)
    
    # FIXED: Check if documents are actually loaded before allowing document questions
    with col1:
        if st.button("📋 Ask about documents"):
            if (st.session_state.chatbot and 
                st.session_state.vectorstore_ready and 
                st.session_state.chatbot.document_processor.vectorstore):
                example_query = "What are the main topics covered in the uploaded documents?"
                st.session_state.messages.append({"role": "user", "content": example_query})
                with st.spinner("Processing..."):
                    response = st.session_state.chatbot.chat(example_query)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            else:
                st.warning("⚠️ Please upload documents first using the sidebar!")
    
    with col2:
        if st.button("📞 Book appointment"):
            example_query = "I'd like to book an appointment"
            st.session_state.messages.append({"role": "user", "content": example_query})
            with st.spinner("Processing..."):
                response = st.session_state.chatbot.chat(example_query)
                st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col3:
        if st.button("🔄 Clear chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()