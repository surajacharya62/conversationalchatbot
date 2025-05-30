## Project Structure
ConversationalChatBotUsingLangchain/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (API keys)
├── .gitignore                     # Git ignore file
├── README.md                      # Project documentation
│
├── utils/                         # Utility modules
│   ├── __init__.py               # Makes utils a Python package
│   ├── validators.py             # Input validation (email, phone, date/time)
│   └── document_processor.py     # Document loading and vector processing
│
├── agents/                       # AI agent modules
│   ├── __init__.py              # Makes agents a Python package
│   └── simple_chatbot.py        # Main chatbot logic and conversation handling
│
└── sample_documents/             # Sample files for testing
    ├── company_info.txt          # Sample company information
    └── product_manual.pdf        # Sample product documentation



## Architecture
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                  │
│                         (app.py)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   SimpleChatbot                            │
│                (agents/simple_chatbot.py)                  │
│  ┌─────────────────┬─────────────────┬──────────────────┐  │
│  │ Document Q&A    │ Booking Form    │ General Chat     │  │
│  │                 │                 │                  │  │
└──┼─────────────────┼─────────────────┼──────────────────┼──┘
   │                 │                 │                  │
   ▼                 ▼                 ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│Document      │ │Validators    │ │Google Gemini │ │ChromaDB      │
│Processor     │ │              │ │LLM           │ │Vector Store  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘