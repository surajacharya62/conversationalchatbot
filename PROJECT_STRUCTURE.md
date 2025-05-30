## Project Structure
ConversationalChatBotUsingLangchain/ <br>
├── app.py                         <br>
├── requirements.txt                <br>
├── .env                          <br>
├── .gitignore                     <br>
├── README.md                      <br>
│
├── utils/                         <br>
│   ├── __init__.py               <br>
│   ├── validators.py             <br>
│   └── document_processor.py     <br>
│
├── agents/                       <br>
│   ├── __init__.py              <br>
│   └── simple_chatbot.py        <br>
│




## Architecture
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface  <br>                │
│                         (app.py)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │<br>
┌─────────────────────▼───────────────────────────────────────┐
│                   SimpleChatbot    <br>                        │
│                (agents/simple_chatbot.py)  <br>                │
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