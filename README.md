# Scribe: RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that lets you chat with your documents.

## Features

- Upload PDF and text documents
- Process documents into searchable chunks
- Chat with your documents using GPT-4o-mini
- Select which documents to include in your knowledge base

## Requirements

- Python 3.7+
- OpenAI API key

## Installation

1. Clone the repository
2. Create a virtual environment (optional):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key (see `.env.example`)

## Usage

Run the application:

```
python main.py
```

The Streamlit interface will open in your web browser. From there:

1. Enter your OpenAI API key in the sidebar
2. Upload documents
3. Process the documents
4. Select which documents to include in the RAG system
5. Start chatting with your documents!

## Project Structure

- `main.py`: Entry point for the application
- `components/`: Core application modules
  - `app.py`: Streamlit user interface
  - `document.py`: Document processing and vectorization
  - `chat.py`: RAG implementation with LangChain
- `documents/`: Storage for uploaded and processed documents
