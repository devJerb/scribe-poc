__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import tempfile
import streamlit as st
from chat import RAGChat
from document import DocumentProcessor
from constant import DIRECTORY as directory

# Access the credentials
if "google_credentials" in st.secrets:
    # You can access individual fields
    api_key = st.secrets.get("google_api_key", "")
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Or write the full credentials to a temporary file if needed
    credentials = st.secrets["google_credentials"]
    # Convert AttrDict to a regular dict before JSON serialization
    credentials_dict = dict(credentials)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(credentials_dict, f)
        credentials_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Page config
st.set_page_config(page_title="Scribe | RAG + Chat", page_icon="🤖")
st.title("📚 Scribe - Chat with Documents") 

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_list" not in st.session_state:
    st.session_state.document_list = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = {}
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

# Initialize document processor
doc_processor = DocumentProcessor()

# Display welcome message if first time
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hi there! I'm your AI assistant. You can chat with me normally, or upload documents to enable RAG capabilities. How can I help you today?"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar for configuration and document management
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Google API Key", type="password")
    
    if st.button("Set API Key") or st.session_state.api_key_set:
        st.session_state.api_key_set = True
        st.success("API Key set! You can now chat with or without documents.")
    
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents", 
        accept_multiple_files=True, 
        type=["txt", "pdf", "docx", "doc"]
    )
    
    if uploaded_files:
        if st.button("Process New Documents"):
            if not google_api_key:
                st.error("Please enter your Google API key first")
            else:
                with st.spinner("Processing documents..."):
                    for file in uploaded_files:
                        if file.name not in st.session_state.document_list:
                            texts = doc_processor.process_document(file)
                            st.session_state.processed_docs[file.name] = texts
                            if file.name not in st.session_state.document_list:
                                st.session_state.document_list.append(file.name)
                    st.success(f"Processed {len(uploaded_files)} new documents!")
    
    # List and select documents
    if st.session_state.document_list:
        st.header("Available Documents")
        st.write("Select documents to include in your RAG:")
        
        # Document selection checkboxes
        selected_docs = {}
        for doc in st.session_state.document_list:
            selected_docs[doc] = st.checkbox(doc, value=True)
        
        # Store selected docs for RAG chain
        st.session_state.selected_docs = [doc for doc, selected in selected_docs.items() if selected]
        
        if st.button("Update RAG with Selected Documents"):
            if not st.session_state.selected_docs:
                st.error("Please select at least one document")
            elif not google_api_key:
                st.error("Please enter your Google API key")
            else:
                with st.spinner("Updating RAG with selected documents..."):
                    # Collect texts from selected documents
                    all_texts = []
                    for doc_name in st.session_state.selected_docs:
                        all_texts.extend(st.session_state.processed_docs[doc_name])
                    
                    # Create/update vector store
                    success = doc_processor.update_vector_store(all_texts, google_api_key)
                    if success:
                        st.session_state.db_initialized = True
                        st.success(f"RAG updated with {len(all_texts)} chunks from {len(st.session_state.selected_docs)} documents!")
                    else:
                        st.error("Failed to update vector store")

# Chat interface
user_input = st.chat_input("Ask me anything...")
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        if not google_api_key:
            st.error("Please enter your Google API key in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                rag_chat = RAGChat(google_api_key)
                response = rag_chat.get_response(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Show current RAG status
if st.session_state.db_initialized and hasattr(st.session_state, 'selected_docs') and st.session_state.selected_docs:
    st.info(f"RAG enabled with {len(st.session_state.selected_docs)} documents: {', '.join(st.session_state.selected_docs)}")
elif st.session_state.api_key_set:
    st.info("Chatting in general mode. Upload documents to enable RAG capabilities.")
