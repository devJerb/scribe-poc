import streamlit as st
from document import DocumentProcessor
from chat import RAGChat

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 RAG Chatbot with LangChain")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_list" not in st.session_state:
    st.session_state.document_list = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = {}
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

# Initialize document processor
doc_processor = DocumentProcessor()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar for configuration and document management
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["txt", "pdf", "docx"])
    
    if uploaded_files:
        if st.button("Process New Documents"):
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
            elif not openai_api_key:
                st.error("Please enter your OpenAI API key")
            else:
                with st.spinner("Updating RAG with selected documents..."):
                    # Collect texts from selected documents
                    all_texts = []
                    for doc_name in st.session_state.selected_docs:
                        all_texts.extend(st.session_state.processed_docs[doc_name])
                    
                    # Create/update vector store
                    success = doc_processor.update_vector_store(all_texts, openai_api_key)
                    if success:
                        st.session_state.db_initialized = True
                        st.success(f"RAG updated with {len(all_texts)} chunks from {len(st.session_state.selected_docs)} documents!")
                    else:
                        st.error("Failed to update vector store")

# Chat interface
user_input = st.chat_input("Ask a question about your documents")
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not st.session_state.db_initialized:
            st.error("Please select and update RAG with documents first.")
        else:
            with st.spinner("Thinking..."):
                rag_chat = RAGChat(openai_api_key)
                response = rag_chat.get_response(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Show current RAG status
if hasattr(st.session_state, 'selected_docs') and st.session_state.selected_docs:
    st.info(f"Current RAG includes {len(st.session_state.selected_docs)} documents: {', '.join(st.session_state.selected_docs)}")