import os
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class DocumentProcessor:
    def __init__(self):
        # Create necessary directories
        self.directory = "./documents"
        os.makedirs(f"{self.directory}/temp_docs", exist_ok=True)
        os.makedirs(f"{self.directory}/processed_docs", exist_ok=True)
        
    def process_document(self, file):
        """Process a single document file"""
        # Save uploaded file temporarily
        file_path = f"{self.directory}/temp_docs/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Select appropriate loader based on file extension
        file_extension = os.path.splitext(file.name)[1].lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
            else:
                # Default to text loader for other types
                loader = TextLoader(file_path)
                
            documents = loader.load()
            
            # Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            texts = text_splitter.split_documents(documents)
            
            # Move to processed folder
            processed_path = f"{self.directory}/processed_docs/{file.name}"
            shutil.move(file_path, processed_path)
            
            return texts
            
        except Exception as e:
            # If error occurs, clean up and re-raise with more info
            if os.path.exists(file_path):
                os.remove(file_path)
            raise Exception(f"Error processing {file.name}: {str(e)}")
    
    def update_vector_store(self, texts, openai_api_key):
        """Create or update vector store with provided texts"""
        if not texts:
            return False
            
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Clear existing database if it exists
        if os.path.exists(f"{self.directory}/chroma_db"):
            shutil.rmtree(f"{self.directory}/chroma_db")
        
        # Create new database
        db = Chroma.from_documents(texts, embeddings, persist_directory=f"{self.directory}/chroma_db")
        db.persist()
        
        return True
