import os
import time
import shutil
import random
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from constant import DIRECTORY as directory

class DocumentProcessor:
    def __init__(self):
        # Create necessary directories
        self.directory = directory
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
    
    def update_vector_store(self, texts, google_api_key):
        """Create or update vector store with provided texts"""
        if not texts:
            return False
            
        # Fix permissions on the directory first
        chroma_dir = f"{self.directory}/chroma_db"
        
        # Ensure directory exists with proper permissions
        if not os.path.exists(chroma_dir):
            os.makedirs(chroma_dir, mode=0o755)
        else:
            # Fix permissions on existing directory
            os.chmod(chroma_dir, 0o755)
            
            # Fix permissions on all files in the directory
            for root, dirs, files in os.walk(chroma_dir):
                for dir in dirs:
                    os.chmod(os.path.join(root, dir), 0o755)
                for file in files:
                    os.chmod(os.path.join(root, file), 0o644)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=google_api_key,
            model="models/embedding-001"
        )
        
        try:
            # Try to open existing DB first
            db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
            
            # Process in smaller batches with retry logic
            batch_size = 10  # Adjust based on your quota limits
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                success = False
                retries = 0
                max_retries = 5
                
                while not success and retries < max_retries:
                    try:
                        db.add_documents(batch)
                        success = True
                        print(f"Successfully processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                    except Exception as e:
                        retries += 1
                        if "429" in str(e) or "Resource has been exhausted" in str(e):
                            # Calculate exponential backoff time
                            wait_time = (2 ** retries) + random.uniform(0, 1)
                            print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            # If it's not a rate limit error, raise to outer exception handler
                            raise e
                
                if not success:
                    raise Exception(f"Failed to process batch after {max_retries} retries")
                
                # Add a small delay between successful batches
                time.sleep(1)
                
        except Exception as e:
            print(f"Error updating existing DB: {e}")
            # If that fails, create new DB
            if os.path.exists(chroma_dir):
                shutil.rmtree(chroma_dir)
                
            # Create new database with batch processing for new DB creation
            db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
            
            # Process in smaller batches with retry logic
            batch_size = 10  # Adjust based on your quota limits
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                success = False
                retries = 0
                max_retries = 5
                
                while not success and retries < max_retries:
                    try:
                        db.add_documents(batch)
                        success = True
                        print(f"Successfully processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                    except Exception as e:
                        retries += 1
                        if "429" in str(e) or "Resource has been exhausted" in str(e):
                            # Calculate exponential backoff time
                            wait_time = (2 ** retries) + random.uniform(0, 1)
                            print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            # If it's not a rate limit error, re-raise
                            raise e
                
                if not success:
                    raise Exception(f"Failed to process batch after {max_retries} retries")
                
                # Add a small delay between successful batches
                time.sleep(1)
        
        # Make sure to persist at the end
        db.persist()
        return True
