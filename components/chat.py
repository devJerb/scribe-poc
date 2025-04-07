import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

class RAGChat:
    def __init__(self, google_api_key):
        self.directory = "./documents"
        self.google_api_key = google_api_key
        os.environ["GOOGLE_API_KEY"] = google_api_key
    
    def create_rag_chain(self):
        """Create RAG chain with retrieval and LLM"""
        if not self.google_api_key:
            return None
        
        # Create embeddings - use GoogleGenerativeAIEmbeddings instead of GooglePalmEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=self.google_api_key,
            model="models/embedding-001"
        )
        
        # Load the vector store
        if not os.path.exists(f"{self.directory}/chroma_db"):
            return None
            
        db = Chroma(persist_directory=f"{self.directory}/chroma_db", embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # Create the RAG prompt template
        template = """
        You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer the question based on the context provided. If you cannot find the answer in the context, say "I don't have enough information to answer this question."
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=1, google_api_key=self.google_api_key)
        
        # Create the chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain
        
    def get_response(self, question):
        """Get response for a user question"""
        chain = self.create_rag_chain()
        if not chain:
            return "Error: RAG chain could not be created. Please check your API key and document selection."
            
        try:
            response = chain.run(question)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
