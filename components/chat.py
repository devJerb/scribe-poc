import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

class RAGChat:
    def __init__(self, openai_api_key):
        self.directory = "./documents"
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    def create_rag_chain(self):
        """Create RAG chain with retrieval and LLM"""
        if not self.openai_api_key:
            return None
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
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
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=self.openai_api_key)
        
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