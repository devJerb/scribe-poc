import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

class RAGChat:
    def __init__(self, google_api_key):
        self.directory = "./documents"
        self.google_api_key = google_api_key
        os.environ["GOOGLE_API_KEY"] = google_api_key
        # Initialize the LLM for direct conversation
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite", 
            temperature=1, 
            google_api_key=self.google_api_key
        )
    
    def create_rag_chain(self):
        """Create RAG chain with retrieval and LLM"""
        if not self.google_api_key:
            return None
        
        # Check if vector store exists
        if not os.path.exists(f"{self.directory}/chroma_db"):
            return None
            
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=self.google_api_key,
            model="models/embedding-001"
        )
        
        # Load the vector store
        db = Chroma(persist_directory=f"{self.directory}/chroma_db", embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # Create the RAG prompt template
        template = """
        You're a friendly, helpful assistant having a conversation with the user. You have access to some documents that might be relevant to the conversation.

        If the user's question relates to the documents, use the context below to help inform your response. However, don't be limited by the documents - feel free to have a natural conversation and draw on your general knowledge when appropriate.

        Context from documents:
        {context}

        User: {question}

        Assistant: 
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        # Create the chain
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain
    
    def create_conversation_chain(self):
        """Create a simple conversation chain for when no documents are available"""
        conversation_template = """
        You're a friendly, helpful assistant having a natural conversation with the user. Respond in a warm, conversational way while being helpful and informative.
        
        If they ask about specific documents or document content, you can mention that no documents have been uploaded yet, but you can still try to answer their question based on your general knowledge. Be helpful rather than focusing on what you can't do.

        User: {question}
        
        Assistant:
        """
        
        prompt = PromptTemplate(
            input_variables=["question"],
            template=conversation_template,
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain
        
    def get_response(self, question):
        """Get response for a user question, using RAG if available, otherwise conversation"""
        try:
            # First try to use RAG if documents are available
            rag_chain = self.create_rag_chain()
            
            if rag_chain:
                # Documents exist, use RAG
                response = rag_chain.run(question)
                return response
            else:
                # No documents, use general conversation
                conversation_chain = self.create_conversation_chain()
                response = conversation_chain.run(question=question)
                return response
                
        except Exception as e:
            # Fallback to direct LLM if any errors occur
            try:
                response = self.llm.invoke(f"User question: {question}\n\nProvide a helpful response:")
                return response.content
            except:
                return f"I'm having trouble processing your request. Please check your API key or try asking a different question."
