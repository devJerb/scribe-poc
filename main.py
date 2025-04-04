"""
Main entry point for the RAG Chatbot application.
Run this file to start the Streamlit app.
"""
import streamlit
import subprocess
import sys

def main():
    """Run the Streamlit app"""
    print("Starting RAG Chatbot with LangChain...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "components/app.py"])

if __name__ == "__main__":
    main()