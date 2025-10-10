#!/usr/bin/env python3
"""
RAG Pipeline with Ollama and LangChain
Basic structure and imports
"""

import os
import sys

# Import LangChain components
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

print("--- RAG Pipeline with Ollama and LangChain ---")
print("Setting up basic structure...")

# Initialize Ollama models
def initialize_models():
    """Initialize Ollama LLM and embedding models"""
    print("Initializing Ollama models...")
    try:
        llm = Ollama(model="llama3")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        print("Models initialized successfully!")
        return llm, embeddings
    except Exception as e:
        print(f"Error initializing models: {e}")
        return None, None

def load_documents(data_dir="./data"):
    """Load documents from the specified directory"""
    print(f"Loading documents from {data_dir}...")
    
    if not os.path.exists(data_dir):
        print(f"Creating data directory: {data_dir}")
        os.makedirs(data_dir)
        print("Please add some .txt or .pdf files to the data directory")
        return []
    
    documents = []
    
    # Load text files
    try:
        txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        print(f"Loaded {len(txt_docs)} text documents")
    except Exception as e:
        print(f"Error loading text files: {e}")
    
    # Load PDF files  
    try:
        pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        print(f"Loaded {len(pdf_docs)} PDF documents")
    except Exception as e:
        print(f"Error loading PDF files: {e}")
        
    print(f"Total documents loaded: {len(documents)}")
    return documents

def split_documents(documents):
    """Split documents into smaller chunks for processing"""
    if not documents:
        print("No documents to split")
        return []
        
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"Documents split into {len(splits)} chunks")
    return splits

def create_vector_store(splits, embeddings):
    """Create ChromaDB vector store from document splits"""
    if not splits:
        print("No document splits to process")
        return None
        
    print("Creating ChromaDB vector store...")
    print("This may take a while for embedding generation...")
    
    try:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("Vector store created successfully!")
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

# TODO: Add RAG chain
# TODO: Add text splitting  
# TODO: Add vector store
# TODO: Add RAG chain
# TODO: Add chat interface

if __name__ == "__main__":
    print("RAG Pipeline - Under Development")
    llm, embeddings = initialize_models()
    
    if llm and embeddings:
        documents = load_documents()
        if documents:
            splits = split_documents(documents)
            if splits:
                vectorstore = create_vector_store(splits, embeddings)
                if vectorstore:
                    print("RAG pipeline setup complete!")