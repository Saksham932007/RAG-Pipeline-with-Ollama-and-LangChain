#
# --- RAG Pipeline with Ollama and LangChain ---
#
# Objective:
# This script implements a full RAG pipeline using local models via Ollama.
# It will load documents, split them into chunks, embed them, store them in a
# vector database, and use a language model to answer questions based on the
# retrieved context.

# --- 1. Imports ---
# Import necessary libraries and modules from LangChain.
# We need loaders for documents, text splitters, the Ollama class for models,
# the Chroma vector store, and components to build the RAG chain.

import os
import sys
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("--- RAG Pipeline with Ollama and LangChain Initialized ---")

# --- 2. Configuration and Model Initialization ---
# Define the models to be used. 'nomic-embed-text' is for creating embeddings,
# and 'llama3' is for generating responses.

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        # Test connection by initializing a simple model
        test_llm = Ollama(model="llama3")
        # Try a simple generation to check if the model is available
        test_response = test_llm.invoke("Hello", stop=["Hi"])
        return True
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please make sure:")
        print("1. Ollama is installed and running")
        print("2. You have pulled the required models:")
        print("   ollama pull nomic-embed-text")
        print("   ollama pull llama3")
        return False

print("Checking Ollama connection...")
if not check_ollama_connection():
    print("Failed to connect to Ollama. Exiting...")
    sys.exit(1)

print("Initializing Ollama models...")
llm = Ollama(model="llama3")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
print("Models initialized successfully.")

# --- 3. Document Loading and Processing ---
# Load documents from the './data' directory. We will use a DirectoryLoader
# that can handle both .txt and .pdf files.
# Then, split the loaded documents into smaller chunks for processing.

print("Loading and processing documents...")

# Check if data directory exists
data_dir = './data'
if not os.path.exists(data_dir):
    print(f"Data directory '{data_dir}' not found. Creating it...")
    os.makedirs(data_dir)
    print(f"Please add some .txt or .pdf files to the '{data_dir}' directory and run the script again.")
    sys.exit(1)

# Load both PDF and text files
loaders = []

# Load PDF files
try:
    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()
    if pdf_docs:
        loaders.extend(pdf_docs)
        print(f"Loaded {len(pdf_docs)} PDF documents.")
except Exception as e:
    print(f"Error loading PDF files: {e}")

# Load text files
try:
    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()
    if txt_docs:
        loaders.extend(txt_docs)
        print(f"Loaded {len(txt_docs)} text documents.")
except Exception as e:
    print(f"Error loading text files: {e}")

documents = loaders

if not documents:
    print(f"No documents found in the '{data_dir}' directory.")
    print("Please add some .txt or .pdf files and run the script again.")
    sys.exit(1)

print(f"Total documents loaded: {len(documents)}")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
splits = text_splitter.split_documents(documents)
print(f"Documents split into {len(splits)} chunks.")

# --- 4. Vector Store Creation ---
# Create a Chroma vector store from the document splits and the Ollama embeddings.
# This will embed the document chunks and store them in a local directory.

print("Creating Chroma vector store...")
print("This may take a while as we're generating embeddings for all document chunks...")

# Create a persistent Chroma vector store
persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    persist_directory=persist_directory
)

print("Vector store created successfully.")
print(f"Vector database persisted to: {persist_directory}")

# --- 5. RAG Chain Definition ---
# Define the RAG chain using LangChain Expression Language (LCEL).
# This chain will:
# 1. Retrieve relevant documents from the vector store.
# 2. Format the retrieved documents and the user's question into a prompt.
# 3. Pass the prompt to the language model.
# 4. Parse the output.

print("Defining RAG chain...")

# Create retriever from vector store
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
)

# Define the prompt template
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer based on the context, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

# Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain defined successfully.")

# --- 6. Interaction Loop ---
# Create a loop to continuously ask for user input and provide answers
# using the RAG chain until the user types 'exit'.

def main_chat_loop():
    """Main interaction loop for chatting with documents"""
    print("\n" + "="*60)
    print("--- Start Chatting with your Documents ---")
    print("="*60)
    print("Type 'exit' to end the conversation.")
    print("Type 'info' to see information about loaded documents.")
    print("="*60)

    while True:
        try:
            question = input("\n🤔 Your Question: ").strip()
            
            if question.lower() == 'exit':
                print("\n👋 Goodbye! Thanks for using the RAG pipeline.")
                break
            
            if question.lower() == 'info':
                print(f"\n📚 Loaded Documents Information:")
                print(f"   - Total documents: {len(documents)}")
                print(f"   - Total chunks: {len(splits)}")
                print(f"   - Vector store location: {persist_directory}")
                continue
            
            if not question:
                print("Please enter a valid question.")
                continue

            print("\n🔍 Searching relevant documents...")
            
            # Invoke the RAG chain with the user's question
            answer = rag_chain.invoke(question)
            
            print(f"\n🤖 Assistant: {answer}")
            
            # Show which documents were retrieved (optional)
            relevant_docs = retriever.invoke(question)
            if relevant_docs:
                print(f"\n📄 Based on {len(relevant_docs)} relevant document chunks.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the RAG pipeline.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with a different question.")

# --- 7. Main Execution ---
if __name__ == "__main__":
    try:
        main_chat_loop()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)