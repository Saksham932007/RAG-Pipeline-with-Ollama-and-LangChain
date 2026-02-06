## Project Motivation

* This project focuses on AI *systems engineering*, not model research.
* The goal is to understand how retrieval, vector databases, and
* local inference are composed into a reliable end-to-end pipeline.


# RAG Pipeline with Ollama and LangChain

A complete Retrieval-Augmented Generation (RAG) pipeline that runs entirely locally using Ollama for embeddings and text generation, LangChain for orchestration, and ChromaDB for vector storage.

## Features

- **100% Local**: No external API keys required - everything runs on your machine
- **Document Support**: Processes both PDF and text files
- **Persistent Storage**: Vector embeddings are saved locally for reuse
- **Interactive Chat**: Continuous Q&A interface with your documents
- **Error Handling**: Robust error handling and user guidance

## Prerequisites

### 1. Install Ollama

#### Linux/macOS:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows:
Download and install from: https://ollama.ai/download

### 2. Start Ollama Service

```bash
# Start Ollama (if not already running as a service)
ollama serve
```

### 3. Download Required Models

```bash
# Download embedding model (smaller, faster)
ollama pull nomic-embed-text

# Download generation model (larger, more capable)
ollama pull llama3
```

**Note**: Model downloads can be large (several GB). Make sure you have sufficient disk space and a stable internet connection.

### 4. Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Project Structure

```
RAG/
├── rag_with_ollama.py    # Main RAG pipeline script
├── requirements.txt      # Python dependencies
├── data/                # Directory for your documents
│   ├── document1.pdf
│   ├── document2.txt
│   └── ...
├── chroma_db/           # Vector database (created automatically)
└── README.md           # This file
```

## Usage

### 1. Add Your Documents

Place your documents in the `data/` directory:
- Supported formats: `.pdf`, `.txt`
- The script will automatically detect and process all files

### 2. Run the RAG Pipeline

```bash
python rag_with_ollama.py
```

### 3. Start Chatting

Once the script starts, you can:
- Ask questions about your documents
- Type `info` to see document statistics
- Type `exit` to quit

Example interaction:
```
🤔 Your Question: What is the main topic of the documents?

🔍 Searching relevant documents...

🤖 Assistant: Based on the provided context, the main topic appears to be...

📄 Based on 3 relevant document chunks.
```

## Configuration

You can modify the following parameters in `rag_with_ollama.py`:

```python
# Text splitting configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Size of each text chunk
    chunk_overlap=200,      # Overlap between chunks
)

# Retrieval configuration
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Number of chunks to retrieve
)
```

## Troubleshooting

### Common Issues

1. **"Error connecting to Ollama"**
   - Make sure Ollama is installed and running: `ollama serve`
   - Verify models are downloaded: `ollama list`

2. **"No documents found"**
   - Check that files are in the `data/` directory
   - Ensure files have `.pdf` or `.txt` extensions

3. **Slow performance**
   - First run will be slower as embeddings are generated
   - Subsequent runs will be faster as embeddings are cached
   - Consider using smaller chunk sizes for faster processing

4. **Memory issues**
   - Reduce `chunk_size` in the text splitter
   - Process fewer documents at once
   - Ensure sufficient RAM for the selected models

### Model Information

- **nomic-embed-text**: Lightweight embedding model (~274MB)
- **llama3**: Powerful language model (~4.7GB)

Alternative models you can try:
```bash
# Smaller language model
ollama pull llama3:8b

# Different embedding model
ollama pull mxbai-embed-large
```

## Advanced Features

### Custom Prompt Template

Modify the prompt template in the script to change how the AI responds:

```python
template = """You are a helpful research assistant.
Based on the following context, provide a detailed analysis of the question.
Include relevant quotes when possible.

Question: {question}
Context: {context}

Detailed Analysis:"""
```

### Different Retrieval Strategies

```python
# Maximum Marginal Relevance (MMR) for diverse results
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)

# Similarity threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8}
)
```

## Performance Tips

1. **Pre-process large document collections**: Run the script once to generate embeddings, then restart for faster queries
2. **Optimize chunk size**: Experiment with different chunk sizes based on your document types
3. **Use appropriate models**: Balance model size with performance requirements
4. **Persistent storage**: Vector database is automatically saved and reloaded between sessions

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and pull requests to improve the pipeline!
