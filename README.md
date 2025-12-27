# Chat myPDF: RAG System with LangChain, Ollama, and FAISS

A Retrieval-Augmented Generation (RAG) system that enables you to chat with your PDF documents using local LLMs through Ollama. This project implements a complete RAG pipeline using LangChain for orchestration, FAISS for vector storage, and Ollama for embeddings and language models.

## 🚀 Features

- **Document Processing**: Load and process multiple PDF documents from directories
- **Intelligent Chunking**: Recursive text splitting with configurable chunk size and overlap
- **Vector Embeddings**: Uses Ollama's `nomic-embed-text` model for generating embeddings
- **FAISS Vector Store**: Efficient similarity search using Facebook AI Similarity Search
- **RAG Pipeline**: Retrieval-Augmented Generation with Llama 3.2 via Ollama
- **Multiple Retrieval Strategies**: Supports similarity search and MMR (Maximum Marginal Relevance) retrieval

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.8+**
2. **Ollama** - [Install Ollama](https://ollama.ai/)
3. **Required Ollama Models**:
   - `nomic-embed-text` (for embeddings)
   - `llama3.2` (for language model)

### Installing Ollama Models

After installing Ollama, pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Chat_my_pdf
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
.
├── 3.chat_myPDF.ipynb          # Main Jupyter notebook
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── PDF_dataset/                 # Directory containing PDF documents
│   ├── gym supplements/
│   └── health supplements/
└── .env                         # Environment variables (create if needed)
```

## 🔧 Configuration

1. **Start Ollama service**:
   Make sure Ollama is running on `http://localhost:11434` (default port)

2. **Prepare your PDF documents**:
   Place your PDF files in the `PDF_dataset/` directory. The notebook will automatically discover all PDFs in subdirectories.

3. **Environment Variables** (optional):
   Create a `.env` file if you need to configure custom settings:
   ```
   OLLAMA_BASE_URL=http://localhost:11434
   ```

## 📖 Usage

### Running the Notebook

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open `3.chat_myPDF.ipynb`** and run cells sequentially

### Workflow

The notebook follows this workflow:

1. **Document Ingestion**: Loads all PDF files from the `PDF_dataset/` directory
2. **Document Chunking**: Splits documents into smaller chunks (1000 characters with 100 character overlap)
3. **Vector Embedding**: Generates embeddings using Ollama's embedding model
4. **Vector Store Creation**: Builds a FAISS index for efficient similarity search
5. **Retrieval**: Implements MMR (Maximum Marginal Relevance) retrieval for diverse results
6. **RAG Chain**: Combines retrieval with Llama 3.2 to answer questions based on document context

### Example Queries

After running the notebook, you can ask questions like:

- "What is used to gain muscle mass?"
- "What is used to reduce weight?"
- "What are the side effects of supplements?"
- "What are the benefits of supplements?"

## 🔍 How It Works

### 1. Document Loading
- Uses `PyMuPDFLoader` from LangChain to extract text from PDF files
- Automatically discovers all PDFs in the dataset directory

### 2. Text Chunking
- Implements `RecursiveCharacterTextSplitter` with:
  - Chunk size: 1000 characters
  - Chunk overlap: 100 characters
- Ensures context preservation across chunks

### 3. Embedding Generation
- Uses Ollama's `nomic-embed-text` model
- Generates vector embeddings for each document chunk
- Embeddings are stored in FAISS for fast similarity search

### 4. Vector Store
- FAISS (Facebook AI Similarity Search) for efficient vector operations
- Supports similarity search and MMR retrieval
- Can be saved locally for reuse

### 5. Retrieval-Augmented Generation
- **Retriever**: Uses MMR to find diverse, relevant document chunks
- **Prompt Template**: Custom prompt that instructs the model to answer based on retrieved context
- **LLM**: Llama 3.2 via Ollama for generating answers
- **Output**: Formatted responses based on document content

## 🎯 Key Components

- **LangChain**: Framework for building LLM applications
- **Ollama**: Local LLM inference server
- **FAISS**: Vector similarity search library
- **PyMuPDF**: PDF text extraction
- **TikToken**: Token counting for chunk size validation

## 📝 Customization

### Adjusting Chunk Size
Modify the `RecursiveCharacterTextSplitter` parameters:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust based on your documents
    chunk_overlap=100     # Adjust overlap for context preservation
)
```

### Changing Retrieval Parameters
Modify the retriever configuration:
```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        'k': 3,              # Number of documents to retrieve
        'fetch_k': 100,      # Number of documents to fetch for MMR
        'lambda_mult': 1     # Diversity parameter (0-1)
    }
)
```

### Using Different Models
Change the Ollama models:
```python
# For embeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

# For LLM
model = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
```

## 🐛 Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check if models are installed: `ollama list`
- Verify the base URL matches your Ollama configuration

### Memory Issues
- Reduce chunk size if processing large documents
- Process documents in batches
- Use `faiss-cpu` instead of `faiss-gpu` if GPU memory is limited

### Empty Retrieval Results
- Check if documents were loaded successfully
- Verify embeddings were generated correctly
- Adjust retrieval parameters (k, fetch_k)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This project uses local LLMs via Ollama, ensuring privacy and no data transmission to external services. All processing happens on your local machine.

