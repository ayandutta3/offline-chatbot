# PDF/Excel Chatbot with Local LLM

A Streamlit-based application that allows users to chat with their PDF and Excel documents using local LLM (Language Learning Model) and embeddings. The app processes documents locally, maintaining privacy and working offline.

## Features

- 📄 Upload and process multiple PDF and Excel files
- 🔒 Local processing - no data leaves your machine
- 💾 Persistent document indexing using Chroma DB
- 🤖 Local LLM integration via Ollama
- 📊 Support for both PDF and Excel file formats
- 🔍 Semantic search using sentence-transformers
- 💬 Interactive chat interface

## Requirements

- Python 3.8+
- Ollama (for local LLM)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install Ollama and pull the Mistral model:
```bash
# Install Ollama from https://ollama.ai/
ollama pull mistral
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the displayed URL (typically http://localhost:8501)

3. Upload your PDF or Excel files using the file uploader

4. Wait for the indexing process to complete (first run will take longer)

5. Start chatting with your documents!

## How It Works

1. **Document Processing**:
   - PDFs are processed using PyPDF
   - Excel files are processed using Pandas
   - Text is extracted and split into chunks

2. **Indexing**:
   - Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings
   - Stores vectors in a local Chroma database
   - Persistent storage prevents re-indexing on restart

3. **Question Answering**:
   - Uses Ollama's Mistral model for local LLM processing
   - Retrieves relevant context using vector similarity
   - Generates responses based on retrieved context

## Project Structure

```
.
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── uploads/           # Temporary storage for uploaded files
└── chroma_db/         # Persistent vector database storage
```

## Dependencies

- streamlit: Web interface
- pypdf: PDF processing
- pandas: Excel file processing
- langchain: LLM framework
- chromadb: Vector database
- sentence-transformers: Text embeddings
- torch: Required for transformers

## Notes

- First-time indexing may take several minutes depending on document size
- The Chroma DB is persisted locally, so subsequent runs are faster
- All processing happens locally - no internet connection required after setup
- Requires sufficient RAM for document processing and LLM operation

## Limitations

- Large documents may require significant RAM
- Processing speed depends on local hardware capabilities
- Currently supports PDF and Excel files only