# PDF/Excel Chatbot (Local + Optional GPT API)

A Streamlit-based chatbot that lets you upload and chat with your PDF and Excel files. The app indexes documents locally using embeddings and a Chroma vector store, and can answer questions using a local LLM (Ollama) or an API-backed LLM (OpenAI).

## Highlights

- Upload multiple PDF and Excel files and build a local semantic index
- Local embeddings using HuggingFace (sentence-transformers)
- Local vector store (Chroma) persisted to `./chroma_db`
- Local LLM support via Ollama (offline) or optional OpenAI GPT (online)
- Inline source citations (file + page/row) in answers

## Requirements

- Python 3.8+
- Optional: Ollama (for local LLM)
- Optional: OpenAI API key (if you want to use OpenAI GPT)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):

```powershell
# Create virtual environment
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1
```

On macOS / Linux use:

```bash
source venv/bin/activate
```

3. Install Python dependencies:

```powershell
pip install -r requirements.txt
# If you want to use the modern LangChain split packages, install these too:
pip install -U langchain langchain-community langchain-huggingface langchain-chroma langchain-ollama
```

4. (Optional) Install Ollama and pull a local model (for offline LLM):

```powershell
# Install Ollama: https://ollama.ai/
ollama pull mistral
```

5. (Optional) If you plan to use OpenAI instead of local Ollama, set your API key:

```powershell
setx OPENAI_API_KEY "your_api_key_here"
# then restart your shell or set it in the current session:
$env:OPENAI_API_KEY = "your_api_key_here"
```

## Run the App

```powershell
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Usage

- Upload one or more PDF / Excel files using the file uploader.
- Optionally check "Rebuild index" to delete the existing index and recreate it from uploaded files.
- Ask questions in the chat box. Responses include source citations when available.

## How it works (brief)

- The app extracts text from PDFs (page-by-page) and Excel rows.
- Text is split into chunks via a RecursiveCharacterTextSplitter.
- Chunks are embedded using a HuggingFace embeddings model (sentence-transformers/all-MiniLM-L6-v2).
- Embeddings are stored in a local Chroma DB (persisted to `./chroma_db`).
- A retriever selects top-k chunks for context; the LLM answers based on that context.

## Clearing / Rebuilding the Chroma DB

To start from a clean index, stop the app and delete the `chroma_db` folder:

PowerShell (Windows):
```powershell
Remove-Item -Path ".\chroma_db" -Recurse -Force
```

macOS / Linux:
```bash
rm -rf ./chroma_db
```

If you see errors about files being used by another process on Windows, ensure Streamlit is stopped and try again. The app also includes a "Rebuild index" checkbox which attempts to delete the old DB before re-creating — if that fails, use the manual delete command above.

## Troubleshooting

- EmptyFileError when uploading: ensure the upload completed and the file is not zero bytes. The app will skip empty/unreadable files and warn you.
- LangChain deprecation warnings / import errors: install the recommended modern packages shown above. The code contains fallbacks but installing `langchain-huggingface`, `langchain-chroma`, and `langchain-ollama` removes warnings.
- Chroma persistence: newer Chroma versions auto-persist; the app no longer calls `db.persist()` explicitly.

## Environment variables

- `OPENAI_API_KEY` — (optional) your OpenAI API key if using the OpenAI mode.

## Development & Testing

- A simple smoke test is to run the app and upload the included example PDFs in `uploads/`.
- To run a Python syntax check locally:

```powershell
python -m py_compile app.py
```

## Project Structure

```
.
├── app.py              # Main application file (Streamlit)
├── requirements.txt    # Python dependencies
├── uploads/            # Place example PDFs / Excels here for quick testing
└── chroma_db/          # Persistent vector database storage (auto-created)
```

## Contributing

- PRs welcome. If you change imports for LangChain packages, update `requirements.txt` accordingly and verify the app runs in a fresh virtual environment.

## License

MIT License