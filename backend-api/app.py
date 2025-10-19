"""Document Chat Application (app.py)

Minimal, clean FastAPI backend that indexes PDF/Excel files and
answers questions using a local or OpenAI LLM. The original
Streamlit UI was removed and replaced with these endpoints.
"""

# Standard libs
import os
import shutil
import time
import threading
from io import StringIO
from typing import List, Tuple

# Third-party
import pandas as pd
from pypdf import PdfReader
from dotenv import load_dotenv

# LangChain abstractions (try community namespaces as needed)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# Optional import fallbacks for embeddings / chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

# Callback handler (minimal, kept for compatibility with LLMs)
from langchain.callbacks.base import BaseCallbackHandler

# FastAPI
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

load_dotenv()


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Minimal callback used by LLM wrappers for streaming hooks.

    This is a no-op handler here (kept to preserve call signatures).
    Frontends that support streaming should implement their own handlers
    or the backend can be extended to stream via SSE / websockets.
    """

    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token

    def on_llm_end(self, response, **kwargs):
        pass


# -----------------------------------------------------
# Utilities: File Handling
# -----------------------------------------------------


def save_uploaded_file(uploaded_file: UploadFile | object, dest_folder: str = "uploads") -> str:
    """Save uploaded file-like object to disk and return its path.

    Accepts either a FastAPI UploadFile or a Streamlit UploadedFile-like
    object (duck-typed). The function creates the destination folder
    if needed.
    """
    os.makedirs(dest_folder, exist_ok=True)
    filename = getattr(uploaded_file, "filename", None) or getattr(uploaded_file, "name", None)
    out_path = os.path.join(dest_folder, filename)
    # FastAPI UploadFile exposes .file (a SpooledTemporaryFile)
    if hasattr(uploaded_file, "file"):
        with open(out_path, "wb") as f:
            shutil.copyfileobj(uploaded_file.file, f)
    else:
        # fallback for objects with getbuffer()
        with open(out_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return out_path


def load_pdf_pages(pdf_path: str) -> List[Tuple[str, dict]]:
    """Extract text and page metadata from a PDF file."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            meta = {"source": os.path.basename(pdf_path), "page": i, "type": "pdf"}
            pages.append((text, meta))
    return pages


def load_excel_rows(xlsx_path: str) -> List[Tuple[str, dict]]:
    """Extract rows and row metadata from an Excel file."""
    df = pd.read_excel(xlsx_path)
    rows = []
    for idx, row in df.iterrows():
        text = row.to_string()
        meta = {"source": os.path.basename(xlsx_path), "row": int(idx), "type": "excel"}
        rows.append((text, meta))
    return rows


# -----------------------------------------------------
# Utilities: Embeddings + Index
# -----------------------------------------------------

# Simple cached resources
_text_splitter = None
_embeddings = None


def get_text_splitter(chunk_size: int = 800, overlap: int = 150):
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return _text_splitter


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def build_documents_from_files(file_paths: List[str]) -> List[Document]:
    """Process files into a list of LangChain Document objects."""
    splitter = get_text_splitter()
    docs: List[Document] = []
    for path in file_paths:
        if path.lower().endswith(".pdf"):
            items = load_pdf_pages(path)
        elif path.lower().endswith((".xls", ".xlsx")):
            items = load_excel_rows(path)
        else:
            continue
        for i, (text, meta) in enumerate(items):
            for chunk_i, piece in enumerate(splitter.split_text(text)):
                chunk_meta = meta.copy()
                chunk_meta.update({"chunk": chunk_i, "orig_index": i})
                docs.append(Document(page_content=piece, metadata=chunk_meta))
    return docs


def build_or_rebuild_chroma(docs: List[Document], persist_directory: str = "chroma_db", rebuild: bool = False):
    """Create or update a Chroma vector store from documents."""
    embeddings = get_embeddings()
    if rebuild and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        if docs:
            db.add_documents(docs)
    else:
        db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    return db


# -----------------------------------------------------
# LLM Definitions
# -----------------------------------------------------
def get_local_llm():
    """Return an Ollama-based LLM if available, otherwise None."""
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model="mistral", streaming=True, callbacks=[StreamlitCallbackHandler()])
    except Exception:
        try:
            from langchain_community.llms import Ollama
            return Ollama(model="mistral")
        except Exception:
            return None


def get_openai_llm():
    """Return an OpenAI Chat model configured from environment variable."""
    from langchain.chat_models import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        streaming=True,
        callbacks=[StreamlitCallbackHandler()],
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


# -----------------------------------------------------
# FastAPI JSON Backend (replaces previous Streamlit UI)
# -----------------------------------------------------

app = FastAPI(title="Document Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state
_db_lock = threading.Lock()
_db = None
_retriever = None
_history: List[dict] = []
_uploaded_paths: List[str] = []


@app.post("/upload")
async def api_upload(files: List[UploadFile] = File(...)):
    """Upload files and save them to the uploads/ folder."""
    saved = []
    for f in files:
        out_path = save_uploaded_file(f, dest_folder="uploads")
        saved.append(out_path)
        _uploaded_paths.append(out_path)
    return {"saved": saved}


@app.post("/index")
async def api_index(rebuild: bool = Form(False)):
    """Build or rebuild the Chroma index from uploaded files."""
    global _db, _retriever
    with _db_lock:
        file_paths = list(dict.fromkeys(_uploaded_paths))
        docs = build_documents_from_files(file_paths)
        _db = build_or_rebuild_chroma(docs, persist_directory="chroma_db", rebuild=rebuild)
        _retriever = _db.as_retriever(search_kwargs={"k": 4}) if _db else None
    return {"indexed_files": file_paths, "doc_count": len(docs)}


@app.post("/query")
async def api_query(
    question: str = Form(...),
    backend: str = Form("openai"),
    knowledge_mode: str = Form("files_only"),
    k: int = Form(4),
):
    """Query the indexed documents using the chosen backend."""
    global _retriever
    if not _retriever:
        return JSONResponse({"error": "No index available. Upload and index documents first."}, status_code=400)

    llm = get_openai_llm() if "openai" in backend.lower() else get_local_llm()
    if not llm:
        return JSONResponse({"error": "Requested LLM backend not available."}, status_code=400)

    try:
        retrieved_docs = _retriever.get_relevant_documents(question)
        if not retrieved_docs:
            return {"answer": "I couldn't find any relevant information in the uploaded documents.", "sources": []}

        context_text = "\\n\\n".join([doc.page_content for doc in retrieved_docs])
        if "files_only" in knowledge_mode:
            prompt = f"""Answer strictly using only the provided context. If not found, reply that it's not in the documents.\\n\\nQuestion: {question}\\n\\nContext:\\n{context_text}"""
        else:
            prompt = f"""Use your general knowledge and the provided context, prioritizing document content.\\n\\nQuestion: {question}\\n\\nContext:\\n{context_text}"""

        try:
            if hasattr(llm, "invoke"):
                result = llm.invoke(prompt)
                final_response = getattr(result, "content", str(result))
            elif hasattr(llm, "predict"):
                final_response = llm.predict(prompt)
            else:
                final_response = str(llm(prompt))
        except Exception:
            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=_retriever, return_source_documents=True
            )
            out = qa({"query": question})
            final_response = out.get("result") or out.get("answer") or str(out)

        citations = []
        for src in retrieved_docs:
            meta = src.metadata
            if meta.get("type") == "pdf":
                citations.append(f"{meta['source']} — page {meta.get('page')}")
            elif meta.get("type") == "excel":
                citations.append(f"{meta['source']} — row {meta.get('row')}")

        entry = {"question": question, "answer": final_response, "sources": list(dict.fromkeys(citations))}
        _history.append(entry)
        return entry

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/history")
async def api_history():
    return {"history": _history}


@app.post("/clear")
async def api_clear():
    global _history
    _history = []
    return {"cleared": True}


@app.get("/download")
async def api_download():
    """Return a simple markdown file of history. Writes a temp file and returns it."""
    md = StringIO()
    for item in _history:
        md.write(f"**You:** {item['question']}\\n\\n")
        md.write(f"**Bot:** {item['answer']}\\n\\n")
        if item["sources"]:
            md.write("Sources:\\n")
            for s in item["sources"]:
                md.write(f"- {s}\\n")
            md.write("\\n")
    tmp_path = "chat_history.md"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(md.getvalue())
    return FileResponse(tmp_path, media_type="text/markdown", filename="chat_history.md")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

