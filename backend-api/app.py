import os
import shutil
import threading
from io import StringIO
from typing import List, Tuple

import pandas as pd
from pypdf import PdfReader
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain.callbacks.base import BaseCallbackHandler
import uvicorn

load_dotenv()


# -------------------------------------------------------------------
# CALLBACK HANDLER FOR STREAMING TOKEN OUTPUT
# -------------------------------------------------------------------
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token

    def on_llm_end(self, response, **kwargs):
        pass


# -------------------------------------------------------------------
# FILE SAVING
# -------------------------------------------------------------------
def save_uploaded_file(uploaded_file: UploadFile, dest_folder: str = "uploads") -> str:
    os.makedirs(dest_folder, exist_ok=True)
    out_path = os.path.join(dest_folder, uploaded_file.filename)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(uploaded_file.file, f)
    return out_path


# -------------------------------------------------------------------
# PDF LOADING
# -------------------------------------------------------------------
def load_pdf_pages(pdf_path: str) -> List[Tuple[str, dict]]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            meta = {
                "source": os.path.basename(pdf_path),
                "page": i,
                "type": "pdf"
            }
            pages.append((text, meta))
    return pages


# -------------------------------------------------------------------
# EXCEL LOADING
# -------------------------------------------------------------------
def load_excel_rows(xlsx_path: str) -> List[Tuple[str, dict]]:
    df = pd.read_excel(xlsx_path)
    rows = []
    for idx, row in df.iterrows():
        text = row.to_string()
        meta = {
            "source": os.path.basename(xlsx_path),
            "row": int(idx),
            "type": "excel"
        }
        rows.append((text, meta))
    return rows


# -------------------------------------------------------------------
# TEXT SPLITTER + EMBEDDINGS (lazy load)
# -------------------------------------------------------------------
_text_splitter = None
_embeddings = None


def get_text_splitter(chunk_size: int = 800, overlap: int = 150):
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
    return _text_splitter


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


# -------------------------------------------------------------------
# DOCUMENT BUILDING
# -------------------------------------------------------------------
def build_documents_from_files(file_paths: List[str]) -> List[Document]:
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
            chunks = splitter.split_text(text)
            for chunk_i, piece in enumerate(chunks):
                new_meta = meta.copy()
                new_meta["chunk"] = chunk_i
                docs.append(Document(page_content=piece, metadata=new_meta))

    return docs


# -------------------------------------------------------------------
# CHROMA MANAGEMENT
# -------------------------------------------------------------------
def build_or_rebuild_chroma(docs: List[Document], persist_directory: str = "chroma_db", rebuild=False):
    embeddings = get_embeddings()

    if rebuild and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        if docs:
            db.add_documents(docs)
    else:
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_directory
        )
    return db


# -------------------------------------------------------------------
# LLM PROVIDERS
# -------------------------------------------------------------------
def get_openai_llm():
    from langchain.chat_models import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.5,
        streaming=True,
        callbacks=[StreamlitCallbackHandler()],
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


def get_local_llm():
    try:
        from langchain_community.llms import Ollama

        return Ollama(
            model="mistral",
            callbacks=[StreamlitCallbackHandler()]
        )
    except Exception as e:
        print("Ollama load failed:", e)
        return None


# -------------------------------------------------------------------
# FASTAPI CONFIG
# -------------------------------------------------------------------
app = FastAPI(title="Document Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

_db = None
_retriever = None
_db_lock = threading.Lock()
_history: List[dict] = []
_uploaded_paths: List[str] = []


# -------------------------------------------------------------------
# UPLOAD
# -------------------------------------------------------------------
@app.post("/upload")
async def api_upload(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        p = save_uploaded_file(f)
        _uploaded_paths.append(p)
        saved.append(p)
    return {"saved": saved}


# -------------------------------------------------------------------
# INDEX DOCUMENTS
# -------------------------------------------------------------------
@app.post("/index")
async def api_index(rebuild: bool = Form(False)):
    global _db, _retriever

    with _db_lock:
        docs = build_documents_from_files(_uploaded_paths)
        _db = build_or_rebuild_chroma(docs, "chroma_db", rebuild=rebuild)
        _retriever = _db.as_retriever(search_kwargs={"k": 4})

    return {"indexed": True, "doc_count": len(docs)}


# -------------------------------------------------------------------
# QUERY
# -------------------------------------------------------------------
@app.post("/query")
async def api_query(
    question: str = Form(...),
    backend: str = Form("openai"),
    knowledge_mode: str = Form("files_only"),
    k: int = Form(4)
):
    global _retriever

    if not _retriever:
        return JSONResponse({"error": "No index available."}, status_code=400)

    llm = get_openai_llm() if backend == "openai" else get_local_llm()

    if not llm:
        return JSONResponse({"error": "LLM backend not available."}, status_code=400)

    docs = _retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    if knowledge_mode == "files_only":
        prompt = f"""Answer using only this context:

Question: {question}

Context:
{context}
"""
    else:
        prompt = f"""Use your knowledge + the context.

Question: {question}

Context:
{context}
"""

    try:
        if hasattr(llm, "invoke"):
            result = llm.invoke(prompt)
            answer = getattr(result, "content", str(result))
        else:
            answer = llm(prompt)
    except:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_retriever
        )
        result = qa({"query": question})
        answer = result.get("result", str(result))

    sources = []
    for doc in docs:
        meta = doc.metadata
        if meta["type"] == "pdf":
            sources.append(f"{meta['source']} (page {meta['page']})")
        else:
            sources.append(f"{meta['source']} (row {meta['row']})")

    record = {
        "question": question,
        "answer": answer,
        "sources": list(dict.fromkeys(sources))
    }

    _history.append(record)
    return record


# -------------------------------------------------------------------
# HISTORY
# -------------------------------------------------------------------
@app.get("/history")
async def api_history():
    return {"history": _history}


# -------------------------------------------------------------------
# CLEAR HISTORY
# -------------------------------------------------------------------
@app.post("/clear")
async def api_clear():
    _history.clear()
    return {"cleared": True}


# -------------------------------------------------------------------
# DOWNLOAD HISTORY
# -------------------------------------------------------------------
@app.get("/download")
async def api_download():
    md = StringIO()
    for entry in _history:
        md.write(f"### You: {entry['question']}\n\n")
        md.write(f"### Bot: {entry['answer']}\n\n")
        if entry["sources"]:
            md.write("Sources:\n")
            for s in entry["sources"]:
                md.write(f"- {s}\n")
        md.write("\n---\n\n")

    path = "chat_history.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md.getvalue())

    return FileResponse(path, filename="chat_history.md")


# -------------------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
