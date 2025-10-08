"""
app.py
------
Hybrid PDF/Excel Chatbot (Local + GPT API)
===========================================

A Streamlit chatbot that lets you upload and chat with your PDF/Excel files.
It can work in two modes:
1. 🧠 Local Mode (Offline) — uses Ollama + Mistral model
2. ☁️ Online Mode — uses OpenAI GPT API (gpt-4o / gpt-4o-mini)

Features:
---------
- Upload multiple PDF/Excel files
- Local Chroma-based vector indexing
- HuggingFace embeddings (sentence-transformers)
- Retrieval-Augmented QA
- Mode toggle between local and OpenAI GPT
- Inline citations and chat history
"""

import os
import shutil
import streamlit as st
from typing import List, Tuple
from pypdf import PdfReader
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# -------------------------------
# Environment Setup
# -------------------------------
load_dotenv()

# -------------------------------
# Conditional Imports
# -------------------------------
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

# -------------------------------
# Utility: File Handling
# -------------------------------
def save_uploaded_file(uploaded_file, dest_folder="uploads") -> str:
    os.makedirs(dest_folder, exist_ok=True)
    out_path = os.path.join(dest_folder, uploaded_file.name)
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def load_pdf_pages(pdf_path: str) -> List[Tuple[str, dict]]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            meta = {"source": os.path.basename(pdf_path), "page": i, "type": "pdf"}
            pages.append((text, meta))
    return pages


def load_excel_rows(xlsx_path: str) -> List[Tuple[str, dict]]:
    df = pd.read_excel(xlsx_path)
    rows = []
    for idx, row in df.iterrows():
        text = row.to_string()
        meta = {"source": os.path.basename(xlsx_path), "row": int(idx), "type": "excel"}
        rows.append((text, meta))
    return rows


# -------------------------------
# Utility: Text Splitting + Embeddings
# -------------------------------
@st.cache_resource
def get_text_splitter(chunk_size: int = 800, overlap: int = 150):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


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
            for chunk_i, piece in enumerate(splitter.split_text(text)):
                chunk_meta = meta.copy()
                chunk_meta.update({"chunk": chunk_i, "orig_index": i})
                docs.append(Document(page_content=piece, metadata=chunk_meta))
    return docs


def build_or_rebuild_chroma(docs: List[Document], persist_directory="chroma_db", rebuild=False):
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


# -------------------------------
# Local LLM (Ollama)
# -------------------------------
def get_local_llm():
    """Return a locally hosted model from Ollama (Mistral)."""
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model="mistral")
    except Exception:
        try:
            from langchain_community.llms import Ollama
            return Ollama(model="mistral")
        except Exception:
            return None


# -------------------------------
# OpenAI GPT API
# -------------------------------
def get_openai_llm():
    """Return an OpenAI GPT model using API key."""
    from langchain.chat_models import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Hybrid PDF/Excel Chatbot", layout="wide")
st.title("📄 Chat with Your Documents (Local / GPT)")

# Select model type (flag)
mode = st.radio(
    "Select AI Mode:",
    ["Local (Ollama - Offline)", "OpenAI GPT (Online)"],
    index=1,
    help="Choose whether to use your local Ollama model or OpenAI GPT API."
)

uploaded_files = st.file_uploader(
    "Upload PDF/Excel files",
    accept_multiple_files=True,
    type=["pdf", "xlsx"],
    key="files"
)

rebuild_index = st.checkbox("Rebuild index (delete old and re-create)", value=False)

if uploaded_files:
    file_paths = [save_uploaded_file(f, "uploads") for f in uploaded_files]
    docs = build_documents_from_files(file_paths)
    with st.spinner("Indexing your documents..."):
        db = build_or_rebuild_chroma(docs, persist_directory="chroma_db", rebuild=rebuild_index)

    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Select model based on mode flag
    if "GPT" in mode:
        llm = get_openai_llm()
    else:
        llm = get_local_llm()

    if llm is None:
        st.error("❌ No valid model found. Make sure Ollama or OpenAI API key is configured.")
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        if "history" not in st.session_state:
            st.session_state.history = []
        if "query" not in st.session_state:
            st.session_state.query = ""

        query = st.text_input("Ask a question:", value=st.session_state.query, key="chat_input")

        if st.button("Send"):
            if query.strip():
                with st.spinner("🤔 Thinking... Generating answer..."):
                    result = qa.invoke({"query": query})
                    answer = result.get("result", "")
                    sources = result.get("source_documents", [])
                    citations = []
                    for src in sources:
                        meta = src.metadata
                        if meta.get("type") == "pdf":
                            citations.append(f"{meta['source']} (page {meta.get('page')})")
                        elif meta.get("type") == "excel":
                            citations.append(f"{meta['source']} (row {meta.get('row')})")
                    citation_text = "\n\n**Sources:** " + ", ".join(set(citations)) if citations else ""
                    final_answer = answer + citation_text

                    st.session_state.history.append(("You", query))
                    st.session_state.history.append(("Bot", final_answer))

                st.session_state.query = ""
                st.rerun()

        if st.session_state.history:
            st.markdown("---")
            for role, msg in st.session_state.history[::-1]:
                st.markdown(f"**{role}:** {msg}")