# app.py
"""
Offline Multi-file Chatbot with PDF/Excel Support
================================================

This Streamlit app allows you to upload multiple PDF/Excel files,
index them locally with embeddings (HuggingFace + ChromaDB),
and chat with the content using a local LLM (via Ollama).

Features:
---------
- Upload multiple PDF and Excel files
- Build a local vector index using Chroma
- Ask questions to a local LLM with answers grounded in your data
- Inline citations (file + page/row) are included in answers
- "Rebuild Index" option to reset the knowledge base
- Loader while generating answers
- Chat input auto-clears after sending
"""

import os
import shutil
import streamlit as st
from typing import List, Tuple
from pypdf import PdfReader
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


# -------------------------------
# Local LLM loader (Ollama)
# -------------------------------
def get_local_llm():
    """
    Returns a local LLM instance from Ollama.
    Make sure Ollama is installed and the model (e.g., mistral) is pulled.
    """
    try:
        from langchain.llms import Ollama
        return Ollama(model="mistral")  # You can change "mistral" to another local model
    except Exception:
        return None


# -------------------------------
# File handling utilities
# -------------------------------
def save_uploaded_file(uploaded_file, dest_folder="uploads") -> str:
    """
    Save an uploaded file from Streamlit uploader to the local filesystem.
    Returns the path of the saved file.
    """
    os.makedirs(dest_folder, exist_ok=True)
    out_path = os.path.join(dest_folder, uploaded_file.name)
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def load_pdf_pages(pdf_path: str) -> List[Tuple[str, dict]]:
    """
    Extract text page by page from a PDF.
    Returns a list of (text, metadata) tuples.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            meta = {"source": os.path.basename(pdf_path), "page": i, "type": "pdf"}
            pages.append((text, meta))
    return pages


def load_excel_rows(xlsx_path: str) -> List[Tuple[str, dict]]:
    """
    Extract row-wise text from an Excel file.
    Each row is converted into a string and stored with metadata.
    """
    df = pd.read_excel(xlsx_path)
    rows = []
    for idx, row in df.iterrows():
        text = row.to_string()
        meta = {"source": os.path.basename(xlsx_path), "row": int(idx), "type": "excel"}
        rows.append((text, meta))
    return rows


# -------------------------------
# Indexing utilities
# -------------------------------
@st.cache_resource
def get_text_splitter(chunk_size: int = 800, overlap: int = 150):
    """Return a text splitter to break large text into smaller chunks."""
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)


@st.cache_resource
def get_embeddings():
    """Return a HuggingFace embedding model for vectorization."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_documents_from_files(file_paths: List[str]) -> List[Document]:
    """
    Read multiple files (PDF/Excel) and split their contents into chunks.
    Returns a list of LangChain Document objects with metadata.
    """
    splitter = get_text_splitter()
    documents: List[Document] = []

    for path in file_paths:
        if path.lower().endswith(".pdf"):
            items = load_pdf_pages(path)
        elif path.lower().endswith((".xls", ".xlsx")):
            items = load_excel_rows(path)
        else:
            continue

        for item_index, (text, meta) in enumerate(items):
            pieces = splitter.split_text(text)
            for chunk_i, piece in enumerate(pieces):
                chunk_meta = meta.copy()
                chunk_meta.update({"chunk": chunk_i, "orig_index": item_index})
                documents.append(Document(page_content=piece, metadata=chunk_meta))

    return documents


def build_or_rebuild_chroma(documents: List[Document], persist_directory: str = "chroma_db", rebuild: bool = False):
    """
    Build or rebuild a Chroma vectorstore index.
    - If rebuild=True, the old index is cleared and recreated.
    - Documents are added and persisted for future use.
    """
    embeddings = get_embeddings()

    if rebuild:
        # Close and clear old DB before deleting
        try:
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            db.delete_collection()
        except Exception:
            pass

        if os.path.exists(persist_directory):
            try:
                shutil.rmtree(persist_directory)
            except PermissionError:
                import time
                time.sleep(1)
                try:
                    shutil.rmtree(persist_directory)
                except Exception as e:
                    print("Warning: could not delete chroma_db fully:", e)

    # Load or create DB
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        if documents:
            db.add_documents(documents)
            db.persist()
    else:
        if documents:
            db = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
            db.persist()
        else:
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    return db


# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Offline Multi-file Chatbot", layout="wide")
st.title("📄 Offline Multi-file Chatbot with Sources")

# File upload
uploaded_files = st.file_uploader(
    "Upload PDF/Excel files",
    accept_multiple_files=True,
    type=["pdf", "xlsx"],
    key="files"
)

# Option to rebuild the index
rebuild_index = st.checkbox("Rebuild index (delete old and re-create)", value=False)

if uploaded_files:
    # Save uploaded files locally
    file_paths = [save_uploaded_file(f, "uploads") for f in uploaded_files]

    # Build docs and index
    documents = build_documents_from_files(file_paths)
    with st.spinner("Indexing your documents..."):
        db = build_or_rebuild_chroma(documents, persist_directory="chroma_db", rebuild=rebuild_index)

    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = get_local_llm()

    if llm is None:
        st.error("Local LLM not available. Install Ollama and pull a model (e.g. `ollama pull mistral`).")
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

        # Chat input
        query = st.text_input("Ask a question:", value=st.session_state.query, key="chat_input")

        # Send button
        if st.button("Send"):
            if query.strip():
                with st.spinner("🤔 Thinking... Generating answer..."):
                    result = qa(query)
                    answer = result["result"]
                    sources = result.get("source_documents", [])

                    # Collect citations
                    citations = []
                    for src in sources:
                        meta = src.metadata
                        if meta.get("type") == "pdf":
                            citations.append(f"{meta['source']} (page {meta.get('page')})")
                        elif meta.get("type") == "excel":
                            citations.append(f"{meta['source']} (row {meta.get('row')})")

                    citation_text = "\n\n**Sources:** " + ", ".join(set(citations)) if citations else ""
                    final_answer = answer + citation_text

                    # Save history
                    st.session_state.history.append(("You", query))
                    st.session_state.history.append(("Bot", final_answer))

                # Clear input box after sending
                st.session_state.query = ""
                st.rerun()

        # Show conversation history
        if st.session_state.history:
            st.markdown("---")
            for role, msg in st.session_state.history[::-1]:
                st.markdown(f"**{role}:** {msg}")