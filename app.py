"""Document Chat Application (app.py)

A Streamlit-based chat application for conversations with PDF and Excel documents
using local LLMs (via Ollama) or OpenAI's GPT models.

Components:
    1. Document Processing:
        - PDF text extraction (page-wise)
        - Excel row processing
        - Text chunking with overlap
    
    2. Vector Storage:
        - Local Chroma DB for document embeddings
        - HuggingFace sentence-transformers
        - Persistent storage with rebuild option

    3. LLM Integration:
        - Local: Ollama (default: mistral model)
        - Cloud: OpenAI GPT (requires API key)
        - Real-time response streaming

    4. Chat Features:
        - Two knowledge modes:
            * Files Only: Strict responses from documents
            * Files + Model: Blend of document and model knowledge
        - Source citations with file/page references
        - Downloadable chat history
        - Clear chat option

Configuration:
    Environment Variables:
        OPENAI_API_KEY: Required for OpenAI GPT mode
        OPENAI_MODEL: Optional, defaults to gpt-4o-mini

    File Storage:
        uploads/: Temporary document storage
        chroma_db/: Persistent vector store

Dependencies:
    Core:
        streamlit: Web UI framework
        langchain: LLM/embedding orchestration
        chromadb: Vector store
        sentence-transformers: Document embeddings

    Document Processing:
        pypdf: PDF text extraction
        pandas: Excel file handling

Author: Ayan Dutta
Version: 1.0.0
— Hybrid PDF/Excel Chatbot (Streaming + Dual Mode + Sources)
====================================================================
Features:
- Local LLM (Ollama / Mistral)
- OpenAI GPT (gpt-4o-mini)
- Strict “Files Only” or blended “Files + Model Knowledge” modes
- Streaming responses (typing effect)
- Expandable “Show Sources” citations
- Downloadable and clearable chat history
"""

# -----------------------------------------------------
# Imports
# -----------------------------------------------------
import os, shutil, time
from io import StringIO
from typing import List, Tuple
import pandas as pd
from pypdf import PdfReader
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# -----------------------------------------------------
# Setup
# -----------------------------------------------------
load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain.callbacks.base import BaseCallbackHandler


# -----------------------------------------------------
# Streamlit Streaming Callback Handler
# -----------------------------------------------------
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM tokens to Streamlit.

    Enables real-time "typing" effect in the chat interface by updating
    a Streamlit container with each token as it's generated.

    Attributes:
        container: A Streamlit empty container for token display
        text (str): Accumulated text from received tokens

    Example:
        >>> handler = StreamlitCallbackHandler()
        >>> llm = ChatOpenAI(streaming=True, callbacks=[handler])
    """
    def __init__(self):
        self.container = st.empty()
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, response, **kwargs):
        self.container.markdown(self.text)


# -----------------------------------------------------
# Utilities: File Handling
# -----------------------------------------------------
def save_uploaded_file(uploaded_file, dest_folder="uploads") -> str:
    """Save uploaded Streamlit file to local filesystem.

    Args:
        uploaded_file: Streamlit UploadedFile object
        dest_folder (str): Save directory path
    
    Returns:
        str: Absolute path to saved file
    
    Example:
        >>> file = st.file_uploader("Upload file")
        >>> path = save_uploaded_file(file)
        >>> print(f"Saved to {path}")

    Note:
        Creates dest_folder if it doesn't exist
    """
    os.makedirs(dest_folder, exist_ok=True)
    out_path = os.path.join(dest_folder, uploaded_file.name)
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def load_pdf_pages(pdf_path: str) -> List[Tuple[str, dict]]:
    """Extract text and metadata from PDF pages.

    Args:
        pdf_path (str): Path to PDF file

    Returns:
        List[Tuple[str, dict]]: List of (text, metadata) tuples 
            where metadata contains source filename, page number,
            and document type

    Example:
        >>> pages = load_pdf_pages("doc.pdf")
        >>> text, meta = pages[0]
        >>> print(f"Page {meta['page']}: {text[:50]}...")
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
    """Extract rows and metadata from Excel files.

    Args:
        xlsx_path (str): Path to Excel file

    Returns:
        List[Tuple[str, dict]]: List of (text, metadata) tuples where 
            metadata has source filename, row number, and type

    Example:
        >>> rows = load_excel_rows("data.xlsx")
        >>> text, meta = rows[0]
        >>> print(f"Row {meta['row']}: {text[:50]}...")
    """
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
@st.cache_resource
def get_text_splitter(chunk_size: int = 800, overlap: int = 150):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_documents_from_files(file_paths: List[str]) -> List[Document]:
    """
    Process multiple files and convert them into LangChain Document objects.
    
    This function:
    1. Takes a list of file paths (PDFs/Excel)
    2. Extracts text content appropriately for each file type
    3. Splits text into chunks with overlap
    4. Creates Document objects with metadata
    
    Args:
        file_paths: List of absolute paths to PDF/Excel files
        
    Returns:
        List[Document]: List of LangChain Document objects with:
            - page_content: Text chunk
            - metadata: Source file, page/row, chunk number
            
    The metadata helps track where each piece of text came from,
    enabling source citations in chat responses.
    """
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
    """Initialize or update a persistent Chroma vector store.

    Creates a new DB if none exists, loads existing DB if present,
    optionally rebuilds from scratch, and adds new documents to DB.
    The vector store persists to disk for reuse across sessions.

    Args:
        docs (List[Document]): LangChain Document objects to index
        persist_directory (str, optional): DB storage location. Defaults to "chroma_db"
        rebuild (bool, optional): If True, recreate DB from scratch. Defaults to False

    Returns:
        Chroma: Initialized vector store with documents indexed

    Note:
        Newer Chroma versions handle persistence automatically.
    """
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
    """Initialize a local LLM using Ollama with streaming support.

    Attempts to load Ollama model in order:
    1. langchain_ollama.OllamaLLM (preferred)
    2. langchain_community.llms.Ollama (fallback)
    3. None if Ollama unavailable

    Returns:
        Union[OllamaLLM, None]: Configured LLM instance or None

    Example:
        >>> llm = get_local_llm()
        >>> if llm is None:
        ...     st.error("Please install Ollama")

    Note:
        Requires Ollama installed and model pulled:
        `ollama pull mistral`
    """
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
    """Initialize an OpenAI GPT model with streaming support.
    
    Creates a ChatOpenAI instance with:
    - Real-time token streaming
    - Temperature = 0.7 (balanced creativity)
    - gpt-4o-mini model by default

    Returns:
        ChatOpenAI: Configured GPT chat model

    Raises:
        ValueError: If OPENAI_API_KEY not set

    Example:
        >>> llm = get_openai_llm()
        >>> response = llm.invoke("Tell me a joke")
        
    Note:
        Set environment variable OPENAI_API_KEY first
    """
    from langchain.chat_models import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=True,
        callbacks=[StreamlitCallbackHandler()],
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


# -----------------------------------------------------
# Streamlit Chat UI
# -----------------------------------------------------
st.set_page_config(page_title="Chat with Your Documents", layout="wide")
st.title("💬 Offline Multi-File Chatbot (Streaming + Sources)")

# --- Backend & Knowledge Mode Selection ---
mode = st.radio(
    "Select AI Backend:",
    ["Local (Ollama - Offline)", "OpenAI GPT (Online)"],
    index=0
)
knowledge_mode = st.radio(
    "Knowledge Mode:",
    ["Files Only 🗂️", "Files + Model Knowledge 🧠"],
    index=0,
    help="Files Only → answers strictly from uploaded files.\nFiles + Model Knowledge → combines files with model’s own knowledge."
)

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload PDF/Excel files",
    accept_multiple_files=True,
    type=["pdf", "xlsx"]
)
rebuild_index = st.checkbox("Rebuild index (delete old and re-create)", value=False)

# --- Main Workflow ---
if uploaded_files:
    file_paths = [save_uploaded_file(f, "uploads") for f in uploaded_files]
    docs = build_documents_from_files(file_paths)

    with st.spinner("📦 Indexing your documents..."):
        db = build_or_rebuild_chroma(docs, persist_directory="chroma_db", rebuild=rebuild_index)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = get_openai_llm() if "GPT" in mode else get_local_llm()
    if not llm:
        st.error("❌ Model not available. Please check Ollama or OpenAI setup.")
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )

        # --- Session State ---
        if "history" not in st.session_state:
            st.session_state.history = []
        if "is_generating" not in st.session_state:
            st.session_state.is_generating = False

        st.markdown("### 💬 Chat History")
        for role, content in st.session_state.history:
            with st.chat_message("user" if role == "You" else "assistant"):
                if role == "You":
                    st.markdown(content)
                else:
                    st.markdown(content["response"])
                    if content["sources"]:
                        with st.expander("📚 Show Sources"):
                            st.markdown("\n".join(content["sources"]))

        user_input = st.chat_input("Ask a question about your documents...")

        # -------------------------------------------------
        # Chat Handling
        # -------------------------------------------------
        if user_input and not st.session_state.is_generating:
            st.session_state.is_generating = True
            st.session_state.history.append(("You", user_input))

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                chat_placeholder = st.empty()
                partial_response = ""
                chat_placeholder.markdown("_🤔 Thinking..._")

                try:
                    retrieved_docs = retriever.get_relevant_documents(user_input)
                    if not retrieved_docs:
                        final_response = "❌ I couldn’t find any relevant information in your uploaded documents."
                        chat_placeholder.markdown(final_response)
                        st.session_state.history.append(("Bot", final_response))
                    else:
                        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

                        # Dynamic prompt
                        if "Files Only" in knowledge_mode:
                            prompt = f"""
Answer the following question **strictly using only the provided context**.
If not found, say:
"I couldn’t find that information in the uploaded documents."

Question: {user_input}

Context:
{context_text}
"""
                        else:
                            prompt = f"""
Use both your general knowledge and the provided document context.
Prioritize the uploaded file information for accuracy.

Question: {user_input}

Context:
{context_text}
"""

                        # --- Streaming output ---
                        if hasattr(llm, "stream"):
                            update_interval = 0.1
                            last_update_time = time.time()
                            for chunk in llm.stream(prompt):
                                token = ""
                                if hasattr(chunk, "content"):
                                    token = chunk.content or ""
                                elif isinstance(chunk, str):
                                    token = chunk
                                elif hasattr(chunk, "text"):
                                    token = chunk.text or ""
                                else:
                                    token = str(chunk)
                                partial_response += token
                                if time.time() - last_update_time > update_interval:
                                    chat_placeholder.markdown(partial_response + "▌")
                                    last_update_time = time.time()
                            chat_placeholder.markdown(partial_response)
                            final_response = partial_response
                        else:
                            result = llm.invoke(prompt)
                            final_response = getattr(result, "content", str(result))
                            chat_placeholder.markdown(final_response)

                        # --- Show Sources ---
                        citations = []
                        for src in retrieved_docs:
                            meta = src.metadata
                            if meta.get("type") == "pdf":
                                citations.append(f"📘 **{meta['source']}** — page {meta.get('page')}")
                            elif meta.get("type") == "excel":
                                citations.append(f"📊 **{meta['source']}** — row {meta.get('row')}")

                        # Store both response and citations in history
                        st.session_state.history.append(("Bot", {
                            "response": final_response,
                            "sources": list(set(citations)) if citations else []
                        }))

                except Exception as e:
                    chat_placeholder.markdown(f"_⚠️ Error generating response: {e}_")

            st.session_state.is_generating = False
            st.rerun()

        # --- Control Buttons ---
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Chat"):
                st.session_state.history = []
                st.rerun()
        with col2:
            if st.button("💾 Download Chat History"):
                chat_md = StringIO()
                for role, content in st.session_state.history:
                    if role == "You":
                        chat_md.write(f"**{role}:** {content}\n\n")
                    else:
                        chat_md.write(f"**{role}:** {content['response']}\n")
                        if content['sources']:
                            chat_md.write("\nSources:\n")
                            for source in content['sources']:
                                chat_md.write(f"- {source}\n")
                        chat_md.write("\n")
                st.download_button(
                    label="⬇️ Save Chat as Markdown",
                    data=chat_md.getvalue(),
                    file_name="chat_history.md",
                    mime="text/markdown"
                )
