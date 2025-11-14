# PDF/Excel Chatbot — FastAPI backend + Angular frontend

This repository hosts a local document chatbot that indexes PDF and Excel files and answers questions using a local or API-backed LLM. The project has been reworked from a Streamlit UI to a modern split architecture:

- Backend: FastAPI (Python) — document ingestion, indexing, embeddings, and LLM integration
- Frontend: Angular — single-page application that talks to the FastAPI JSON API

Highlights
- ✅ Upload and index PDF / Excel files
- ✅ Local embeddings (HuggingFace sentence-transformers) and local Chroma vector store
- ✅ Local LLM support via Ollama (optional) and optional OpenAI support
- ✅ Modern Angular 20 UI with file upload, indexing, and chat interface
- ✅ FastAPI REST backend with Swagger/ReDoc documentation
- ✅ Standalone components, strict TypeScript, ES2022 target

Architecture overview

- `backend-api/app.py` (backend) — FastAPI application exposing REST endpoints such as `/upload`, `/index`, `/query`, `/history`, `/clear`, `/download`.
- `angular-app/` (frontend) — Modern Angular 20 single-page application with standalone components, TypeScript strict mode, and ES2022 target. Provides chat UI and file-upload interface that communicates with the FastAPI backend.
- `chroma_db/` — persisted Chroma vector store folder (created/used by backend).
- `uploads/` — sample PDFs and Excel files for testing the backend.

Backend architecture (detailed)

The backend implements a standard retrieval-augmented generation (RAG) pipeline using the following components:

- LangChain (or LangChain-style orchestration)
	- Responsible for wiring together document loaders, text splitters, embedding calls, vector store operations, retrievers and the final QA chain.
	- We use LangChain abstractions (Document, TextSplitter, Embeddings, VectorStoreRetriever, Chains) as the conceptual flow; the code in `app.py` follows the same steps but may use direct LangChain imports or helper utilities.

- Document ingestion and splitting
	- Document loaders extract raw text from PDFs (page-by-page) and Excel files (row-by-row). Each extracted unit is turned into a LangChain `Document` with metadata fields like `source` (filename), `page` or `row`, and `chunk_id`.
	- A TextSplitter (for example RecursiveCharacterTextSplitter) breaks long text into chunks with a configurable `chunk_size` (e.g. 1000 characters) and `chunk_overlap` (e.g. 200) to preserve context across chunk boundaries.

- Embeddings (HuggingFace sentence-transformers)
	- The backend uses HuggingFace embeddings (for example `sentence-transformers/all-MiniLM-L6-v2`) to convert text chunks into dense vectors.
	- Embeddings are produced in batches for throughput. If you have a GPU or ONNX runtime available, performance improves significantly; the README `requirements` include `sentence-transformers`, `transformers`, and `onnxruntime` for that reason.
	- Embedding objects include metadata; store the text chunk, source, and any original offsets so answers can cite the source location.

- Vector store — Chroma DB
	- Chroma stores the dense vectors and associated metadata. The project persists the Chroma DB under `backend-api/chroma_db` (or `chroma_db/`) so indexes survive restarts.
	- Typical operations:
		- create / open collection (name derived from dataset or a default name)
		- add vectors + metadata (on upload / index)
		- query vector search (k nearest neighbors) via a retriever
		- optionally delete/clear the collection when doing a full rebuild
	- Keep in mind Chroma may change APIs between versions — check `chromadb` docs if you upgrade.

- Retriever and RAG (LangChain retrieval QA)
	- A VectorStoreRetriever wraps the Chroma collection and returns the top-k most relevant chunks for a query.
	- The RAG flow is:
		1. Receive a user question via `/query` endpoint
		2. Use the same embedding model to embed the question
		3. Ask the retriever for top-k chunks (configurable, e.g. k=4)
		4. Build a prompt that includes the retrieved context and the user question
		5. Send the prompt to the LLM chain (local Ollama or remote OpenAI) and return the answer
	- LangChain's `RetrievalQA` (or a custom prompt + LLM chain) is a common pattern used here.

- LLM model options
	- Ollama (local) — the code supports an offline model via Ollama. Advantages: runs locally, lower latency for small deployments, and no external API cost. Limitations: model size, system resources, and local model availability.
	- OpenAI (optional) — pass `OPENAI_API_KEY` via environment variable to use OpenAI models (e.g. gpt-4/3.5). Pros: highest-quality responses for some prompts; cons: cost and network requirement.
	- The backend chooses the model based on configuration or environment variables and constructs the LLM call accordingly. Responses are combined with the retrieved context; the implementation can format the assistant output to include inline citations (source filename + page/row) when available.

- Data shapes and metadata
	- Document chunk (stored in Chroma) typically contains:
		- id / chunk_id
		- text (the chunk)
		- metadata: { source: filename, page: number | null, row: number | null, chunk_index }
	- Query flow returns: { answer: string, source_documents: [{ source, page|row, text_excerpt }] }

- Endpoints (behavior)
	- POST /upload — accepts file uploads (PDF/Excel). Extracts, splits, embeds, and stores vectors. Returns status and any warnings.
	- POST /index — (re)build the index from `uploads/` or an uploaded set of files. Useful when you want a full rebuild.
	- POST /query — accepts JSON { question: string } and returns an aggregated answer plus source citations.
	- GET /history — returns local chat history (if the backend stores conversations)
	- POST /clear — clears chat history and optionally the vector store
	- GET /download — downloads the chat history as a text file

Operational considerations

- Performance
	- Batch embedding calls when indexing many chunks.
	- Use `onnxruntime` or an accelerated transformer runtime if embedding on CPU is slow; on GPU, transformers with CUDA deliver the best throughput.
	- Tune `chunk_size`, `chunk_overlap` and `top_k` for quality vs latency trade-offs.

- Storage and persistence
	- Chroma collections are persisted to disk. When rebuilding the index, the service can delete the collection folder and re-create it.
	- Keep backups of `chroma_db` if the corpus is expensive to reprocess.

- Concurrency and safety
	- File upload and indexing may be long-running; the API uses background tasks or synchronous calls depending on the implementation. Consider making indexing asynchronous for large corpora.
	- Sanitize and validate uploaded files. Avoid unbounded memory growth by limiting file sizes or chunk counts.

- Security
	- If using OpenAI, set `OPENAI_API_KEY` as an environment variable and do not commit it.
	- If exposing the FastAPI server externally, add authentication and CORS rules to the API.

- Troubleshooting
	- "Form data requires \"python-multipart\" to be installed" → `pip install python-multipart`
	- Missing `fastapi` / `uvicorn` → ensure the virtualenv is active and run `pip install -r backend-api/requirements.txt`
	- Chroma errors after upgrading versions → check the `chromadb` changelog and migration notes

This section documents the main backend responsibilities. The frontend (`angular-app/`) is a thin client that uploads files, triggers indexing and sends queries to the backend; see the `angular-app/src/app/services/api.ts` service for the exact request shapes used by the UI.

Frontend technologies (Angular 20)

The frontend is built with modern Angular 20 featuring:

- **Standalone Components** — No NgModules; components bootstrap directly with `bootstrapApplication()`
- **Strict TypeScript** — Full type checking enabled in `tsconfig.json` for safety and IDE support
- **ES2022 Target** — Modern JavaScript syntax with async/await, nullish coalescing, optional chaining, etc.
- **Reactive Forms** — Uses Angular Forms module (`FormsModule`) for two-way data binding
- **HttpClient** — Direct HTTP communication to FastAPI backend with `provideHttpClient()`
- **RxJS 7.8+** — Reactive streams for handling async operations
- **Bootstrap 5.3** — Responsive CSS framework for styling
- **Common Utilities** — `CommonModule` for `*ngIf`, `*ngFor`, etc.

The app structure is minimal:
- Single `AppComponent` (standalone) handles file uploads, document indexing, and chat queries
- `ChatService` wraps API calls to the backend
- HTML template with Bootstrap styling for responsive UI
- Clean separation of concerns: UI → Service → Backend API

See `angular-app/README.md` for frontend-specific setup and development notes.

Prerequisites

- Python 3.8+ (3.10/3.11/3.12/3.13 should work)
- Node.js 18.13.0+ (v20+ recommended; this repo used v22 in development)
- npm v9+ (or pnpm/yarn)
- Optional: Ollama (for local LLM)
- Optional: OpenAI API key (if you want to use OpenAI GPT)

Quick start — full local development (backend + frontend)

1. Clone the repository and open a terminal:

```powershell
git clone <repository-url>
cd <repository-name>
```


2. Backend (in `backend-api/`): create and activate a Python virtual environment, install requirements

```powershell
# switch to the backend folder
cd backend-api

# create virtualenv (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# install backend dependencies
pip install -r requirements.txt
```

Notes: `backend-api/requirements.txt` includes packages such as `fastapi`, `uvicorn`, and `python-multipart`. If you hit missing package errors, install them manually (from the `backend-api` folder or active virtualenv):

```powershell
pip install fastapi uvicorn python-multipart
```

3. Frontend: install Angular dependencies

```powershell
cd angular-app
npm install
```

If you don't have the Angular CLI globally, you can still run the dev server via the npm scripts (Angular CLI is a dev dependency in modern projects). To install the CLI globally (optional):

```powershell
npm install -g @angular/cli
```


4. Run backend and frontend (development)

Open two terminals.

Terminal A — start FastAPI backend (from `backend-api`):

```powershell
cd backend-api
.\venv\Scripts\python.exe -m uvicorn app:app --reload
# Server will be available at http://127.0.0.1:8000
```

Terminal B — start Angular dev server (from repo root):

```powershell
cd angular-app
npm run start
# or: ng serve --open
# Frontend will be available at http://localhost:4200
```

Open the Angular app in your browser. The UI calls the FastAPI endpoints at `http://localhost:8000` (this is the default base URL used by the frontend service). API docs for the backend are available at:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

Production build (optional)

If you want to produce a static build of the Angular app and serve it with another web server (or wire it into FastAPI):

```powershell
cd angular-app
npm run build
# built files are in angular-app/dist/
```

Serve the `dist/` folder using nginx, a static-file server, or copy the files into a STATIC folder served by your production backend.

Environment variables

- `OPENAI_API_KEY` — (optional) your OpenAI API key if using the OpenAI mode.

Optional: Ollama (local LLM)

Install and run Ollama as described on https://ollama.ai/. Pull a model with:

```powershell
ollama pull <model-name>
```

Troubleshooting notes

- If Python raises ModuleNotFoundError for `fastapi` or `uvicorn`, ensure you installed packages into the active virtual environment (check `which python` / path).
- FastAPI endpoints that accept form/file uploads require `python-multipart`. If you see an error like "Form data requires \"python-multipart\" to be installed", install it:

```powershell
pip install python-multipart
```

- Angular 20 `npm install` might show peer dependency warnings about `zone.js`. This is normal; zone.js 0.15 is compatible with Angular 20. Warnings can be safely ignored or suppressed with `--legacy-peer-deps` if needed.

- If the Angular dev server fails to start, ensure:
  1. You're in the `angular-app` directory
  2. Node.js version is 18.13.0 or later (`node --version`)
  3. Clear cache: `npm cache clean --force` then reinstall
  4. Check `angular.json` for valid configuration (no `defaultProject` in Angular 20+)


Development checklist & quick commands

- Backend (from `backend-api`): create venv, install, run

```powershell
cd backend-api
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app:app --reload
```

- Frontend (Angular 20): install and run

```powershell
cd angular-app
npm install
npm start
# Frontend will open at http://localhost:4200
```

- Ensure both servers are running before testing the full application.

Stopping servers
- Use Ctrl+C in the terminals running the FastAPI or Angular dev servers to stop them.

Project layout (high-level)

```
.
├── backend-api/
│   ├── app.py                 # FastAPI backend (REST endpoints)
│   ├── requirements.txt       # Python dependencies (fastapi, uvicorn, langchain, chroma, etc.)
│   ├── chroma_db/             # Persistent Chroma vector database
│   └── uploads/               # Example PDF/Excel files for testing
├── angular-app/
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.component.ts      # Main standalone component
│   │   │   ├── app.component.html    # Chat UI template
│   │   │   ├── app.component.css     # Component styles
│   │   │   └── chat.service.ts       # API service (calls backend)
│   │   ├── main.ts                   # Bootstrap entry point
│   │   ├── index.html                # HTML template
│   │   └── styles.css                # Global styles
│   ├── package.json                  # Dependencies (Angular 20, TypeScript 5.8+, Bootstrap 5)
│   ├── angular.json                  # Angular CLI configuration
│   ├── tsconfig.json                 # TypeScript configuration (ES2022, strict mode)
│   └── README.md                     # Frontend-specific documentation
├── README.md                  # This file (main documentation)

License

MIT