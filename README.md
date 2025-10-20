# PDF/Excel Chatbot — FastAPI backend + Angular frontend

This repository hosts a local document chatbot that indexes PDF and Excel files and answers questions using a local or API-backed LLM. The project has been reworked from a Streamlit UI to a modern split architecture:

- Backend: FastAPI (Python) — document ingestion, indexing, embeddings, and LLM integration
- Frontend: Angular — single-page application that talks to the FastAPI JSON API

Highlights
-- Upload and index PDF / Excel files
-- Local embeddings (HuggingFace sentence-transformers) and local Chroma vector store
-- Local LLM support via Ollama (optional) and optional OpenAI support
-- Angular UI with file upload, index control and chat interface

Architecture overview

- `app.py` (backend) — FastAPI application exposing endpoints such as `/upload`, `/index`, `/query`, `/history`, `/clear`, `/download`.
- `angular-app/` (frontend) — Angular application that provides the chat UI and file-upload pages and calls the FastAPI endpoints.
- `chroma_db/` — persisted Chroma vector store folder (created/used by backend).

Prerequisites

- Python 3.8+ (3.10/3.11/3.12/3.13 should work)
- Node.js (v16+ recommended; this repo used Node v22 in development)
- npm (or pnpm/yarn)
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

- Angular `npm install` peer dependency issues (zone.js): you may see a peer-deps conflict requiring `zone.js@~0.13.0`. Resolve by either:

	1) Using legacy peer deps while installing:

```powershell
cd angular-app
npm install --legacy-peer-deps
```

	2) Updating the `zone.js` dependency in `angular-app/package.json` to `"~0.13.0"` and re-running `npm install`.


Development checklist & quick commands

- Backend (from `backend-api`): create venv, install, run

```powershell
cd backend-api
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app:app --reload
```


- Frontend: install and run

```powershell
cd angular-app
npm install
npm run start
```

Stopping servers
- Use Ctrl+C in the terminals running the FastAPI or Angular dev servers to stop them.

Project layout (high-level)

```
.
├── backend-api/
│   ├── app.py             # FastAPI backend
│   ├── requirements.txt   # Backend Python dependencies (fastapi, uvicorn, ...)
│   ├── chroma_db/         # Local Chroma DB (created/used by backend)
│   └── uploads/           # Uploads and example files for backend testing
├── angular-app/           # Angular frontend (src, package.json, etc.)
```

License

MIT