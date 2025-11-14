
# Document Chat Frontend (Angular 20)

Modern Angular 20 single-page application for the offline document chatbot. This frontend connects to the FastAPI backend and provides a user-friendly interface for uploading documents, indexing them, and asking questions.

## Features

- **File Upload** — Upload PDF, Excel, and other document types
- **Document Indexing** — Index uploaded files with semantic embeddings (HuggingFace)
- **Chat Interface** — Ask questions about indexed documents with source citations
- **Chat History** — View, clear, and download conversation history
- **Backend Integration** — Seamless REST API communication with FastAPI backend at `http://localhost:8000`

## Prerequisites

- **Node.js** v18.13.0+ (v20+ recommended)
- **npm** v9+ or yarn/pnpm
- **FastAPI backend** running on `http://localhost:8000` (see main README for setup)

## Setup & Installation

### 1. Install dependencies

```powershell
cd angular-app
npm install
```

**Note:** If you encounter peer dependency warnings about `zone.js`, you can safely ignore them or use:
```powershell
npm install --legacy-peer-deps
```

### 2. Run the development server

```powershell
npm start
# or: ng serve --open
```

The app will open automatically at **http://localhost:4200**

### 3. Configure backend URL (if needed)

The default backend URL is `http://localhost:8000`. If your backend is on a different URL, update:

```
src/app/services/api.ts → private baseUrl = 'http://localhost:8000';
```

## Project Structure

```
angular-app/
├── src/
│   ├── app/
│   │   ├── app.component.ts         # Main app component (standalone)
│   │   ├── app.component.html       # App template
│   │   ├── app.component.css        # App styles
│   │   └── chat.service.ts          # API service for backend communication
│   ├── main.ts                      # Bootstrap entry point
│   ├── index.html                   # HTML template
│   └── styles.css                   # Global styles
├── package.json                     # Dependencies (Angular 20, TypeScript 5.8+)
├── angular.json                     # Angular CLI config
├── tsconfig.json                    # TypeScript config (ES2022 target)
└── README.md                        # This file
```

## Building for Production

```powershell
npm run build
# or: ng build --configuration production
```

Built files are in `angular-app/dist/doc-chat-frontend/`. You can:
- Serve with a static file server (nginx, Apache, etc.)
- Wire into FastAPI by serving the `dist/` folder as static files
- Deploy to a CDN or hosting platform

## Usage Workflow

1. **Start backend server** (from `backend-api/`)
   ```powershell
   cd backend-api
   .\venv\Scripts\python.exe -m uvicorn app:app --reload
   ```

2. **Start Angular dev server** (from `angular-app/`)
   ```powershell
   npm start
   ```

3. **Upload documents** — Use the file upload form to select PDF/Excel files

4. **Index documents** — Click "Upload & Index" to process files and build the vector store

5. **Ask questions** — Type questions in the chat box and get answers with source citations

6. **Manage history** — View, download, or clear chat history

## API Integration

The frontend communicates with the backend via REST endpoints:

- `POST /upload` — Upload files
- `POST /index` — Build/rebuild the document index
- `POST /query` — Submit a question and get an answer
- `GET /history` — Fetch chat history
- `POST /clear` — Clear chat history
- `GET /download` — Download chat history as a file

See `src/app/chat.service.ts` for the API service implementation.

## Angular 20 Setup Notes

This project uses:
- **Angular 20** with standalone components (no NgModules)
- **TypeScript 5.8+** with strict mode enabled
- **RxJS 7.8+** for reactive data flow
- **zone.js 0.15** for change detection

### Standalone Components

The app uses Angular 20's standalone component API:

```typescript
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent { }
```

### Bootstrap

The app bootstraps directly without NgModule:

```typescript
bootstrapApplication(AppComponent, {
  providers: [provideHttpClient()]
})
```

## Troubleshooting

**Issue:** `npm install` fails with peer dependency errors
- **Solution:** Use `npm install --legacy-peer-deps` or update `zone.js` in `package.json`

**Issue:** Backend returns 404 errors
- **Solution:** Ensure FastAPI backend is running on `http://localhost:8000`; check CORS settings if serving from different origin

**Issue:** Changes not reflecting in browser
- **Solution:** Hard refresh (Ctrl+Shift+R) or clear browser cache

**Issue:** TypeScript compilation errors
- **Solution:** Run `npm run build` to see full error details; check `tsconfig.json` strict mode settings

## Development

- **Lint:** `ng lint` (if ESLint is configured)
- **Test:** `ng test` (if Karma/Jasmine is set up)
- **Format:** Use Prettier or your preferred code formatter

## Contributing

Pull requests welcome! Ensure:
- Code follows Angular style guide
- Components are standalone
- TypeScript strict mode compliance
- No breaking changes to API service

## License

MIT
