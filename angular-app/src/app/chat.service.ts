import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../environments/environment';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private base = environment.apiUrl;
  constructor(private http: HttpClient) {}

  uploadFiles(files: File[]): Observable<any> {
    const form = new FormData();
    for (const f of files) form.append('files', f, f.name);
    return this.http.post(`${this.base}/upload`, form);
  }

  index(rebuild = false): Observable<any> {
    const form = new FormData();
    form.append('rebuild', String(rebuild));
    return this.http.post(`${this.base}/index`, form);
  }

  query(question: string, backend = 'openai', knowledge_mode = 'files_only', k = 4): Observable<any> {
    const form = new FormData();
    form.append('question', question);
    form.append('backend', backend);
    form.append('knowledge_mode', knowledge_mode);
    form.append('k', String(k));
    return this.http.post(`${this.base}/query`, form);
  }

  history(): Observable<any> {
    return this.http.get(`${this.base}/history`);
  }

  clear(): Observable<any> {
    return this.http.post(`${this.base}/clear`, {});
  }

  download(): Observable<Blob> {
    return this.http.get(`${this.base}/download`, { responseType: 'blob' });
  }
}
