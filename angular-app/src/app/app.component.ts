import { Component, OnInit } from '@angular/core';
import { ChatService } from './chat.service';

interface ChatMessage {
  role: 'You' | 'Bot';
  response?: string;
  sources?: string[];
  text?: string;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  files: File[] = [];
  isIndexing = false;
  isQuerying = false;
  backend = 'openai';
  knowledgeMode = 'files_only';
  question = '';
  chat: ChatMessage[] = [];
  k = 4;

  constructor(private api: ChatService) {}

  ngOnInit(): void {
    this.loadHistory();
  }

  onFilesSelected(event: any) {
    if (event.target.files && event.target.files.length) {
      this.files = Array.from(event.target.files);
    }
  }

  async uploadAndIndex(rebuild = false) {
    if (!this.files.length) {
      alert('Select at least one PDF/XLSX file.');
      return;
    }
    try {
      this.isIndexing = true;
      await this.api.uploadFiles(this.files).toPromise();
      await this.api.index(rebuild).toPromise();
      alert('Indexing complete.');
      await this.loadHistory();
    } catch (err) {
      console.error(err);
      alert('Upload or indexing failed. See console.');
    } finally {
      this.isIndexing = false;
    }
  }

  async sendQuery() {
    const q = this.question?.trim();
    if (!q) return;
    this.isQuerying = true;
    this.chat.push({ role: 'You', text: q });
    this.question = '';

    try {
      const res: any = await this.api.query(q, this.backend, this.knowledgeMode, this.k).toPromise();
      const answerText = res.answer ?? res.result ?? res.answer ?? res;
      const sources = res.sources ?? res.sources_list ?? [];
      this.chat.push({ role: 'Bot', response: answerText, sources: sources });
      await this.loadHistory();
    } catch (err) {
      console.error(err);
      this.chat.push({ role: 'Bot', response: 'Error answering: ' + (err?.message || err), sources: [] });
    } finally {
      this.isQuerying = false;
    }
  }

  setBackend(sel: string) {
    if (sel === 'openai') {
      this.backend = 'openai';
    } else {
      this.backend = 'local';
    }
  }

  async loadHistory() {
    try {
      const resp: any = await this.api.history().toPromise();
      if (resp?.history) {
        this.chat = resp.history.map((h: any) => ({ role: 'You', text: h.question }))
          .concat(resp.history.map((h: any) => ({ role: 'Bot', response: h.answer, sources: h.sources || [] })))
          .slice(-200);
      }
    } catch (err) {
      console.warn('Could not load history', err);
    }
  }

  async clearChat() {
    try {
      await this.api.clear().toPromise();
      this.chat = [];
    } catch (err) {
      console.error(err);
      alert('Failed to clear chat on server.');
    }
  }

  async downloadHistory() {
    try {
      const blob = await this.api.download().toPromise();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'chat_history.md';
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      alert('Failed to download chat history.');
    }
  }
}
