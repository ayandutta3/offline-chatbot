import { Component, OnInit } from '@angular/core';
import { ChatService } from './chat.service';

interface ChatMessage {
  role: 'You' | 'Bot';
  text?: string;
  response?: string;
  sources?: string[];
  showSources?: boolean;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
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

  // ✅ Inline status message variables
  statusMessage = '';
  statusType: 'success' | 'error' | 'info' | '' = '';

  constructor(private api: ChatService) {}

  ngOnInit(): void {
    this.loadHistory();
  }

  // Helper: show status message with optional auto-hide
  showStatus(msg: string, type: 'success' | 'error' | 'info' = 'info', autoHide = true) {
    this.statusMessage = msg;
    this.statusType = type;
    if (autoHide) {
      setTimeout(() => {
        this.statusMessage = '';
        this.statusType = '';
      }, 5000); // ⏱ Hide after 5 seconds
    }
  }

  onFilesSelected(event: any) {
    if (event.target.files && event.target.files.length) {
      this.files = Array.from(event.target.files);
    }
  }

  async uploadAndIndex(rebuild = false) {
    if (!this.files.length) {
      this.showStatus('⚠️ Please select at least one PDF/XLSX file.', 'error');
      return;
    }
    try {
      this.isIndexing = true;
      this.showStatus('⏳ Indexing in progress...', 'info', false);
      await this.api.uploadFiles(this.files).toPromise();
      await this.api.index(rebuild).toPromise();
      this.showStatus('✅ Indexing complete.', 'success');
      await this.loadHistory();
    } catch (err) {
      console.error(err);
      this.showStatus('❌ Upload or indexing failed.', 'error');
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
      const res: any = await this.api
        .query(q, this.backend, this.knowledgeMode, this.k)
        .toPromise();

      const answerText = res.answer ?? res.result ?? String(res);
      const sources = res.sources ?? [];
      this.chat.push({
        role: 'Bot',
        response: answerText,
        sources,
        showSources: false,
      });
      await this.loadHistory();
    } catch (err: any) {
      console.error(err);
      this.chat.push({
        role: 'Bot',
        response: '⚠️ Error answering: ' + (err?.message || err),
        sources: [],
        showSources: false,
      });
    } finally {
      this.isQuerying = false;
    }
  }

  async loadHistory() {
    try {
      const resp: any = await this.api.history().toPromise();
      if (resp?.history) {
        this.chat = [];
        for (const h of resp.history) {
          this.chat.push({ role: 'You', text: h.question });
          this.chat.push({
            role: 'Bot',
            response: h.answer,
            sources: h.sources || [],
            showSources: false,
          });
        }
      }
    } catch (err) {
      console.warn('Could not load history', err);
    }
  }

  async clearChat() {
    try {
      await this.api.clear().toPromise();
      this.chat = [];
      this.showStatus('🧹 Chat cleared successfully.', 'info');
    } catch (err) {
      console.error(err);
      this.showStatus('❌ Failed to clear chat.', 'error');
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
      this.showStatus('💾 Chat history downloaded.', 'success');
    } catch (err) {
      console.error(err);
      this.showStatus('❌ Failed to download chat.', 'error');
    }
  }

  toggleSources(msg: ChatMessage) {
    msg.showSources = !msg.showSources;
  }
}
