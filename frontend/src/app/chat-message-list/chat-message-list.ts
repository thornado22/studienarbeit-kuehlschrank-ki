import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { marked } from 'marked';

export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  food?: string[];
  text: string;
  images?: string[];
  createdAt: Date;
}

@Component({
  selector: 'app-chat-message-list',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './chat-message-list.html',
  styleUrls: ['./chat-message-list.css'],
})
export class ChatMessageList {
  @Input() messages: ChatMessage[] = [];

  constructor(private sanitizer: DomSanitizer) {}

  parseMarkdown(text: string) {
    const html = marked.parse(text || '') as string;
    return this.sanitizer.bypassSecurityTrustHtml(html);
  }
}
