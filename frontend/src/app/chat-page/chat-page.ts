import { ChatInput } from '../chat-input/chat-input';
import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatMessageList, ChatMessage } from '../chat-message-list/chat-message-list';
import { ImageStateService } from '../shared/image-state.service';
import { SendToCnnService } from '../utils/sendToCnn';


@Component({
  selector: 'app-chat-page',
  standalone: true,
  imports: [CommonModule, ChatInput, ChatMessageList],
  templateUrl: './chat-page.html',
  styleUrls: ['./chat-page.css'],

  
})
export class ChatPage implements OnInit {
  messages: ChatMessage[] = [];
  private hasStartedRecipeChat = false;

  constructor(
    private readonly imageState: ImageStateService,
    private readonly cnnService: SendToCnnService,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    const imgs = this.imageState.images;
    if (imgs.length > 0) {
      const initialMessage: ChatMessage = {
        role: 'user',
        text: '',
        food: [],
        images: imgs.map(i => i.dataUrl),
        createdAt: new Date(),
      };

      this.onMessageReceived(initialMessage);
      this.imageState.clearImage();
    }
  }

  onMessageReceived(message: ChatMessage): void {
    console.log('[ChatPage] Nachricht angekommen:', message);
    this.messages = [...this.messages, message];

    const requestImages = message.images ?? [];
    const question = message.text.trim();

    if (!this.hasStartedRecipeChat) {
      if (!requestImages.length) {
        const assistantReply: ChatMessage = {
          role: 'assistant',
          text: 'Bitte lade mindestens ein Bild hoch, damit ich ein Rezept vorschlagen kann.',
          food: [],
          createdAt: new Date(),
        };
        this.messages = [...this.messages, assistantReply];
        return;
      }

      this.cnnService.sendImageToCnnInitial(requestImages).subscribe({
        next: (response) => {
          this.hasStartedRecipeChat = true;
          const assistantReply: ChatMessage = {
            role: 'assistant',
            food: response.food,
            text: response.recipe,
            createdAt: new Date(),
          };

          this.messages = [...this.messages, assistantReply];
          this.cdr.detectChanges();
          console.log('[ChatPage] Assistant-Antwort aus API hinzugefuegt:', assistantReply);
        },
        error: (err) => {
          console.error('[ChatPage] API Fehler:', err);
          const assistantReply: ChatMessage = {
            role: 'assistant',
            food: [],
            text: 'Die Rezept-Antwort konnte nicht geladen werden. Bitte versuche es erneut.',
            createdAt: new Date(),
          };
          this.messages = [...this.messages, assistantReply];
          this.cdr.detectChanges();
        },
      });
      return;
    }

    if (!question && !requestImages.length) {
      return;
    }

    this.cnnService.sendImageToCnnFollowUp(requestImages, question).subscribe({
      next: (response) => {
        const assistantReply: ChatMessage = {
          role: 'assistant',
          food: response.food,
          text: response.recipe,
          createdAt: new Date(),
        };

        this.messages = [...this.messages, assistantReply];
        this.cdr.detectChanges();
        console.log('[ChatPage] Assistant-Follow-Up-Antwort aus API hinzugefuegt:', assistantReply);
      },
      error: (err) => {
        console.error('[ChatPage] API Fehler:', err);
        const assistantReply: ChatMessage = {
          role: 'assistant',
          food: [],
          text: 'Die Rezept-Antwort konnte nicht geladen werden. Bitte versuche es erneut.',
          createdAt: new Date(),
        };
        this.messages = [...this.messages, assistantReply];
        this.cdr.detectChanges();
      },
    });
  }
}
