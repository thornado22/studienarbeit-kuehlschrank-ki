import { Component, Output, EventEmitter, ViewChild, ElementRef, HostListener } from '@angular/core';
import { SendButton } from '../send-button/send-button';
import { RemoveButton } from '../remove-button/remove-button';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ImageUploaderComponent } from '../image-uploader/image-uploader.component';
import { WebcamSnapshotComponent } from '../webcam-snapshot/webcam-snapshot.component';
import { ImageSource, ImageStateService } from '../shared/image-state.service';


type ChatRole = 'user' | 'assistant';

interface ChatMessage {
  role: ChatRole;
  text: string;
  createdAt: Date;
  images?: string[];
}


@Component({
  selector: 'app-chat-input',
  standalone: true,
  imports: [SendButton, RemoveButton, CommonModule, FormsModule, ImageUploaderComponent, WebcamSnapshotComponent],
  templateUrl: './chat-input.html',
  styleUrls: ['./chat-input.css'],
  
})
export class ChatInput {
  @Output() messageSent = new EventEmitter<ChatMessage>();

  @ViewChild('imageUploaderComponent') imageUploader?: ImageUploaderComponent;
  @ViewChild('webcamSnapshotComponent') webcamSnapshot?: WebcamSnapshotComponent;
  @ViewChild('popupMenu') popupMenu?: ElementRef<HTMLElement>;
  @ViewChild('popupToggle') popupToggle?: ElementRef<HTMLElement>;

  constructor(private readonly imageState: ImageStateService) {}
  draftMessage = '';
  showPopup = false;

  switchOnVideo = false;

  onDraftChange(value: string): void {
    this.draftMessage = value;
    console.log('[ChatInput] draftMessage aktualisiert:', this.draftMessage);
  }

  onEnter(event: Event): void {
    if (!(event instanceof KeyboardEvent)) {
      return;
    }

    if (event.shiftKey) {
      return;
    }

    event.preventDefault();
    this.sendMessage();
  }

  togglePopup(event?: Event): void {
    event?.stopPropagation();
    this.showPopup = !this.showPopup;
  }

  closePopup(): void {
    this.showPopup = false;
  }

  onPopupRemoveClick(): void {
    this.imageUploader?.resetImage();
    this.webcamSnapshot?.resetSnapshot();
    this.imageState.clearImage();
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    if (!this.showPopup) {
      return;
    }

    const target = event.target as Node | null;
    const menu = this.popupMenu?.nativeElement;
    const toggle = this.popupToggle?.nativeElement;

    if (!target) {
      return;
    }

    if (menu?.contains(target) || toggle?.contains(target)) {
      return;
    }

    this.closePopup();
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    this.closePopup();
  }

  sendMessage(): void {
    const text = this.draftMessage.trim();
    const images = this.imageState.images.map(img => img.dataUrl);

    if (!text && images.length === 0) return;
    
    console.log('[ChatInput] sendMessage mit Text:', text, 'und Bildern:', images);

    this.messageSent.emit({
      role: 'user',
      text,
      createdAt: new Date(),
      images
    });

    this.draftMessage = '';

    this.imageState.clearImage();
  }
  onImagesUploaded(images: string[], source: ImageSource) {
    this.imageState.setImages(images, source);
  }

}
