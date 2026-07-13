import { Component, Output, EventEmitter, Input } from '@angular/core';

@Component({
  selector: 'app-send-button',
  imports: [],
  templateUrl: './send-button.html',
  styleUrls: ['./send-button.css'],
  standalone: true
})
export class SendButton {
  constructor() {}

  @Input() label = 'Entdecke Rezepte';
  @Input() floating = true;
  @Output() sendClicked = new EventEmitter<void>();

  onSendClick(): void {
    console.log('[SendButton] click erkannt');
    this.sendClicked.emit();
  }
}

