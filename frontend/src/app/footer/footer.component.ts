import { Component, Output, Input, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-footer',
  templateUrl: './footer.component.html',
  styleUrls: ['./footer.component.css'],
  standalone: true,
  imports: [CommonModule]
})

export class FooterComponent {
  @Output() aboutClicked = new EventEmitter<void>();
  @Input() activeView: 'home' | 'chatbot' | 'about' | 'info' = 'home';

  onAboutClicked(): void {
    this.aboutClicked.emit();
  }
}