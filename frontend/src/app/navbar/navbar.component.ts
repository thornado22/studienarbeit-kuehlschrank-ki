import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.css'],
  standalone: true,
  imports: [CommonModule]
})

export class NavbarComponent {
  @Input() activeView: 'home' | 'chatbot' | 'about' | 'info' = 'home';
  @Output() homeClicked = new EventEmitter<void>();
  @Output() infoClicked = new EventEmitter<void>();

  onHomeClick(): void {
    this.homeClicked.emit();
  }

  onInfoClicked(): void {
    this.infoClicked.emit();
  }

}
