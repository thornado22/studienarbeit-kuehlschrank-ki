import { Component, Output, EventEmitter, Input } from '@angular/core';

@Component({
  selector: 'app-remove-button',
  imports: [],
  templateUrl: './remove-button.html',
  styleUrls: ['./remove-button.css'],
})
export class RemoveButton {

  @Input() floating = true;

  @Output() removeClicked = new EventEmitter<void>();

  onRemoveClick(): void {
    console.log('[RemoveButton] click erkannt');
    this.removeClicked.emit();
  }
}

