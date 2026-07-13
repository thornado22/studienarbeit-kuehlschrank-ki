import {Component, ElementRef, EventEmitter, Input, Output, ViewChild} from '@angular/core';
import {CommonModule} from '@angular/common';
import { NotificationService } from '../shared/notification.service';

@Component({
  selector: 'image-uploader',
  templateUrl: './image-uploader.component.html',
  styleUrls: ['./image-uploader.component.css',"../../styles.css"],
  imports: [CommonModule],
  standalone: true
})

export class ImageUploaderComponent {
  @Output() imagesUploaded = new EventEmitter<string[]>();
  @Output() imageRemoved= new EventEmitter<void>();
  @Input() activeColor: string = 'rgba(250,210,234,0.7)';
  @Input() baseColor: string = 'rgba(250, 250,250, 0,5)';
  @Input() text: string = '';
  @Input() hint: string = '';
  @Input() icon: string = '';

  @ViewChild('fileInput') fileInput?: ElementRef<HTMLInputElement>;

  private readonly MAX_IMAGES = 5;
  private readonly MAX_IMAGE_SIZE_MB = 8;

  dragging: boolean = false;
  imageLoaded: boolean = false;
  imageSrcList: string[] = [];

  constructor(private readonly notificationService: NotificationService) {}

  handleDragEnter() {
    this.dragging = true;
  }

  handleDragLeave() {
    this.dragging = false;
  }

  handleDrop(event: DragEvent) {
    event.preventDefault();
    this.dragging = false;
    void this.handleInputChange(event);
  }

  handleImageLoad() {
    this.imageLoaded = true;
  }

  openFileDialog(): void {
    this.fileInput?.nativeElement.click();
  }

  async handleInputChange(event: DragEvent | Event): Promise<void> {
    const files = this.extractFiles(event);
    if (!files.length) {
      return;
    }

    const openSlots = this.MAX_IMAGES - this.imageSrcList.length;

    if (openSlots <= 0) {
      this.notificationService.notify(`Maximal ${this.MAX_IMAGES} Bilder möglich.`);
      return;
    }

    if (files.length > openSlots) {
      this.notificationService.notify(`Maximal ${this.MAX_IMAGES} Bilder möglich.`);
    }

    const acceptedFiles = files.slice(0, openSlots).filter((file) => this.isValidImageFile(file));
    if (!acceptedFiles.length) {
      this.notificationService.notify('Bitte nur gültige Bilddateien bis 8 MB hochladen.');
      return;
    }

    const converted = await Promise.all(
      acceptedFiles.map((file) => this.readAsDataUrl(file))
    );

    this.imageSrcList = [...this.imageSrcList, ...converted];
    this.imageLoaded = false;
    this.imagesUploaded.emit(this.imageSrcList);

    if (event.target instanceof HTMLInputElement) {
      event.target.value = '';
    }
  }

  private extractFiles(event: DragEvent | Event): File[] {
    if (event instanceof DragEvent && event.dataTransfer?.files) {
      return Array.from(event.dataTransfer.files);
    }

    if (event.target instanceof HTMLInputElement && event.target.files) {
      return Array.from(event.target.files);
    }

    return [];
  }

  private isValidImageFile(file: File): boolean {
    return file.type.startsWith('image/') && file.size <= this.MAX_IMAGE_SIZE_MB * 1024 * 1024;
  }

  private readAsDataUrl(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          resolve(reader.result);
          return;
        }

        reject(new Error('Datei konnte nicht gelesen werden.'));
      };
      reader.onerror = () => reject(reader.error ?? new Error('Datei konnte nicht gelesen werden.'));
      reader.readAsDataURL(file);
    });
  }

  resetImage(){
    this.imageSrcList = [];
    this.imageLoaded = false;
    this.imageRemoved.emit();
  }


}
