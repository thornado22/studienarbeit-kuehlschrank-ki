import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { SendToCnnService } from '../utils/sendToCnn';

export type ImageSource = 'upload' | 'webcam';

export interface StoredImage {
  dataUrl: string;
  source: ImageSource;
}

@Injectable({
  providedIn: 'root'
})
export class ImageStateService {
  private readonly imagesSubject = new BehaviorSubject<StoredImage[]>([]);

  get image(): string | null {
    return this.imagesSubject.value.at(-1)?.dataUrl ?? null;
  }

  get images(): StoredImage[] {
    return this.imagesSubject.value;
  }

  constructor(private readonly cnnService: SendToCnnService) {
  }

  setImages(images: string[], source: ImageSource): void {
    const cleaned = images.filter((img) => !!img);
    const preserved = this.imagesSubject.value.filter((entry) => entry.source !== source);
    const incoming = cleaned.map((dataUrl) => ({ dataUrl, source, kind: 'original' as const }));
    this.updateImages([...preserved, ...incoming]);
  }

  /**
   * Clears all stored image state
   */
  clearImage(): void {
    this.updateImages([]);
  }

  clearIfSource(source: ImageSource): void {
    this.updateImages(this.imagesSubject.value.filter((entry) => entry.source !== source));
  }

  private updateImages(images: StoredImage[]): void {
    this.imagesSubject.next(images);
  }
}
