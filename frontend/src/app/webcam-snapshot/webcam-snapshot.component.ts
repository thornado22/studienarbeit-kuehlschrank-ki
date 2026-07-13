import { Component, ElementRef, EventEmitter, Input, Output, ViewChild, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NotificationService } from '../shared/notification.service';

@Component({
  selector: 'webcam-snapshot',
  templateUrl: './webcam-snapshot.component.html',
  styleUrls: ['./webcam-snapshot.component.css'],
  standalone: true,
  imports: [CommonModule],
})

export class WebcamSnapshotComponent implements OnDestroy {
  @Output() imagesSnapped = new EventEmitter<string[]>();
  @Output() imageRemoved = new EventEmitter<void>();

  @ViewChild("video") public video?: ElementRef<HTMLVideoElement>;
  @ViewChild("canvas") public canvas?: ElementRef<HTMLCanvasElement>;

  @Input() text: string = '';
  @Input() icon: string = '';

  private readonly MAX_SNAPSHOTS = 5;

  error = '';
  isCameraRunning = false;

  width = 290;
  height = 230;

  captures: string[] = [];

  constructor(private readonly notificationService: NotificationService) {}

  ngOnDestroy(): void {
    this.turnCameraOff();
  }


  async onLabelClick() {
    if (!this.isCameraRunning) {
      this.isCameraRunning = true; // sorgt dafür, dass video/canvas im DOM erscheinen
      setTimeout(() => this.setupDevices(), 0);
    }
  }

  /**
   * Requests camera access and initializes media stream
   * @description Checks browser support, validates DOM elements, requests getUserMedia, handles errors
   */
  async setupDevices() {
    if (!navigator.mediaDevices?.getUserMedia) {
      this.error = 'Camera API not supported in this browser';
      this.isCameraRunning = false;
      return;
    }

    if (!this.video?.nativeElement) {
      this.error = 'Video element not available yet';
      this.isCameraRunning = false;
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      this.video.nativeElement.srcObject = stream;
      await this.video.nativeElement.play();
      this.error = '';
      this.isCameraRunning = true;
    } catch (e: any) {
      this.error = e?.message || 'Camera access failed';
      this.isCameraRunning = false;
    }
  }

  /**
   * Captures current video frame to canvas and stores as base64
   * @description Extracts image data, appends to local capture list, emits full batch
   */
  capture() {
    if (!this.video?.nativeElement || !this.canvas?.nativeElement) {
      this.error = 'Camera is not ready yet';
      return;
    }

    if (this.captures.length >= this.MAX_SNAPSHOTS) {
      this.notificationService.notify(`Maximal ${this.MAX_SNAPSHOTS} Bilder möglich.`);
      return;
    }

    const imageData = this.getImageData();
    if (!imageData) {
      this.error = 'Snapshot konnte nicht erstellt werden.';
      return;
    }

    this.captures.push(imageData);
    this.imagesSnapped.emit(this.captures);
  }

  /**
   * Fully resets component and global state (e.g., called by parent remove button)
   * @description Clears all captures, canvas, closes camera, clears global store, emits removal
   */
  resetSnapshot(): void {
    this.captures = [];

    if (this.canvas?.nativeElement) {
      const ctx = this.canvas.nativeElement.getContext('2d');
      ctx?.clearRect(0, 0, this.width, this.height);
    }

    this.turnCameraOff();
    this.imageRemoved.emit();
  }

  /**
   * Extracts video dimensions and renders video frame to canvas
   * @param image HTMLVideoElement with active stream
   * @description Syncs canvas size to video resolution, applies video frame to 2D context
   */
  drawImageToCanvas(image: HTMLVideoElement) {
    if (!this.video?.nativeElement || !this.canvas?.nativeElement) return;

    this.width = this.video.nativeElement.videoWidth;
    this.height = this.video.nativeElement.videoHeight;
    this.canvas.nativeElement.width = this.width;
    this.canvas.nativeElement.height = this.height;
    this.canvas.nativeElement.getContext("2d")?.drawImage(image, 0, 0, this.width, this.height);
  }

  /**
   * Encodes current canvas as base64 PNG data URL
   * @returns base64 encoded image string, or empty string on error
   */
  getImageData() {
    if (!this.video?.nativeElement || !this.canvas?.nativeElement) return '';
    this.drawImageToCanvas(this.video.nativeElement);
    return this.canvas.nativeElement.toDataURL("image/png");
  }

  /**
   * Stops all active media stream tracks and closes camera
   */
  turnCameraOff() {
    if (this.video?.nativeElement?.srcObject) {
      const stream = this.video.nativeElement.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      this.video.nativeElement.srcObject = null;
    }
    this.isCameraRunning = false;
  }
}