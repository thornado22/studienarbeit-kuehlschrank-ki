import { Component, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NavbarComponent } from './navbar/navbar.component';
import { ImageUploaderComponent } from './image-uploader/image-uploader.component';
import { WebcamSnapshotComponent } from './webcam-snapshot/webcam-snapshot.component';
import { FooterComponent } from './footer/footer.component';
import { SendButton } from './send-button/send-button';
import { ChatPage } from './chat-page/chat-page';
import { RemoveButton } from './remove-button/remove-button';
import { Aboutus } from './aboutus/aboutus';
import { Infopage } from './infopage/infopage';
import { ImageSource, ImageStateService } from './shared/image-state.service';
import { NotificationService } from './shared/notification.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  standalone: true,
  imports: [
    CommonModule,
    NavbarComponent,
    ImageUploaderComponent,
    WebcamSnapshotComponent,
    FooterComponent,
    SendButton,
    ChatPage,
    RemoveButton,
    Aboutus,
    Infopage
  ]
})
export class AppComponent {
  title = 'KüKi';
  activeView: 'home' | 'chatbot' | 'about' | 'info' = 'home';

  setActiveView(view: 'home' | 'chatbot' | 'about' | 'info'): void {
    this.activeView = view;
  }

  imageAvailable = false;
  switchOnVideo = false;
  image: string | null = null;
  imagesCount = 0;
  uploadError: string | null = null;
  sendButtonDisabled = false;
  notificationMessage: string | null = null;

  @ViewChild('imageUploaderComponent') imageUploader!: ImageUploaderComponent;
  @ViewChild('webcamSnapshotComponent') webcamSnapshot?: WebcamSnapshotComponent;

  constructor(
    private readonly imageState: ImageStateService,
    private readonly notificationService: NotificationService,
  ) {
    this.notificationService.notification$.subscribe((message) => {
      this.notificationMessage = message;
    });
  }

  private syncImageState(): void {
    this.imagesCount = this.imageState.images.length;
    this.image = this.imageState.image;
    this.imageAvailable = this.imagesCount > 0;
  }

  onImagesUploaded(images: string[], source: ImageSource){
    this.imageState.setImages(images, source);
    this.syncImageState();
  }

  onImageRemoved(source: ImageSource){
    this.imageState.clearIfSource(source);
    this.syncImageState();
    this.uploadError = ""
  }

  onRemovalButtonClick() {
    this.imageState.clearImage();
    this.syncImageState();
    this.uploadError = null;
    this.sendButtonDisabled = false;

    this.imageUploader?.resetImage();
    this.webcamSnapshot?.resetSnapshot();
  }

  onSendClicked() {
    this.setActiveView('chatbot');
  }
}