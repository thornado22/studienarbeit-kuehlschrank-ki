import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class NotificationService {
  private readonly notificationSubject = new BehaviorSubject<string | null>(null);
  readonly notification$: Observable<string | null> = this.notificationSubject.asObservable();
  private timeoutId?: number;

  notify(message: string): void {
    if (this.timeoutId) {
      window.clearTimeout(this.timeoutId);
    }

    this.notificationSubject.next(message);
    this.timeoutId = window.setTimeout(() => {
      this.notificationSubject.next(null);
      this.timeoutId = undefined;
    }, 2000);
  }

  clear(): void {
    if (this.timeoutId) {
      window.clearTimeout(this.timeoutId);
      this.timeoutId = undefined;
    }
    this.notificationSubject.next(null);
  }
}
