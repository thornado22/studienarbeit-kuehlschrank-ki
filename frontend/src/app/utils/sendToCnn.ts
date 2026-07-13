import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

interface RecipeResponse {
  recipe: string;
  food: string[];
  chat_history: unknown[];
}

@Injectable({ providedIn: 'root' })
export class SendToCnnService {

  private readonly initialApiUrl = 'http://localhost:8000/api/recipe/initial';
  private readonly followUpApiUrl = 'http://localhost:8000/api/recipe/followup';

  constructor(private readonly http: HttpClient) {}

  sendImageToCnnInitial(images: string[]): Observable<RecipeResponse> {

    const payload = {
      images
    };

    console.log('POST payload:', payload);

    return this.http.post<RecipeResponse>(this.initialApiUrl, payload);
  }

  sendImageToCnnFollowUp(images: string[], question: string): Observable<RecipeResponse> {

    const payload = {
      images,
      question
    };

    console.log('POST payload:', payload);

    return this.http.post<RecipeResponse>(this.followUpApiUrl, payload);
  }
}