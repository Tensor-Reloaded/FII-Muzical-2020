import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})

export class SoundCloudService {

  currentSongId: number;

  constructor() { }

  initializeSong(songId: number) {
  }
}
