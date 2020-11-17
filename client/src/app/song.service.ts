import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { Song } from './song';
import { HttpClient} from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})

export class SongService {

  constructor(private httpService: HttpClient) {
  }

  getAllSongs(): Observable<Song[]> {
  }

  getSong(songId: number): Song {
  }

  createSong(options: any): Observable<Song> {
  }

  renameSong(songId: number, newName: string): Observable<Song> {
  }

  deleteSong(songId: number): void {
  }
}
