import { Component, OnInit } from '@angular/core';
import { Genre } from '../genre.enum';
import { SongService } from '../song.service';

@Component({
  selector: 'app-create-song',
  templateUrl: './create-song.component.html',
  styleUrls: ['./create-song.component.css']
})
export class CreateSongComponent implements OnInit {

  songTitle: string;
  instrumentalOptionSelected: boolean;
  lyricsOptionSelected: boolean;
  selectedGenre: Genre;

  constructor(songService: SongService) {
  }

  ngOnInit(): void {
  }

  createSong() {
  }

  cancelCreateSong() {
  }
}
