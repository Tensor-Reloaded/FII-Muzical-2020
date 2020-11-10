import { Input } from '@angular/core';
import { Component, OnInit } from '@angular/core';
import { Song } from '../song';
import { SongService } from '../song.service';

@Component({
  selector: 'app-song-item',
  templateUrl: './song-item.component.html',
  styleUrls: ['./song-item.component.css']
})

export class SongItemComponent implements OnInit {

  @Input() song: Song;

  constructor(songService: SongService) {
  }

  ngOnInit(): void {
  }

  showSongDetails() {
  }

  renameSong(newName: string) {
  }

  deleteSong() {
  }
}
