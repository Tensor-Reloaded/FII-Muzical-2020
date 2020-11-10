import { Component, Input, OnInit } from '@angular/core';
import { Song } from '../song';
import { SoundCloudService } from '../soundcloud.service';

@Component({
  selector: 'app-song-details',
  templateUrl: './song-details.component.html',
  styleUrls: ['./song-details.component.css']
})

export class SongDetailsComponent implements OnInit {

  @Input() song: Song;

  hasInstrumental: boolean;
  hasLyrics: boolean;

  constructor(soundCloudService: SoundCloudService) {
  }

  ngOnInit(): void {
  }

  initializeSoundCloudSong(songId: number) {
  }

  initializeLyrics() {
  }
}
