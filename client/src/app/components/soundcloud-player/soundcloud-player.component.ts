import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-soundcloud-player',
  templateUrl: './soundcloud-player.component.html',
  styleUrls: ['./soundcloud-player.component.css']
})
export class SoundcloudPlayerComponent implements OnInit {

  currentSongId: number;

  constructor() {
  }

  ngOnInit(): void {
  }

  initializeSong(id: number) {
  }

}
