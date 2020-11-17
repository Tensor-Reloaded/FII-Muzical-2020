import { Component, Input, OnInit } from '@angular/core';
import { Song } from '../song';
import { SongService } from '../song.service';

@Component({
  selector: 'app-song-list',
  templateUrl: './song-list.component.html',
  styleUrls: ['./song-list.component.css']
})

export class SongListComponent implements OnInit {

  @Input() songList: Song[];
  songCount: number;
  activeSortingCriteria: string;
  activeFilterCriteria: string;

  constructor(songService: SongService) { }

  ngOnInit(): void {
  }

  sortSongsBy(sortingCriteria: string) {
  }

  filterSongsBy(filterCriteria: string) {
  }

}
