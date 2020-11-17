import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { SongItemComponent } from './song-item/song-item.component';
import { SongListComponent } from './song-list/song-list.component';
import { SongDetailsComponent } from './song-details/song-details.component';
import { SoundcloudPlayerComponent } from './soundcloud-player/soundcloud-player.component';
import { SongSearchComponent } from './song-search/song-search.component';
import { SongFilterComponent } from './song-filter/song-filter.component';
import { CreateSongComponent } from './create-song/create-song.component';

@NgModule({
  declarations: [
    AppComponent,
    SongItemComponent,
    SongListComponent,
    SongDetailsComponent,
    SoundcloudPlayerComponent,
    SongSearchComponent,
    SongFilterComponent,
    CreateSongComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
