@Injectable()
export class FacadeService {
  
  private _songService: SongService;
  public get songService(): SongService {
    if(!this._songService){
      this._songService = this.injector.get(SongService);
    }
    return this._songService;
  }
  
  constructor(private injector: Injector) {  }

  getAllSongs(): Observable<Song[]> {
    return this._songService.getAllSongs();
  }
  createSong(options: any): Observable<Song> {
    return this._songService.createSong(options);
  }

  getSong(songId: number): Song {
    return this._songService.getSong(songId);
  }

  renameSong(songId: number, newName: string): Observable<Song> {
    return this._songService.renameSong(songId, newName);
  }

  deleteSong(songId: number): void {
    return this._songService.deleteSong(songId);
  }
}