import { Genre } from './genre.enum';

export interface Song {
    Id: number;
    Title: string;
    Duration: number;
    Genre: Genre;
    HasLyrics: boolean;
    HasInstrumental: boolean;
    CreationDate: string;
}
