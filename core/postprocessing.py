from fractions import Fraction
from music21 import *


class PostprocessMidi:
    def __init__(self):
        self.path_output = "data/output_songs/"

    def decode(self, data, time):
        data = data.split("_")
        data[0] = data[0].replace("-"," ")
        
        if '.' in data[1]:
            str_chord = data[1].replace('.', " ")
            new_chord = chord.Chord(str_chord)
            new_chord.duration.quarterLength = Fraction(data[2])
            new_chord.storedInstrument = data[0]
            new_chord.offset = time
            return new_chord
        
        else:
            new_note = note.Note(data[1])
            new_note.duration.quarterLength = Fraction(data[2])
            new_note.storedInstrument = data[0]
            new_note.offset = time
            return new_note

    def compute_song(self, result, name):
        decoded_notes = []
        time = 0
        print("\n", result)
        for r in result:
            if '_' not in r:
                time += float(r)
            else:
                decoded_notes.append(self.decode(r, time))
        
        midi_stream = stream.Stream(decoded_notes)
        midi_stream.write('midi', fp=self.path_output+str(name)+".mid")