from fractions import Fraction
from music21 import *


class PostprocessMidi:
    def __init__(self):
        self.output_notes = []
        self.last_instrument = ""
        self.last_duration = ""
        self.offset = 0
        self.path_output = "data/output_songs/"

    def compute_song(self, data, name):
        data = data.split(" ")
        for el in data:
            val = el.split("_")
            if len(val) > 1:
                if self.last_instrument == val[0]:
                    self.offset += last_duration

                if '.' in val[1]:
                    notes_in_chord = val[1].replace('.', " ")
                    new_chord = chord.Chord(notes_in_chord)
                    new_chord.duration.quarterLength = Fraction(val[2])
                    new_chord.storedInstrument = val[0]
                    new_chord.offset = self.offset
                    self.output_notes.append(new_chord)

                else:
                    new_note = note.Note(val[1])
                    new_note.duration.quarterLength = Fraction(val[2])
                    new_note.storedInstrument = val[0]
                    new_note.offset = self.offset
                    self.output_notes.append(new_note)

                last_instrument = val[0]
                last_duration = Fraction(val[2])

            if val[0] == "nextOffset":
                if last_instrument != "nextOffset":
                    last_instrument = "nextOffset"
                    self.offset += 0.35

        midi_stream = stream.Stream(self.output_notes)
        midi_stream.write('midi', fp=self.path_output+str(name)+".mid")