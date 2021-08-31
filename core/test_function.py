import numpy as np
import os

from preprocessing import PreprocessMidi
from postprocessing import PostprocessMidi

from music21 import *
from fractions import Fraction

# patch_size = 10
# cross_point = int(patch_size/2)

# files = os.listdir("data/processed_midi/")
# for i in range(len(files)):
#     if files[i].endswith('.npy'):
#         data = np.load("data/processed_midi/" + files[i])
#         h = int(data.shape[0] / patch_size)
#         a = int( h * patch_size / cross_point)
#         x1 = data[:a*cross_point].reshape(a, cross_point)
#         x2 = data[:a*cross_point].reshape(a, cross_point)
#         x1 = np.delete(x1, (-1), axis=0)
#         x2 = np.delete(x2, (0), axis=0)
#         result = np.concatenate((x1, x2),axis=1)
#         result = np.vstack([result, data[data.shape[0]-patch_size:]])
#         print(result)
#         break

m = converter.parse("data/input_songs/Piano/aaa/beethoven_opus90_2.mid")

dataset = {}
filename = "beethoven_opus90_2"
dataset[filename] = {}

s2 = m.parts.stream()

for part in s2:
    notes_to_parse = part.flat.notes
    for el in notes_to_parse:
        instrument_name = part.partName.replace(" ","-")
        el_duration = str(el.duration.quarterLength)
        if isinstance(el, note.Note):
            data = str(el.pitch)
            
        elif isinstance(el, chord.Chord):
            data = '.'.join(str(n.pitch) for n in el)
        
        result = instrument_name + "_" + data + "_" + el_duration

       
        if "/" in str(el.offset):
            x = str(el.offset).split("/")
            r = int(x[0])/int(x[1])
            r = str(round(r, 2))
        else:
            r = str(el.offset)

        if r not in dataset[filename].keys():
            dataset[filename][r] = []
        dataset[filename][r].append(result)


# print(dataset)




# +++++++++++++

mel = dataset["beethoven_opus90_2"]
k = list(mel.keys())
a = np.array(k, dtype=float)
a = np.sort(a)

result = []
for el in range(len(a)-1):
    t = str(a[el])
    result.append(str(a[el+1] - a[el]))
    for val in mel[t]:
        result.append(val)
    # print(a[el+1], mel[t])

# print(result)

convert_midi = PostprocessMidi()
convert_midi.compute_song(result, "test2")

# def decode(a, time):
#     data = a.split("_")
#     data[0] = data[0].replace("-"," ")

#     if '.' in data[1]:
#         str_chord = data[1].replace('.', " ")
#         new_chord = chord.Chord(str_chord)
#         new_chord.duration.quarterLength = Fraction(data[2])
#         new_chord.storedInstrument = data[0]
#         new_chord.offset = time
#         return new_chord
    
#     else:
#         new_note = note.Note(data[1])
#         new_note.duration.quarterLength = Fraction(data[2])
#         new_note.storedInstrument = data[0]
#         new_note.offset = time
#         return new_note

    
# decoded_notes = []
# time = 0

# for r in result:
#     if '_' not in r:
#         time += float(r)
#     else:
#         decoded_notes.append(decode(r, time))

# # print(decoded_notes)

# midi_stream = stream.Stream(decoded_notes)
# midi_stream.write('midi', fp="data/output_songs/test.mid")


