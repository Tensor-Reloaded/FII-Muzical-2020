from music21 import *
import numpy as np
import json
import os

class PreprocessMidi:
    def __init__(self):
        self.path_file = None
        self.midi = None
        self.instruments = []
        self.notes = []
        self.vocab = {}
        self.dataset = {}
        self.path_vocabulary = "data/vocabulary.json"
        self.path_dataset = "data/dataset.json"
        self.top_vocabulary = "data/top_index_vocabulary.json"
        self.path_parsed_data = "data/processed_midi/"

    def extract_notes_from_instrument(self, song, filename):
        s = song.parts.stream()
        for part in s:
            notes_to_parse = part.flat.notes
            for el in notes_to_parse:
                instrument_name = part.partName
                if " " in instrument_name:
                    instrument_name = instrument_name.replace(" ","-")
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

                if instrument_name not in self.instruments:
                    self.instruments.append(instrument_name)
                    self.vocab[instrument_name] = {}

                if data not in self.vocab[instrument_name].keys():
                    self.vocab[instrument_name][data] = set()

                if r not in self.dataset[filename].keys():
                    self.dataset[filename][r] = []
                
                self.vocab[instrument_name][data].add(str(el.duration.quarterLength))
                self.dataset[filename][r].append(result)

    def create_dataset(self, path):
        for filename in os.listdir(path):
            if filename.endswith('.mid'):
                print("Done: ", filename)
                midi = converter.parse(path + filename)
                self.dataset[filename] = {}
                self.extract_notes_from_instrument(midi, filename)

        self.save_dictionary(self.path_vocabulary, self.vocab)
        self.save_dictionary(self.path_dataset, self.dataset)

    def set_convert(self, obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError

    def save_dictionary(self, path, data):
        with open(path, 'w') as fp:
            json.dump(data, fp, default=self.set_convert)

    def get_vocabulary(self, path):
        with open(path) as f:
            data = json.load(f)
        vocab = ["0.35"]
        for instrument in data.keys():
            for notes in data[instrument].keys():
                for duration in data[instrument][notes]:
                    vocab.append(str(instrument).replace(" ", "-") + "_" + str(notes) + "_" + str(duration))
        return vocab

    def get_json_dataset(self, path):
        with open(path) as f:
            dataset = json.load(f)
        return dataset

    def process_song(self, song, vocabulary):
        song_offsets = list(song.keys())
        a = np.array(song_offsets, dtype=float)
        a = np.sort(a)
        result = []
        for el in range(len(a)-1):
            t = str(a[el])
            # data = vocabulary.index(str(round(a[el+1] - a[el],2)))
            # result.append(data)
            result.append(0)
            for val in song[t]:
                data = vocabulary.index(val)
                result.append(data)
            
        return result

    def process_dataset(self, json_dataset_path, vocab_path):
        vocab = self.get_vocabulary(vocab_path)
        json_dataset = self.get_json_dataset(json_dataset_path)
        for song in json_dataset.keys():
            data = np.array(self.process_song(json_dataset[song], vocab))
            path = "data/processed_midi/" + song.replace(".mid", "")
            np.save(path, data)

    def vectorize(self, notes):
        vocab = self.get_vocabulary(self.path_vocabulary)
        result = []
        for note in notes:
            result.append(vocab.index(note))
        return np.asarray(result)

    def get_patch(self, patch_size, data):
        cross_point = int(patch_size/2)
        h = int(data.shape[0] / patch_size)
        a = int( h * patch_size / cross_point)
        x1 = data[:a*cross_point].reshape(a, cross_point)
        x2 = data[:a*cross_point].reshape(a, cross_point)
        x1 = np.delete(x1, (-1), axis=0)
        x2 = np.delete(x2, (0), axis=0)
        result = np.concatenate((x1, x2),axis=1)
        result = np.vstack([result, data[data.shape[0]-patch_size:]])
        return result

    def get_dataset(self, patch_size, skip):
        files = os.listdir(self.path_parsed_data)
        el = 0
        for file in files:
            if file.endswith('.npy'):
                dataX = np.load(self.path_parsed_data + file)
                dataY = dataX[skip:]
                dataY = np.append(dataY, [0]*skip)
                if el == 0:
                    X = self.get_patch(patch_size, dataX)
                    Y = self.get_patch(patch_size, dataY)
                else:
                    X = np.concatenate((X, self.get_patch(patch_size, dataX)),axis=0)
                    Y = np.concatenate((Y, self.get_patch(patch_size, dataY)),axis=0)
                el+=1
        return X, Y

if __name__ == "__main__":
    p = PreprocessMidi()
    p.create_dataset("data/input_songs/Piano/")
    p.process_dataset("data/dataset.json", "data/vocabulary.json")

# if __name__ == "__main__":
#     p = PreprocessMidi()
#     p.get_dataset(64,3)