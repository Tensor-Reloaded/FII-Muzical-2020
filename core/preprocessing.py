from music21 import *
import numpy as np
import json
import os

listInstruments = ["Instrument", "Accordion", "AcousticBass", "AcousticGuitar", "Agogo", "Alto", "AltoSaxophone",
                   "Bagpipes", "Banjo", "Baritone", "BaritoneSaxophone", "Bass", "BassClarinet", "BassDrum",
                   "BassTrombone", "Bassoon", "BongoDrums", "BrassInstrument", "Castanets", "Celesta", "ChurchBells",
                   "Clarinet", "Clavichord", "Conductor", "CongaDrum", "Contrabass", "Contrabassoon", "Cowbell",
                   "CrashCymbals", "Cymbals", "Dulcimer", "ElectricBass", "ElectricGuitar", "ElectricOrgan",
                   "EnglishHorn", "FingerCymbals", "Flute", "FretlessBass", "Glockenspiel", "Gong", "Guitar",
                   "Handbells", "Harmonica", "Harp", "Harpsichord", "HiHatCymbal", "Horn", "Kalimba",
                   "KeyboardInstrument", "Koto", "Lute", "Mandolin", "Maracas", "Marimba", "MezzoSoprano", "Oboe",
                   "Ocarina", "Organ", "PanFlute", "Percussion", "Piano", "Piano right", "Piano left", "Piccolo",
                   "PipeOrgan", "PitchedPercussion", "Ratchet", "Recorder", "ReedOrgan", "RideCymbals",
                   "SandpaperBlocks", "Saxophone", "Shakuhachi", "Shamisen", "Shehnai", "Siren", "Sitar",
                   "SizzleCymbal", "SleighBells", "SnareDrum", "Soprano", "SopranoSaxophone", "SplashCymbals",
                   "SteelDrum", "StringInstrument", "SuspendedCymbal", "Taiko", "TamTam", "Tambourine", "TempleBlock",
                   "Tenor", "TenorDrum", "TenorSaxophone", "Timbales", "Timpani", "TomTom", "Triangle", "Trombone",
                   "Trumpet", "Tuba", "TubularBells", "Ukulele", "UnpitchedPercussion", "Vibraphone", "Vibraslap",
                   "Viola", "Violin", "Violoncello", "Vocalist", "Whip", "Whistle", "WindMachine", "Woodblock",
                   "WoodwindInstrument", "Xylophone", "Functions"]


class PreprocessMidi:
    def __init__(self):
        self.path_file = None
        self.midi = None
        self.instruments = []
        self.notes = []
        self.path_vocabulary = "data/vocabulary.json"
        self.top_vocabulary = "data/top_index_vocabulary.json"
        self.vocab = {}
        self.vocab_top = {1: 0}
        self.path_dataset = "data/dataset.json"
        self.dataset = {}
        self.path_parsed_data = "data/processed_midi/"
        self.maxlen = 100

    def parse_file(self, path_files):
        self.path_file = path_files
        self.midi = converter.parse(self.path_file)

    def extract_notes_from_instrument(self, midi_part, name, filename):
        count = 0
        for el in midi_part.flat.notes:
            if count < self.maxlen:
                val = "None"
                if str(el.offset) not in self.dataset[filename].keys():
                    self.dataset[filename][str(el.offset)] = []

                if isinstance(el, note.Note):
                    if str(el.pitch) not in self.vocab[name].keys():
                        self.vocab[name][str(el.pitch)] = set()
                    self.vocab[name][str(el.pitch)].add(str(el.duration.quarterLength))
                    val = name.replace(" ", "-") + "_" + str(el.pitch) + "_" + str(el.duration.quarterLength)

                elif isinstance(el, chord.Chord):
                    chordValues = '.'.join(str(n.pitch) for n in el)
                    if str(chordValues) not in self.vocab[name].keys():
                        self.vocab[name][str(chordValues)] = set()
                    self.vocab[name][str(chordValues)].add(str(el.duration.quarterLength))
                    val = name.replace(" ", "-") + "_" + str(chordValues) + "_" + str(el.duration.quarterLength)
                self.dataset[filename][str(el.offset)].append(val)
            count += 1

    def create_dataset(self, path, maxlen):
        self.maxlen = maxlen
        for filename in os.listdir(path):
            print("Done: ", filename)
            self.midi = converter.parse(path + filename)
            self.dataset[filename] = {}

            for i in range(len(self.midi.parts)):
                instrument_name = self.midi.parts[i].partName
                # print(str(instrument_name))
                if str(instrument_name) in listInstruments:
                    if str(instrument_name) not in self.instruments:
                        self.instruments.append(instrument_name)
                        self.vocab[instrument_name] = {}
                    instrument_notes = self.midi.parts[i].flat.notes
                    self.extract_notes_from_instrument(instrument_notes, instrument_name, filename)

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
        result = ["[NONE]", "nextOffset"]
        for instrument in data.keys():
            for notes in data[instrument].keys():
                for duration in data[instrument][notes]:
                    result.append(str(instrument).replace(" ", "-") + "_" + str(notes) + "_" + str(duration))
        return result

    def process_dataset(self, path, vocab_path):
        vocabulary = self.get_vocabulary(vocab_path)
        with open(path) as f:
            dataset = json.load(f)
        melodies = dataset.keys()
        for m in melodies:
            song = []
            for k in dataset[m].keys():
                for el in dataset[m][k]:
                    if vocabulary.index(el) not in self.vocab_top.keys():
                        self.vocab_top[vocabulary.index(el)] = 1
                    else:
                        self.vocab_top[vocabulary.index(el)] += 1
                    song.append(vocabulary.index(el))
                song.append(1)
                self.vocab_top[1] += 1

            path = "data/processed_midi/" + m.replace(".mid", "")
            data = np.array(song)
            # print(data.shape)
            np.save(path, data)
        self.save_dictionary(self.top_vocabulary, self.vocab_top)

    def get_instruments(self, path):
        with open(path) as f:
            dataset = json.load(f)
        melodies = dataset.keys()
        print(melodies)

    def vectorize(self, notes):
        vocabulary = self.get_vocabulary(self.path_vocabulary)
        result = []
        for note in notes.split():
            result.append(vocabulary.index(note))
        return np.asarray(result)

    def get_dataset(self, maxlen, skip):
        files = os.listdir(self.path_parsed_data)
        nrOfFiles = 0
        for i in range(len(files)):
            if files[i].endswith('.npy'):
                nrOfFiles += 1

        X = np.zeros((nrOfFiles, maxlen), dtype=int)
        Y = np.zeros((nrOfFiles, maxlen), dtype=int)

        el = 0
        for i in range(len(files)):
            if files[i].endswith('.npy'):
                data = np.load(self.path_parsed_data + files[i])
                if data.shape[0] >= maxlen + skip:
                    X[el] = data[:maxlen]
                    Y[el] = data[skip:maxlen + skip]
                el += 1

        return X, Y

# if __name__ == "__main__":
#     p = PreprocessMidi()
#     p.create_dataset("data/input_songs/Piano/", 100)
#     p.process_dataset("data/dataset.json", "data/vocabulary.json")