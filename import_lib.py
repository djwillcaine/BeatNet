import os
import sys
import wave

import matplotlib.pyplot as plt
import numpy as np
import lxml.etree as ET
from urllib.parse import unquote
from multiprocessing.dummy import Pool as ThreadPool

BUFFER = 5

class Track:
    def __init__(self, trackid, location, bpm, inizio):
        self.trackid = trackid
        self.location = os.path.abspath(unquote(location))
        self.bpm = bpm
        self.inizio = int(inizio * 1000)

    def load_random_wav_sample(self):
        wav = wave.open(self.location)
        self.frame_rate = wav.getframerate()
        self.start_frame = np.random.randint(
            BUFFER * self.frame_rate,
            wav.getnframes() - ((BUFFER + 1) * self.frame_rate))
        wav.setpos(self.start_frame)
        frames = wav.readframes(int(np.round(self.frame_rate * 2.56)))
        wav.close()
        self.samples = np.frombuffer(frames, 'int16')[::2]

        beat_len = int(60 * self.frame_rate / self.bpm)
        offset_samples = (beat_len -
            (self.start_frame - (self.frame_rate * (self.inizio / 1000))) % beat_len)
        self.sample_inizio = int(np.round(offset_samples / self.frame_rate, 3) * 1000)

    def export_specgram(self):
        plt.figure(figsize=(2.56, 1.28), frameon=False).add_axes([0, 0, 1, 1])
        plt.axis('off')
        plt.specgram(self.samples, Fs=self.frame_rate)
        filename = ('specgrams/%s-%s-%s-%s.png' %
            (self.trackid, self.start_frame, self.bpm, self.sample_inizio))
        plt.savefig(filename)
        plt.close()

def load_tracks(lib_xml_file):
    tree = ET.parse(lib_xml_file)
    root = tree.getroot()
    tracks = []
    for trackid in root.find('PLAYLISTS').find('NODE').find('NODE[@Name="BEATNET"]').iter('TRACK'):
        track = root.find('COLLECTION').find('TRACK[@TrackID="' + trackid.get('Key') + '"]')
        location = unquote(track.get('Location')).replace('file://localhost/', '')
        if location[-4:].upper() != '.WAV':
            print('None wav file found, skipping... (%s)' % location)
            continue
        tracks.append(Track(
            trackid.get('Key'),
            location,
            float(track.find('TEMPO').get('Bpm')),
            float(track.find('TEMPO').get('Inizio'))
            ))
    return tracks

def generate_samples(lib_xml_file, n=1000):
    n = int(n)
    print('Loading library...')
    tracks = load_tracks(lib_xml_file)

    print('Generating Spectrograms...')
    for i in range(n):
        track = tracks[np.random.randint(0, len(tracks))]
        track.load_random_wav_sample()
        track.export_specgram()
        progress = i / n * 100.0
        print('\r[%s%s] %3.1f%%' % ('=' * int(progress/2),
                                    ' ' * (50 - int(progress/2)),
                                    progress),
              end='', flush=True)
    
    print('\nDone.')

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 1:
        exit('Please specify a library file')
    generate_samples(*argv)
