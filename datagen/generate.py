import os
import sys
import wave
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import lxml.etree as ET
from urllib.parse import unquote
from multiprocessing.dummy import Pool as ThreadPool

FRAME_RATE = 22050
CHUNK_SIZE = FRAME_RATE * 2
BUFFER = FRAME_RATE * 5

class Track:
    def __init__(self, trackid, location, bpm):
        self.trackid = trackid
        self.location = os.path.abspath(unquote(location))
        self.bpm = bpm

    def generate_random_specgram(self):
        audio = AudioSegment.from_file(self.location)
        audio = audio.set_channels(1).set_frame_rate(FRAME_RATE)
        samples = audio.get_array_of_samples()
        start = np.random.randint(BUFFER, len(samples) - BUFFER)
        chunk = samples[start:start + CHUNK_SIZE]

        filename = ('specgrams/%s-%s-%s.png' % (self.trackid, start, self.bpm))

        plt.figure(figsize=(2.56, 0.32), frameon=False).add_axes([0, 0, 1, 1])
        plt.axis('off')
        plt.specgram(chunk, Fs = FRAME_RATE)
        plt.savefig(filename)
        plt.close()

def load_tracks(lib_xml_file):
    tree = ET.parse(lib_xml_file)
    root = tree.getroot()
    tracks = {}

    valid = 0
    skipped = 0
    for trackid in root.find('PLAYLISTS').find('NODE').find('NODE[@Name="BEATNET"]').iter('TRACK'):
        track = root.find('COLLECTION').find('TRACK[@TrackID="' + trackid.get('Key') + '"]')
        location = unquote(track.get('Location')).replace('file://localhost/', '')

        if (len(track.findall('TEMPO')) != 1):
            skipped += 1
            continue
        valid += 1
            
        bpm = float(track.find('TEMPO').get('Bpm'))
        idx = int(bpm)
        
        if idx not in tracks.keys():
            tracks[idx] = []
            
        tracks[idx].append(Track(
            trackid.get('Key'),
            location,
            bpm
            ))
    print("Found %d valid tracks (skipped %d)" % (valid, skipped))
    return tracks

def generate_samples(lib_xml_file='lib.xml', n=1000):
    n = int(n)

    if (os.path.isfile(lib_xml_file) == False):
        sys.exit('Library file not found: "%s"' % lib_xml_file)
    
    print('Loading library...')
    tracks = load_tracks(lib_xml_file)

    print('Generating Spectrograms...')
    for i in range(n):
        idx = np.random.choice(list(tracks.keys()))
        track = np.random.choice(tracks[idx])
        track.generate_random_specgram()
        progress = i / n * 100.0
        print('\r[%s%s] %3.1f%%' % ('=' * int(progress/2),
                                    ' ' * (50 - int(progress/2)),
                                    progress),
              end='', flush=True)
    
    print('\nDone.')

if __name__ == "__main__":
    try:
        os.makedirs("specgrams")
    except FileExistsError:
        # directory already exists
        pass
    
    argv = sys.argv[1:]
    generate_samples(*argv)
