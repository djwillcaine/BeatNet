import os
import sys
import wave
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import lxml.etree as ET
from urllib.parse import unquote
import multiprocessing

FRAME_RATE = 22050
CHUNK_SIZE = FRAME_RATE * 2
BUFFER = FRAME_RATE * 5
PROGRESS_BAR_SIZE = 50

class Track:
    def __init__(self, trackid, location, bpm):
        self.trackid = trackid
        self.location = os.path.abspath(unquote(location))
        self.bpm = bpm
    
def load_tracks(lib_xml_file):
    tree = ET.parse(lib_xml_file)
    root = tree.getroot()
    tracks = []

    valid = 0
    skipped = 0
    for trackid in root.find('PLAYLISTS').find('NODE').find('NODE[@Name="BEATNET"]').iter('TRACK'):
        track = root.find('COLLECTION').find('TRACK[@TrackID="' + trackid.get('Key') + '"]')
        location = unquote(track.get('Location')).replace('file://localhost/', '')

        if (len(track.findall('TEMPO')) != 1):
            skipped += 1
            continue
        valid += 1
            
        tracks.append(Track(
            trackid.get('Key'),
            location,
            track.find('TEMPO').get('Bpm')
            ))
    print("Found %d valid tracks (skipped %d)" % (valid, skipped))
    return tracks

def generate_random_specgram(track):
    audio = AudioSegment.from_file(track.location)
    audio = audio.set_channels(1).set_frame_rate(FRAME_RATE)
    samples = audio.get_array_of_samples()
    start = np.random.randint(BUFFER, len(samples) - BUFFER)
    chunk = samples[start:start + CHUNK_SIZE]

    filename = ('specgrams/%s-%s-%s.png' % (track.trackid, start, track.bpm))

    plt.figure(figsize=(2.56, 0.32), frameon=False).add_axes([0, 0, 1, 1])
    plt.axis('off')
    plt.specgram(chunk, Fs = FRAME_RATE)
    plt.savefig(filename)
    plt.close()

def generate_samples(lib_xml_file='lib.xml', n=1000):
    print('Loading library...')
    tracks = load_tracks(lib_xml_file)

    print('Generating Spectrograms...')
    samples = []
    for i in range(n):
        samples.append(np.random.choice(tracks))

    pool = multiprocessing.Pool()
    for i, _ in enumerate(pool.imap_unordered(generate_random_specgram, samples)):
        progress = i / n
        print('\r[%s%s] %3.1f%%' % (
            '=' * int(PROGRESS_BAR_SIZE * progress),
            ' ' * (PROGRESS_BAR_SIZE - int(PROGRESS_BAR_SIZE * progress)),
            progress * 100
            ), end='', flush=True)
    
    print('\nDone.')

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    try:
        os.makedirs("specgrams")
    except FileExistsError:
        # directory already exists
        pass

    lib_path = ''
    for file in os.listdir('./'):
        if not file.endswith('.xml'): continue
        lib_path = file
    if not os.path.isfile(lib_path):
        input("Library file not found, press any key to exit...")
        sys.exit(0)

    n = int(input("How many images would you like to generate? "))
    generate_samples(lib_path, n)
    
    input("Finished. Press any key to close...")
    sys.exit(0)
