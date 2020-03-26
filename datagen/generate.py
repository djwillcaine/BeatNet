import os
import sys
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import lxml.etree as ET
from urllib.parse import unquote
import multiprocessing

import warnings

BPM_VARIATION_AMOUNT = 0.1
BPM_VARIATION_PROB = 0.2
FRAME_RATE = 22050
SAMPLE_LENGTH = 2
BUFFER = 5
PROGRESS_BAR_SIZE = 50

class Track:
    def __init__(self, trackid, location, parts):
        self.trackid = trackid
        self.location = os.path.abspath(unquote(location))
        self.parts = parts
    
def load_tracks(lib_xml_file):
    tree = ET.parse(lib_xml_file)
    root = tree.getroot()
    tracks = []

    valid = 0
    
    for trackid in root.find('PLAYLISTS').find('NODE').find('NODE[@Name="BEATNET"]').iter('TRACK'):
        track = root.find('COLLECTION').find('TRACK[@TrackID="' + trackid.get('Key') + '"]')
        location = unquote(track.get('Location')).replace('file://localhost/', '')
        tempo_nodes = track.findall('TEMPO')
        parts = []

        for i in range(len(tempo_nodes)):
            bpm = float(tempo_nodes[i].get('Bpm'))
            start = float(tempo_nodes[i].get('Inizio'))
            if i == 0:
                start += np.ceil(bpm * BUFFER / 60) * 60 / bpm
            if i + 1 < len(tempo_nodes):
                end = float(tempo_nodes[i + 1].get('Inizio')) - SAMPLE_LENGTH
            else:
                end = int(track.get('TotalTime')) - BUFFER - SAMPLE_LENGTH
            if end - start > 0:
                parts.append({'bpm': bpm, 'start': start, 'end': end})

        if len(parts) > 0:
            tracks.append(Track(trackid.get('Key'), location, parts))
            valid += 1

    print("Found %d valid tracks" % valid)
    return tracks

def generate_random_specgram(track):
    # Perform random variations to the BPM (sometimes)
    frame_rate = FRAME_RATE
    part = np.random.choice(track.parts)
    bpm = part['bpm']
    if np.random.random() < BPM_VARIATION_PROB:
        variation = 1 - BPM_VARIATION_AMOUNT + (
            np.random.random() * BPM_VARIATION_AMOUNT * 2)
        bpm *= variation
        bpm = round(bpm, 2)
        frame_rate *= (bpm / part['bpm'])
        frame_rate = int(frame_rate)

    # Read audio data from file
    audio = AudioSegment.from_file(track.location)
    audio = audio.set_channels(1).set_frame_rate(FRAME_RATE)
    samples = audio.get_array_of_samples()
    chunk_length = int(SAMPLE_LENGTH * frame_rate)

    # Choose a chunk starting on a random beat
    x = np.random.randint((part['end'] - part['start']) * part['bpm'] / 60)
    start = int(FRAME_RATE * (part['start'] + x * 60 / part['bpm']))
    chunk = samples[start:start + chunk_length]

    # Plot specgram and save to file
    filename = ('specgrams/%s-%s-%s.png' % (track.trackid, start, bpm))
    plt.figure(figsize=(2.56, 0.64), frameon=False).add_axes([0, 0, 1, 1])
    plt.axis('off')
    plt.specgram(chunk, cmap='gray', Fs=frame_rate)
    plt.xlim(0, SAMPLE_LENGTH)
    plt.ylim(0, frame_rate / 2)
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
    warnings.simplefilter('error', UserWarning)
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
