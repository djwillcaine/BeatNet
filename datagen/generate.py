import os
import sys
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import lxml.etree as ET
from urllib.parse import unquote
import multiprocessing

BPM_VARIATION_AMOUNT = 0.1
BPM_VARIATION_PROB = 0.2
FRAME_RATE = 22050
SAMPLE_LENGTH = 2
BUFFER = 5
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
            float(track.find('TEMPO').get('Bpm'))
            ))
    print("Found %d valid tracks (skipped %d)" % (valid, skipped))
    return tracks

def generate_random_specgram(track):
    # Perform random variations to the BPM (sometimes)
    frame_rate = FRAME_RATE
    bpm = track.bpm
    if np.random.random() < BPM_VARIATION_PROB:
        variation = 1 - BPM_VARIATION_AMOUNT + (
            np.random.random() * BPM_VARIATION_AMOUNT * 2)
        bpm *= variation
        bpm = round(bpm, 2)
        frame_rate *= (bpm / track.bpm)
        frame_rate = int(frame_rate)

    # Read audio data from file
    audio = AudioSegment.from_file(track.location)
    audio = audio.set_channels(1).set_frame_rate(FRAME_RATE)
    samples = audio.get_array_of_samples()
    chunk_length = int(SAMPLE_LENGTH * frame_rate)
    start = np.random.randint(
        BUFFER * FRAME_RATE,
        len(samples) - (BUFFER * FRAME_RATE) - chunk_length)
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

def choose_track(tracks):
    track = np.random.choice(tracks)
    if track.bpm < 120 or track.bpm > 130:
        return choose_track(tracks)
    return track

def generate_samples(lib_xml_file='lib.xml', n=1000):
    print('Loading library...')
    tracks = load_tracks(lib_xml_file)

    print('Generating Spectrograms...')
    samples = []
    for i in range(n):
        samples.append(choose_track(tracks))

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
