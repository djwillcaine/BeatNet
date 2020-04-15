import os
import sys
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import lxml.etree as ET
from urllib.parse import unquote
import multiprocessing
import warnings
warnings.simplefilter("ignore")

BPM_VARIATION_AMOUNT = 0.1
BPM_VARIATION_PROB = 0.5
FRAME_RATE = 22050
N_MELS = 40
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
            bpm = round(float(tempo_nodes[i].get('Bpm')))
            start = float(tempo_nodes[i].get('Inizio'))
            if i == 0:
                start += np.ceil(bpm * BUFFER / 60) * 60 / bpm
            if i + 1 < len(tempo_nodes):
                end = float(tempo_nodes[i + 1].get('Inizio')) - SAMPLE_LENGTH
            else:
                end = int(track.get('TotalTime')) - BUFFER - SAMPLE_LENGTH
            if int(end) > int(start):
                parts.append({'bpm': bpm, 'start': start, 'end': end})

        if len(parts) > 0:
            tracks.append(Track(trackid.get('Key'), location, parts))
            valid += 1

    print("Found %d valid tracks" % valid)
    return tracks


def generate_random_specgram(track):
    frame_rate = FRAME_RATE
    part = np.random.choice(track.parts)
    bpm = part['bpm']
    
    # Perform random variations to the BPM (sometimes)
    if np.random.random() < BPM_VARIATION_PROB:
        bpm = np.random.randint(
            max(70, bpm - (BPM_VARIATION_AMOUNT * bpm)),
            min(198, bpm + (BPM_VARIATION_AMOUNT * bpm)))
        frame_rate = int(frame_rate * (bpm / part['bpm']))

    audio_file, _ = librosa.load(track.location, FRAME_RATE)
    audio, _ = librosa.effects.trim(audio_file)
    chunk_length = int(SAMPLE_LENGTH * frame_rate)

    # Choose a random chunk of chunk_length samples 
    i = np.random.randint(part['start'] * FRAME_RATE, part['end'] * FRAME_RATE)
    chunk = audio[i:i + chunk_length]

    # Convert to mel-scale
    mel = librosa.feature.melspectrogram(chunk, sr=frame_rate, n_fft=2048, n_mels=N_MELS, fmin=20, fmax=5000)
    mel_DB = librosa.power_to_db(mel, ref=np.max)

    # Plot specgram and save to file

    plt.figure(figsize=(2.56, 0.4)).add_axes([0, 0, 1, 1])
    plt.axis('off')
    plt.xlim(0, SAMPLE_LENGTH)
    plt.ylim(0, frame_rate / 2)
    
    librosa.display.specshow(mel_DB, x_axis="time", y_axis="mel", fmin=20, fmax=5000)
    plt.set_cmap('gray')

    filename = ('data/%d/%s-%s.png' % (bpm, track.trackid, i))
    create_dir('data/%d' % bpm)
    plt.savefig(filename)
    plt.close()

def generate_samples(lib_xml_file='lib.xml', n=1000):
    print('Loading library...')
    tracks = load_tracks(lib_xml_file)

    print('Generating Spectrograms...')
    print('\r[%s] 0.0%%' % (' ' * PROGRESS_BAR_SIZE), end='', flush=True)
    samples = []
    for i in range(n):
        samples.append(np.random.choice(tracks))

    pool = multiprocessing.Pool()
    for i, _ in enumerate(pool.imap_unordered(generate_random_specgram, samples)):
        progress = (i + 1) / n
        print('\r[%s%s] %3.1f%%' % (
            '=' * int(PROGRESS_BAR_SIZE * progress),
            ' ' * (PROGRESS_BAR_SIZE - int(PROGRESS_BAR_SIZE * progress)),
            progress * 100
            ), end='', flush=True)
    
    print('\nDone.')

def create_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        # directory already exists
        pass

if __name__ == "__main__":
    multiprocessing.freeze_support()
    create_dir("data")

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
