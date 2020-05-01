import os
import sys
import argparse
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import lxml.etree as ET
from urllib.parse import unquote
import multiprocessing
from functools import partial
import warnings
warnings.simplefilter("ignore")

FRAME_RATE = 22050  # Hz
N_MELS = 40         # Mels
SAMPLE_LENGTH = 10  # Seconds
BUFFER = 5          # Seconds
PROGRESS_BAR_SIZE = 50
AUGMENTATION_MULTIPLIERS = [0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1]

output_dir = 'data'
validation_split = 0.2
test_split = 0.1
limits = [80, 180]
linear = False


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


def generate_augmented_specgrams(output_dir, validation_split, test_split, limits, linear, track): 
    try:
        audio_file, _ = librosa.load(track.location, FRAME_RATE)
        audio, _ = librosa.effects.trim(audio_file)
        for m in AUGMENTATION_MULTIPLIERS:
            plot_and_save_specgram(track, audio, m, output_dir, validation_split, test_split, limits, linear)
    except:
        print("\nFailed to produce image for: " + track.location)


def plot_and_save_specgram(track, audio, augmentation_multiplier, output_dir, validation_split, test_split, limits, linear):
    part = np.random.choice(track.parts)
    bpm = round(part['bpm'] * augmentation_multiplier)

    if bpm < limits[0] or bpm > limits[1]:
        return

    frame_rate = int(FRAME_RATE * (bpm / part['bpm']))
    chunk_length = int(SAMPLE_LENGTH * frame_rate)

    # Choose a random chunk of chunk_length samples 
    i = np.random.randint(part['start'] * FRAME_RATE, part['end'] * FRAME_RATE)
    chunk = audio[i:i + chunk_length]

    # Scale data linearly or by Mel-scale
    if linear:
        data = librosa.amplitude_to_db(np.abs(librosa.stft(chunk)), ref=np.max)
    else:
        mel = librosa.feature.melspectrogram(chunk, sr=frame_rate, n_fft=2048, n_mels=N_MELS, fmin=20, fmax=5000)
        data = librosa.power_to_db(mel, ref=np.max)

    # Configure plot
    plt.figure(figsize=(2.56, 0.4)).add_axes([0, 0, 1, 1])
    plt.axis('off')
    plt.xlim(0, SAMPLE_LENGTH)
    plt.ylim(0, frame_rate / 2)
    
    # Plot specgram
    librosa.display.specshow(data, cmap='gray_r', x_axis="time", y_axis="log")

    # Randomly distribute training/validation/test
    sub_dir = 'training'
    r = np.random.rand()
    if r < validation_split:
        sub_dir = 'validation'
    elif r < validation_split + test_split:
        sub_dir = 'test'

    # Save to file
    create_dir(output_dir + '/%s/%d' % (sub_dir, bpm))
    filename = (output_dir + '/%s/%d/%s-%s.png' % (sub_dir, bpm, track.trackid, i))
    plt.savefig(filename)
    plt.close()


def generate_data(lib_xml_file, n, output_dir, validation_split, test_split, limits, linear):
    print('Loading library...')
    tracks = load_tracks(lib_xml_file)

    print('Generating Spectrograms...')
    print('\r[%s] 0.0%%' % (' ' * PROGRESS_BAR_SIZE), end='', flush=True)
    samples = []
    for i in range(n):
        samples.append(np.random.choice(tracks))

    func = partial(generate_augmented_specgrams, output_dir, validation_split, test_split, limits, linear)
    pool = multiprocessing.Pool()
    for i, _ in enumerate(pool.imap_unordered(func, samples)):
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

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--lib_path', default='lib.xml')
    parser.add_argument('-n', '--num-tracks', type=int, default=1000)
    parser.add_argument('-o', '--output-dir', default='data')
    parser.add_argument('-v', '--validation-split', type=float, default=0.2)
    parser.add_argument('-t', '--test-split', type=float, default=0.1)
    parser.add_argument('-r', '--range', default='80-180')
    parser.add_argument('-l', '--linear', action='store_true')

    args = parser.parse_args()

    generate_data(
        args.lib_path,
        args.num_tracks,
        args.output_dir,
        args.validation_split,
        args.test_split,
        [int(i) for i in args.range.split('-')],
        args.linear)