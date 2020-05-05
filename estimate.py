import os
import re
import argparse
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SAMPLE_RATE = 22050

def estimate(model_path, image_path, track_path, sample_length):
    model = tf.keras.models.load_model(model_path)

    if track_path != None:
        # Load audio
        audio_file, _ = librosa.load(track_path, SAMPLE_RATE)
        audio, _ = librosa.effects.trim(audio_file)
        duration = librosa.get_duration(y=audio, sr=SAMPLE_RATE)

        # Choose random sample
        i = np.random.randint(duration * SAMPLE_RATE)
        chunk = audio[i:i + sample_length * SAMPLE_RATE]
        mel = librosa.feature.melspectrogram(chunk, sr=SAMPLE_RATE, n_fft=2048, n_mels=40, fmin=20, fmax=5000)
        data = librosa.power_to_db(mel, ref=np.max)

        # Plot Mel-spectrogram
        plt.figure(figsize=(2.56, 0.4)).add_axes([0, 0, 1, 1])
        plt.axis('off')
        plt.xlim(0, sample_length)
        plt.ylim(0, SAMPLE_RATE / 2)
        librosa.display.specshow(data, cmap='gray_r', x_axis="time", y_axis="mel")

        # Save to temp file
        create_dir('temp')
        plt.savefig('temp/image.png')
        plt.close()

    elif image_path == 'temp/image.png':
        print('Please specify either an image or track path to estimate.')
        return
   
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [256, 40])
    image /= 255.0
    image = tf.expand_dims(image, 0)

    result = model.predict(image)[0]

    # Format output
    if len(result) != 1:
        result = tf.math.argmax(result, axis=0).numpy()
        m = re.search('\.(\d+)-\d+\.', model_path)
        if m == None:
            result = 'Class(%s)' % result
        else:
            result = result + int(m.group(1))
    else:
        result = result[0]

    print('Estimate: %s BPM' % result)

def create_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        # directory already exists
        pass

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path', default='model.h5',
        help='File path to the model to use for estimation.')
    parser.add_argument('-i', '--image-path', default='temp/image.png',
        help='File path to a spectrogram to estimate (if not specifying an audio file).')
    parser.add_argument('-t', '--track-path',
        help='File path to an audio file to estimate (if not specifying an image file).')
    parser.add_argument('-l', '--sample-length', type=int, default=10,
        help='The length (in seconds) of the audio sample to use, if specifying an audio file. Must match what the model was trained with.')

    args = parser.parse_args()

    estimate(args.model_path, args.image_path, args.track_path, args.sample_length)