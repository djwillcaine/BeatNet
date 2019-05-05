import tensorflow as tf
import sys
import os
import random
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAINING_DATA_DIR = 'specgrams'

def gen_model():
    inputs = tf.keras.layers.Input(shape=(256, 128, 3), dtype=tf.float32)
    d1 = tf.keras.layers.Dense(256, activation='relu')(inputs)
    d2 = tf.keras.layers.Dense(2)(d1)
    bpm, inizio = tf.keras.layers.Lambda(tf.unstack, arguments=dict(axis=-1))(d2)
    model = tf.keras.models.Model(inputs=inputs, outputs=[bpm, inizio])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mean_squared_error',
                  metrics=['mae'])

    return model


def fetch_batch(batch_size=256):
    all_image_paths = []
    all_image_bpms = []
    all_image_inizios = []

    data_root = pathlib.Path(TRAINING_DATA_DIR)
    files = data_root.iterdir()
    
    for file in files:
        file = str(file)
        if file[-4:].upper() != '.PNG':
            continue
        
        all_image_paths.append(os.path.abspath(file))
        bpm, inizio = file[:-4].split('-')[2:]
        bpm = float(bpm) / 200
        inizio = int(inizio) / 1000
        
        all_image_bpms.append(bpm)
        all_image_inizios.append(inizio)

    def preprocess_image(path):
        img_raw = tf.io.read_file(path)
        image = tf.image.decode_png(img_raw, channels=3)
        image = tf.image.resize(image, [256, 128])
        image /= 255.0
        return image

    def preprocess(path, label):
        return preprocess_image(path), tf.unstack(label, axis=-1)

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    bpm_ds = tf.data.Dataset.from_tensor_slices(all_image_bpms)
    inizio_ds = tf.data.Dataset.from_tensor_slices(all_image_inizios)
    
    ds = tf.data.Dataset.zip((image_ds, (bpm_ds, inizio_ds)))
    ds = ds.shuffle(buffer_size=len(os.listdir(TRAINING_DATA_DIR)))
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=1)
    
    return ds

def run(epochs, save_path):
    ds = fetch_batch()
    model = gen_model()
    model.fit(ds, epochs=int(epochs), steps_per_epoch=50)

    model.save('temp/' + save_path)

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 2:
        exit('Program requires 2 arguments: number of epochs and save path.')
        
    try:
        os.makedirs("temp")
    except FileExistsError:
        pass
    
    run(*argv)
