import tensorflow as tf
import sys
import os
import random
import pathlib
from pprint import pprint

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAINING_DATA_DIR = 'specgrams'

def gen_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 16, 3)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mean_squared_error',
                  metrics=['mae'])

    return model


def fetch_batch(batch_size=256):
    all_image_paths = []
    all_image_labels = []

    data_root = pathlib.Path(TRAINING_DATA_DIR)
    files = data_root.iterdir()
    
    frequencies = {}
    
    for file in files:
        file = str(file)
        if file[-4:].upper() != '.PNG':
            continue
        
        label = file[:-4].split('-')[2:3]
        label = float(label[0]) / 200
        
        if label not in frequencies.keys():
            frequencies[label] = 0
            
        if frequencies[label] >= 20:
            continue
        
        frequencies[label] += 1
        all_image_paths.append(os.path.abspath(file))
        all_image_labels.append(label)

    pprint(frequencies)

    def preprocess_image(path):
        img_raw = tf.io.read_file(path)
        image = tf.image.decode_png(img_raw, channels=3)
        image = tf.image.resize(image, [256, 16])
        image /= 255.0
        return image

    def preprocess(path, label):
        return preprocess_image(path), label

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(all_image_labels)
    ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = ds.shuffle(buffer_size=len(os.listdir(TRAINING_DATA_DIR)))
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=1)
    
    return ds

def run(epochs, save_path):
    ds = fetch_batch()
    model = gen_model()
    model.fit(ds, epochs=int(epochs), steps_per_epoch=500)
    
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
