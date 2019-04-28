import tensorflow as tf
import os
import random
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAINING_DATA_DIR = r'specgrams'

def gen_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(256, 128, 3)),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def fetch_batch(batch_size=1000):
    all_image_paths = []
    all_image_labels = []

    data_root = pathlib.Path(TRAINING_DATA_DIR)
    files = data_root.iterdir()
    
    for file in files:
        file = str(file)
        all_image_paths.append(os.path.abspath(file))
        label = file[:-4].split('-')[2:]
        label = float(label[0]) / 200, int(label[1]) / 1000.0
        all_image_labels.append(label)

    def preprocess_image(path):
        img_raw = tf.io.read_file(path)
        image = tf.image.decode_png(img_raw, channels=3)
        image = tf.image.resize(image, [256, 128])
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
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds

ds = fetch_batch()
model = gen_model()
model.fit(ds, epochs=1, steps_per_epoch=10)
