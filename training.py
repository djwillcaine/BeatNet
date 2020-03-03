import tensorflow as tf
import sys
import os
import random
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAINING_DATA_DIR = 'specgrams'

def gen_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 64, 1)),
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


def fetch_ds(dir_name='data/training', batch_size=128):
    all_image_paths = []
    all_image_labels = []

    data_root = pathlib.Path(dir_name)
    files = data_root.iterdir()
    
    for file in files:
        file = str(file)
        if file[-4:].upper() != '.PNG':
            continue
        
        label = file[:-4].split('-')[2]
        label = float(label)
        
        all_image_paths.append(os.path.abspath(file))
        all_image_labels.append(label)

    def preprocess_image(path):
        img_raw = tf.io.read_file(path)
        image = tf.image.decode_png(img_raw, channels=1)
        image = tf.image.resize(image, [256, 64])
        image /= 255.0
        return image

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(all_image_labels)
    ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = ds.shuffle(buffer_size=len(os.listdir(dir_name)))
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=1)

    print(all_image_labels)
    
    return ds

def run(epochs=5):
    train_set = fetch_ds('data/training', batch_size=256)
    val_set = fetch_ds('data/validation', batch_size=128)
    model = gen_model()

    model.fit(train_set,
              epochs=int(epochs),
              steps_per_epoch=500,
              validation_data=val_set,
              validation_steps=100)
    
    model.save('model.h5')

    test_set = fetch_ds('data/test', batch_size=128)
    results = model.evaluate(test_set, steps=100)
    print('Test loss, test acc:', results)

if __name__ == "__main__":
    argv = sys.argv[1:]
    run(*argv)
