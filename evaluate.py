import os
import pathlib
import argparse

import tensorflow as tf
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 40

def evaluate(model_path, write_to_file, ds_dir):

    # Load model and define variables
    model = tf.keras.models.load_model(os.path.abspath(model_path))
    model_info = model.name.split('.')

    categorical = (model_info[2] == 'classification')
    limits = model_info[-1].split('-')
    limits = (int(limits[0]), int(limits[1]))

    if (ds_dir == None):
        ds_dir = os.path.join('data', model_info[0], 'test')

    # Fetch test dataset and predict results
    x, true_bpms, true_cats = fetch_dataset(ds_dir, limits)
    results = model.predict(x)

    # Format results
    if categorical:
        pred_cats = results
        pred_bpms = tf.math.argmax(results, axis=1)
        pred_bpms = [int(b + limits[0]) for b in pred_bpms]
    else:
        pred_bpms = np.around(tf.squeeze(results).numpy())
        indexes = [int(b - limits[0]) for b in pred_bpms]
        pred_cats = tf.one_hot(indexes, limits[1] - limits[0] + 1)

    # Calculate metrics
    mse = tf.keras.losses.MSE(true_bpms, pred_bpms)
    mae = tf.keras.losses.MAE(true_bpms, pred_bpms)
    cce = tf.keras.losses.CategoricalCrossentropy()(true_cats, pred_cats)

    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(true_bpms, pred_bpms)
    accuracy = accuracy.result()

    # Output results to file/console
    if write_to_file:
        file = open('results.csv', 'a')
        file.write('%s,%s,%s,%s,%s,%f,%f,%f,%f\n' % (
            model_info[0],
            model_info[1].capitalize(),
            model_info[2].capitalize(),
            model_info[3].capitalize(),
            model_info[4],
            mse.numpy(),
            mae.numpy(),
            accuracy.numpy(),
            cce.numpy()))
    else:
        print('MSE: ', mse.numpy())
        print('MAE: ', mae.numpy())
        print('CCE: ', cce.numpy())
        print('Accuracy: ', accuracy.numpy())


def img_to_tensor(img_path):
    img_raw = tf.io.read_file(img_path)
    image = tf.image.decode_png(img_raw, channels=1)
    image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
    image /= 255.0
    return tf.expand_dims(image, 0)

def fetch_dataset(ds_dir, limits=(80, 180)):
    if not os.path.isdir(ds_dir):
        print("Dataset directory '%s' not found" % ds_dir)
        return

    image_paths = []
    image_bpms = []

    # Iterate over dataset directory
    for subdir in pathlib.Path(ds_dir).iterdir():
        label = int(subdir.stem)
        if label < limits[0] or label > limits[1]:
            continue

        for file in subdir.iterdir():
            if file.suffix != ".png":
                continue
            image_paths.append(os.path.abspath(str(file)))
            image_bpms.append(label)


    n_classes = limits[1] - limits[0] + 1
    indexes = [b - limits[0] for b in image_bpms]
    categorical = tf.keras.utils.to_categorical(indexes, n_classes)

    # Build dataset from paths/labels
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(img_to_tensor)

    return image_ds, image_bpms, categorical


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path')
    parser.add_argument('-w', '--write-to-file', action='store_true')
    parser.add_argument('-d', '--data-dir')

    args = parser.parse_args()
    evaluate(args.model_path, args.write_to_file, args.data_dir)