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
    image_paths, x, true_values = fetch_dataset(ds_dir, limits)
    results = model.predict(x)

    # Format results
    if categorical:
        pred_values = tf.math.argmax(results, axis=1)
        pred_values = [int(b + limits[0]) for b in pred_values]
    else:
        pred_values = tf.squeeze(results).numpy()

    # Calculate MSE and MAE
    mse = tf.keras.losses.MSE(true_values, pred_values)
    mae = tf.keras.losses.MAE(true_values, pred_values)
    
    # Calculate accuracy1 and accuracy2
    total = len(pred_values)
    acc1 = acc2 = 0
    factors = (1/3, 1/2, 2, 3)
    for t_val, val in zip(true_values, pred_values):
        if t_val * 0.96 <= val and t_val * 1.04 >= val:
            acc1 += 1
            acc2 += 1
        else:
            for f in factors:
                if t_val * 0.96 <= val * f and t_val * 1.04 >= val * f:
                    acc2 += 1
                    continue
    acc1 /= total
    acc2 /= total

    # Output results to file/console
    if write_to_file:
        # Log MSE, MAE, accuray1, and accuracy 2 in eval/results.csv
        file = open(os.path.join('eval', 'results.csv'), 'a')
        file.write('%s,%s,%s,%s,%f,%f,%f,%f\n' % (
            model_info[0],
            model_info[1].capitalize(),
            model_info[2].capitalize(),
            model_info[3].capitalize(),
            mse.numpy(),
            mae.numpy(),
            acc1,
            acc2))
        file.close()

        # Log verbose list of predicitons in eval/model_name.csv
        file = open(os.path.join('eval', '%s.csv' % model.name), 'w')
        file.write('Image Path,True Value,Predicted Value\n')
        for vals in zip(image_paths, true_values, pred_values):
            file.write('%s,%d,%f\n' % vals)
        file.close()
    else:
        print('MSE: ', mse.numpy())
        print('MAE: ', mae.numpy())
        print('Accuracy1: ', acc1)
        print('Accuracy2: ', acc2)


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

    # Build image dataset from paths
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(img_to_tensor)

    return image_paths, image_ds, image_bpms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path',
        help='File path to the model to evaluate.')
    parser.add_argument('-w', '--write-to-file', action='store_true',
        help='Specifying this will write the results to file instead of outputting to console.')
    parser.add_argument('-d', '--data-dir',
        help='The directory to use for the test set. Will be inferred if not specified.')

    args = parser.parse_args()
    evaluate(args.model_path, args.write_to_file, args.data_dir)